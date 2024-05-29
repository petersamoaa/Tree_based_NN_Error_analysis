import argparse
import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from data_utils import find_java_files, parse_java_code_to_ast
from models.code2vec.config import Config as Code2vecConfig
from models.code2vec.net import Code2vecNet
from models.code2vec.prepare_data import prepare_data as code2vec_prepare_data
from models.code2vec.trainer import trainer as code2vec_trainer
from models.tbcc.prepare_data import prepared_data as tbcc_prepared_data
from models.tbcc.trainer import trainer as tbcc_trainer
from models.tbcc.transformer import TransformerModel
from models.transformers.prepare_data import prepared_data as transformer_prepared_data
from models.transformers.trainer import trainer as transformer_trainer
# from models.dou_transformer.dou_transformer import DualTransformerWithCrossAttention
# from models.dou_transformer.prepare_data import prepared_data as dou_transformer_prepared_data
# from models.dou_transformer.trainer import trainer as dou_transformer_trainer
from models.transformers.transformers import ASTTransformer, DualTransformerWithCrossAttention
from models.tree_cnn.embedding import TreeCNNEmbedding
from models.tree_cnn.prepare_data import prepare_nodes as tree_cnn_prepare_nodes
from models.tree_cnn.prepare_data import prepare_trees as tree_cnn_prepare_trees
from models.tree_cnn.tbcnn import TreeConvNet
from models.tree_cnn.trainer import node_trainer as tree_cnn_node_trainer
from models.tree_cnn.trainer import trainer as tree_cnn_trainer
from utils import AttrDict, remove_comments

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_data(data_dir, do_log=True, data_info="data_info.csv"):
    java_files = find_java_files(data_dir)
    data_info = pd.read_csv(os.path.join(data_dir, data_info))
    data_info["Test case"] = data_info["Test case"].apply(lambda x: x.split("/")[-1])
    data = []

    for java_file in tqdm(java_files):
        file_name, full_path, file_type = java_file
        row = data_info[data_info["Test case"] == file_name].to_dict("records")

        if len(row) > 0:
            row = row[0]

            with open(os.path.join(full_path), "r", encoding="utf-8") as f:
                java_code = remove_comments(f.read())
                tree = parse_java_code_to_ast(java_code, logger, jid=full_path)

            data_dir = Path(data_dir)
            full_path = Path(full_path)
            relative_path = full_path.relative_to(data_dir)

            if tree:
                y = row["Runtime in ms"]
                data.append({
                    "name": file_name,
                    "path": str(full_path),
                    "relative_path": str(relative_path),
                    "category": str(relative_path.parts[1]),
                    "type": file_type,
                    "category_type": f"{str(relative_path.parts[1])}-{file_type}",
                    "code": java_code.strip(),
                    "tree": tree,
                    "y": math.log(y) if do_log and y > 0 else y,
                    "runtime": y
                })

    return java_files, data


def main(args):
    output_dir = args.output_dir


    aggregated_data = []
    aggregated_java_files = []

    for train_data_dir in args.train_data_dir:
        java_files, data = read_data(train_data_dir, do_log=args.do_log)
        aggregated_java_files.extend(java_files)
        aggregated_data.extend(data)
    
    data = aggregated_data

    if args.model_path and os.path.exists(args.model_path):
        logger.info(f"Loading the model scaler {args.model_path}")
        y_scaler = torch.load(os.path.join(args.model_path, "y_scaler.bin"))
    else:
        logger.info(f"Starting the model scaler")
        y_scaler = MinMaxScaler(feature_range=(0, 1))
        y_scaler = y_scaler.fit(list(map(lambda x: [x["y"]], data)))
    
    new_y = y_scaler.transform(list(map(lambda x: [x["y"]], data)))
    for i, row in enumerate(data):
        row["y_prime"] = row["y"]
        row["y"] = new_y[i][0]

    java_files = aggregated_java_files
    logger.info(f"Found {len(data)} trainingset for this experiment out of {len(java_files)}!")

    if not args.test_data_dir or not os.path.exists(args.test_data_dir):
        train, test = train_test_split(data, test_size=args.test_size, random_state=args.seed, stratify=list(map(lambda x: x["category"], data)))
        logger.info(f"Train {len(train)} Test {len(test)} for split test_size={args.test_size}")
        same_space = True
    else:
        train = data
        java_files, test = read_data(args.test_data_dir, do_log=args.do_log)
        logger.info(f"Found {len(test)} testsets for this experiment out of {len(java_files)}")
        logger.info(f"Train {len(train)} Test {len(test)}")
        
    
    if args.train_size > 0.0:
        # Adjust train_size to reflect the desired proportion of the original dataset size.
        # This calculates the effective training size to use in train_test_split to achieve the desired proportion from the original dataset.
        adjusted_train_size = args.train_size / (1 - args.test_size)
        
        # Ensure the adjusted_train_size does not exceed 1.0, which can happen if args.train_size is set to a value
        # that's too high relative to args.test_size.
        adjusted_train_size = min(1.0, max(0.0, adjusted_train_size))

        if adjusted_train_size >= 1.0:
            raise ValueError("It's already been trained on the whole training set!")

        _, train = train_test_split(train, test_size=adjusted_train_size, random_state=args.seed, stratify=list(map(lambda x: x["category"], train)))
        logger.info(f"{len(_) + len(train)} => Sample Training ({args.train_size * 100}%) {len(train)}")

    # if args.model_path and os.path.exists(args.model_path):
    #     logger.info(f"Loading the model scaler {args.model_path}")
    #     y_scaler = torch.load(os.path.join(args.model_path, "y_scaler.bin"))
    # else:
    #     logger.info(f"Starting the model scaler")
    #     y_scaler = MinMaxScaler(feature_range=(0, 1))
    #     y_scaler.fit(list(map(lambda x: [x["y"]], train)))


    for model in args.train_on:
        model = model.lower()
        if model == "tree_cnn_emb":
            logger.info(f"starting on {model} ...")
            nodes, node_samples = tree_cnn_prepare_nodes(data=list(map(lambda x: x["tree"], data)), per_node=args.per_node, limit=args.limit)
            logger.info(f"We have {len(nodes)} nodes: {nodes.keys()}")

            if output_dir:
                torch.save(nodes, os.path.join(output_dir, f'nodes.bin'))
                torch.save(node_samples, os.path.join(output_dir, f'node_samples.bin'))

            tree_cnn_embedding = TreeCNNEmbedding(num_classes=len(nodes.keys()), num_feats=args.num_feats, hidden_size=args.hidden_size).to(device)
            if args.model_path and os.path.exists(args.model_path):
                logger.info(f"Loading the model from {args.model_path}")
                tree_cnn_embedding = tree_cnn_embedding.load_state_dict(torch.load(args.model_path))
            
            node_map = {node: i for i, node in enumerate(nodes)}
            tree_cnn_embedding = tree_cnn_node_trainer(
                node_samples, tree_cnn_embedding,
                node_map=node_map,
                device=device, lr=args.lr, batch_size=args.batch_size, epochs=args.repr_epochs, checkpoint=args.checkpoint, output_dir=output_dir)

        if model == "tree_cnn":
            if output_dir:
                torch.save(y_scaler, os.path.join(output_dir, f'y_scaler.bin'))
                torch.save(train, os.path.join(output_dir, f'train.bin'))
                torch.save(test, os.path.join(output_dir, f'test.bin'))

            logger.info(f"starting on {model} ...")

            node_map = torch.load(args.node_map_path)
            embeddings = torch.load(args.embeddings_path)

            train_trees = tree_cnn_prepare_trees(data=train, minsize=args.minsize, maxsize=args.maxsize)
            test_trees = tree_cnn_prepare_trees(data=test, minsize=args.minsize, maxsize=args.maxsize)

            # train_trees = tree_cnn_prepare_trees(data=list(map(lambda x: x["tree"], train)), minsize=-1, maxsize=-1)
            # train_trees = [train_trees, list(map(lambda x: x["y"], train))]
            # test_trees = tree_cnn_prepare_trees(data=list(map(lambda x: x["tree"], test)), minsize=-1, maxsize=-1)
            # test_trees = [test_trees, list(map(lambda x: x["y"], test))]
            logger.info(f"Train size: {len(train_trees)}, Test size: {len(test_trees)}")

            model = TreeConvNet(feature_size=len(embeddings[0]), label_size=1, num_conv=args.num_conv, output_size=args.conv_hidden_size).to(device)
            if args.model_path and os.path.exists(args.model_path):
                logger.info(f"Loading the model from {args.model_path}")
                model.load_state_dict(torch.load(os.path.join(args.model_path, "model.pth")))

            tree_cnn_trainer(
                model,
                train_trees=train_trees,
                test_trees=test_trees,
                y_scaler=None,
                embeddings=embeddings,
                embed_lookup=node_map,
                device=device,
                lr=args.lr, batch_size=args.batch_size, epochs=args.epochs, checkpoint=args.checkpoint, output_dir=output_dir
            )
        elif model == "code2vec":
            if output_dir:
                torch.save(y_scaler, os.path.join(output_dir, f'y_scaler.bin'))
                torch.save(train, os.path.join(output_dir, f'train.bin'))
                torch.save(test, os.path.join(output_dir, f'test.bin'))

            logger.info(f"starting on extractign representation based on {model} ...")
            code2vec_config = Code2vecConfig(set_defaults=True, load_from_args=True, verify=True, args=args)
            train_data, test_data = code2vec_prepare_data(train, code2vec_config, test=test)
            logger.info(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

            model = Code2vecNet(feature_size=train_data[0]["representation"].shape[1]).to(device)
            if args.model_path and os.path.exists(args.model_path):
                logger.info(f"Loading the model from {args.model_path}")
                model.load_state_dict(torch.load(os.path.join(args.model_path, "model.pth")))

            code2vec_trainer(
                model,
                train=train_data,
                test=test_data,
                y_scaler=None,
                device=device,
                lr=args.lr, batch_size=args.batch_size, epochs=args.epochs, checkpoint=args.checkpoint, output_dir=output_dir
            )
        # elif model == "transformer_tree":
        #     # _output_dir = os.path.join(args.output_dir, f"{model}_{datetime.now().strftime('%Y-%m-%dT%H_%M_%S')}")
        #     output_dir = os.path.join(args.output_dir, model)
        #     os.makedirs(output_dir, exist_ok=True)

        #     if output_dir:
        #         torch.save(y_scaler, os.path.join(output_dir, f'y_scaler.bin'))
        #         torch.save(train, os.path.join(output_dir, f'train.bin'))
        #         torch.save(test, os.path.join(output_dir, f'test.bin'))

        #     logger.info(f"starting on extractign representation based on {model} ...")
        #     train_data, test_data, vocabulary, inverse_vocabulary = tbcc_prepared_data(train, max_seq_length=args.max_seq_length, test_records=test)

        #     model = TransformerModel(
        #         vocab_size=len(vocabulary) + 1,
        #         max_seq_length=args.max_seq_length,
        #         embed_dim=args.embed_dim,
        #         num_heads=args.num_heads,
        #         ff_dim=args.ff_dim,
        #         num_transformer_blocks=args.num_transformer_blocks,
        #         num_classes=1,
        #     ).to(device)
        #     tbcc_trainer(
        #         model,
        #         train=train_data,
        #         test=test_data,
        #         y_scaler=y_scaler,
        #         device=device,
        #         max_seq_length=args.max_seq_length,
        #         lr=args.lr, batch_size=args.batch_size, epochs=args.epochs, checkpoint=args.checkpoint, output_dir=output_dir
        #     )
        elif model == "1_transformer":
            if output_dir:
                torch.save(y_scaler, os.path.join(output_dir, f'y_scaler.bin'))
                torch.save(train, os.path.join(output_dir, f'train.bin'))
                torch.save(test, os.path.join(output_dir, f'test.bin'))

            logger.info(f"starting on extractign representation based on {model} ...")
            train_data, test_data, tokenizer = transformer_prepared_data(
                train, max_seq_length=args.max_seq_length, test_records=test, model_name_or_path="google-bert/bert-base-cased",
            )

            model = ASTTransformer(
                vocab_size=len(tokenizer), 
                d_model=args.d_model, 
                n_head=args.n_head, 
                d_ff=args.d_ff, 
                n_layer=args.n_layer, 
                max_seq_len=args.max_seq_length, 
                drop=args.drop,
            ).to(device)
            if args.model_path and os.path.exists(args.model_path):
                logger.info(f"Loading the model from {args.model_path}")
                model.load_state_dict(torch.load(os.path.join(args.model_path, "model.pth")))

            transformer_trainer(
                model,
                train=train_data,
                test=test_data,
                y_scaler=None,
                device=device,
                max_seq_length=args.max_seq_length,
                lr=args.lr, batch_size=args.batch_size, epochs=args.epochs, checkpoint=args.checkpoint, output_dir=output_dir
            )
        elif model == "2_transformer":
            # _output_dir = os.path.join(args.output_dir, f"{model}_{datetime.now().strftime('%Y-%m-%dT%H_%M_%S')}")
            # output_dir = os.path.join(args.output_dir, model)
            # os.makedirs(output_dir, exist_ok=True)

            if output_dir:
                torch.save(y_scaler, os.path.join(output_dir, f'y_scaler.bin'))
                torch.save(train, os.path.join(output_dir, f'train.bin'))
                torch.save(test, os.path.join(output_dir, f'test.bin'))

            logger.info(f"starting on extractign representation based on {model} ...")
            train_data, test_data, tokenizer = transformer_prepared_data(
                train, max_seq_length=args.max_seq_length, test_records=test, model_name_or_path="google-bert/bert-base-cased"
            )

            model = DualTransformerWithCrossAttention(
                nl_vocab_size=len(tokenizer), ast_vocab_size=len(tokenizer), 
                d_model=args.d_model, 
                n_head=args.n_head, 
                d_ff=args.d_ff, 
                n_layer=args.n_layer, 
                max_seq_len=args.max_seq_length, 
                drop=args.drop,
            ).to(device)
            if args.model_path and os.path.exists(args.model_path):
                logger.info(f"Loading the model from {args.model_path}")
                model.load_state_dict(torch.load(os.path.join(args.model_path, "model.pth")))

            transformer_trainer(
                model,
                train=train_data,
                test=test_data,
                y_scaler=None,
                device=device,
                max_seq_length=args.max_seq_length,
                lr=args.lr, batch_size=args.batch_size, epochs=args.epochs, checkpoint=args.checkpoint, output_dir=output_dir
            )

if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    # parser.add_argument('--train_data_dir', type=str, help='provide dataset directory', required=True)
    parser.add_argument('--train_data_dir', nargs='+', type=str, help='provide dataset directory/directories', required=True)
    parser.add_argument('--test_data_dir', type=str, help='provide dataset directory', required=False)
    parser.add_argument('--do_log', action='store_true', required=False, help="Compute the log of the target")
    parser.add_argument('--train_size', type=float, default=0.0, help="proportion of dataset to training and testing sets")
    parser.add_argument('--test_size', type=float, default=0.2, help="proportion of dataset to training and testing sets")
    parser.add_argument('--train_on', nargs='+', help='train on these models', required=True)

    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--repr_epochs', type=int, default=10, help='number of epochs to train representation (default: 10)')
    parser.add_argument('--output_dir', type=str, default="", help='provide output directory')
    parser.add_argument('--model_path', type=str, default="", help='provide model directory')

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size for training (default: 8)')
    parser.add_argument('--checkpoint', type=int, default=-1, help='number of checkpoint to log (default: -1)')

    # TreeCNN
    parser.add_argument('--node_map_path', type=str, default="", help='node_map file path')
    parser.add_argument('--embeddings_path', type=str, default="", help='embeddings file path')
    parser.add_argument('--minsize', type=int, default=-1, help='Minimum size for tree.')
    parser.add_argument('--maxsize', type=int, default=-1, help='Maximum size for tree.')
    parser.add_argument('--limit', type=int, default=-1, help='Limit the number of samples. Set to -1 for no limit.')
    parser.add_argument('--per_node', type=int, default=-1, help='Limit samples per node. Set to -1 for no limit.')
    parser.add_argument('--num_feats', type=int, default=100, help='')
    parser.add_argument('--hidden_size', type=int, default=100, help='')
    parser.add_argument('--num_conv', type=int, default=1, help='')
    parser.add_argument('--conv_hidden_size', type=int, default=100, help='')

    # Code2Vec
    parser.add_argument("-d", "--data", dest="data_path", help="path to preprocessed dataset", required=False)
    parser.add_argument("-te", "--test", dest="test_path", help="path to test file", metavar="FILE", required=False, default='')
    parser.add_argument("-s", "--save", dest="save_path", help="path to save the model file", metavar="FILE", required=False)
    parser.add_argument("-w2v", "--save_word2v", dest="save_w2v", help="path to save the tokens embeddings file", metavar="FILE", required=False)
    parser.add_argument("-t2v", "--save_target2v", dest="save_t2v", help="path to save the targets embeddings file", metavar="FILE", required=False)
    parser.add_argument("-l", "--load", dest="load_path", help="path to load the model from", metavar="FILE", required=False)
    parser.add_argument('--save_w2v', dest='save_w2v', required=False, help="save word (token) vectors in word2vec format")
    parser.add_argument('--save_t2v', dest='save_t2v', required=False, help="save target vectors in word2vec format")
    parser.add_argument('--export_code_vectors', action='store_true', required=False, help="export code vectors for the given examples")
    parser.add_argument('--release', action='store_true', help='if specified and loading a trained model, release the loaded model for a lower model size.')
    parser.add_argument('--predict', action='store_true', help='execute the interactive prediction shell')
    parser.add_argument("-fw", "--framework", dest="dl_framework", choices=['keras', 'tensorflow'], default='tensorflow', help="deep learning framework to use.")
    parser.add_argument("-v", "--verbose", dest="verbose_mode", type=int, required=False, default=1, help="verbose mode (should be in {0,1,2}).")
    parser.add_argument("-lp", "--logs-path", dest="logs_path", metavar="FILE", required=False, help="path to store logs into. if not given logs are not saved to file.")
    parser.add_argument('-tb', '--tensorboard', dest='use_tensorboard', action='store_true', help='use tensorboard during training')
    parser.add_argument('--in_dir', type=str, required=False, help='Java files directory.')
    parser.add_argument('--out_dir', type=str, required=False, help='Code vector directory.')
    parser.add_argument('--jar_path', type=str, default="./scripts/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar")
    parser.add_argument('--max_contexts', type=int, default=200)
    parser.add_argument('--max_path_length', type=int, default=8)
    parser.add_argument('--max_path_width', type=int, default=2)

    # TransformerTree
    parser.add_argument('--max_seq_length', type=int, default=1024 * 1)
    # parser.add_argument('--embed_dim', type=int, default=512)
    # parser.add_argument('--num_heads', type=int, default=8)
    # parser.add_argument('--ff_dim', type=int, default=512)
    # parser.add_argument('--num_transformer_blocks', type=int, default=1)

    # TransformerTree
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--drop', type=float, default=0.1)

    # parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    args = parser.parse_args()

    # setting the seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # preprocess
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)
