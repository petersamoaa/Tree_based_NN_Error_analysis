import logging
import os
import sys
from argparse import ArgumentParser
from math import ceil
from typing import Optional


class Config:
    @classmethod
    def arguments_parser(cls) -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument("-d", "--data", dest="data_path",
                            help="path to preprocessed dataset", required=False)
        parser.add_argument("-te", "--test", dest="test_path",
                            help="path to test file", metavar="FILE", required=False, default='')
        parser.add_argument("-s", "--save", dest="save_path",
                            help="path to save the model file", metavar="FILE", required=False)
        parser.add_argument("-w2v", "--save_word2v", dest="save_w2v",
                            help="path to save the tokens embeddings file", metavar="FILE", required=False)
        parser.add_argument("-t2v", "--save_target2v", dest="save_t2v",
                            help="path to save the targets embeddings file", metavar="FILE", required=False)
        parser.add_argument("-l", "--load", dest="load_path",
                            help="path to load the model from", metavar="FILE", required=False)
        parser.add_argument('--save_w2v', dest='save_w2v', required=False,
                            help="save word (token) vectors in word2vec format")
        parser.add_argument('--save_t2v', dest='save_t2v', required=False,
                            help="save target vectors in word2vec format")
        parser.add_argument('--export_code_vectors', action='store_true', required=False,
                            help="export code vectors for the given examples")
        parser.add_argument('--release', action='store_true',
                            help='if specified and loading a trained model, release the loaded model for a lower model '
                                 'size.')
        parser.add_argument('--predict', action='store_true',
                            help='execute the interactive prediction shell')
        parser.add_argument("-fw", "--framework", dest="dl_framework", choices=['keras', 'tensorflow'],
                            default='tensorflow', help="deep learning framework to use.")
        parser.add_argument("-v", "--verbose", dest="verbose_mode", type=int, required=False, default=1,
                            help="verbose mode (should be in {0,1,2}).")
        parser.add_argument("-lp", "--logs-path", dest="logs_path", metavar="FILE", required=False,
                            help="path to store logs into. if not given logs are not saved to file.")
        parser.add_argument('-tb', '--tensorboard', dest='use_tensorboard', action='store_true',
                            help='use tensorboard during training')

        # extra ....
        parser.add_argument('--in_dir', type=str, required=False, help='Java files directory.')
        parser.add_argument('--out_dir', type=str, required=False, help='Code vector directory.')
        parser.add_argument('--jar_path', type=str, default="./scripts/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar")
        parser.add_argument('--max_contexts', type=int, default=200)
        parser.add_argument('--max_path_length', type=int, default=8)
        parser.add_argument('--max_path_width', type=int, default=2)

        return parser

    def set_defaults(self):
        self.NUM_TRAIN_EPOCHS = 20
        self.SAVE_EVERY_EPOCHS = 1
        self.TRAIN_BATCH_SIZE = 1024
        self.TEST_BATCH_SIZE = self.TRAIN_BATCH_SIZE
        self.TOP_K_WORDS_CONSIDERED_DURING_PREDICTION = 10
        self.NUM_BATCHES_TO_LOG_PROGRESS = 100
        self.NUM_TRAIN_BATCHES_TO_EVALUATE = 1800
        self.READER_NUM_PARALLEL_BATCHES = 6  # cpu cores [for tf.contrib.data.map_and_batch() in the reader]
        self.SHUFFLE_BUFFER_SIZE = 10000
        self.CSV_BUFFER_SIZE = 100 * 1024 * 1024  # 100 MB
        self.MAX_TO_KEEP = 10

        # model hyper-params
        self.MAX_CONTEXTS = 200
        self.MAX_TOKEN_VOCAB_SIZE = 1301136
        self.MAX_TARGET_VOCAB_SIZE = 261245
        self.MAX_PATH_VOCAB_SIZE = 911417
        self.DEFAULT_EMBEDDINGS_SIZE = 128
        self.TOKEN_EMBEDDINGS_SIZE = self.DEFAULT_EMBEDDINGS_SIZE
        self.PATH_EMBEDDINGS_SIZE = self.DEFAULT_EMBEDDINGS_SIZE
        self.CODE_VECTOR_SIZE = self.context_vector_size
        self.TARGET_EMBEDDINGS_SIZE = self.CODE_VECTOR_SIZE
        self.DROPOUT_KEEP_RATE = 0.75
        self.SEPARATE_OOV_AND_PAD = False

    def load_from_args(self, args=None):
        args = self.arguments_parser().parse_args() if not args else args
        # Automatically filled, do not edit:
        self.PREDICT = args.predict
        self.MODEL_SAVE_PATH = args.save_path
        self.MODEL_LOAD_PATH = args.load_path
        self.TRAIN_DATA_PATH_PREFIX = args.data_path
        self.TEST_DATA_PATH = args.test_path
        self.RELEASE = args.release
        self.EXPORT_CODE_VECTORS = args.export_code_vectors
        self.SAVE_W2V = args.save_w2v
        self.SAVE_T2V = args.save_t2v
        self.VERBOSE_MODE = args.verbose_mode
        self.LOGS_PATH = args.logs_path
        self.DL_FRAMEWORK = 'tensorflow' if not args.dl_framework else args.dl_framework
        self.USE_TENSORBOARD = args.use_tensorboard

        self.IN_DIR = args.in_dir
        self.OUT_DIR = args.out_dir
        self.JAR_PATH = args.jar_path
        self.MAX_PATH_LENGTH = args.max_path_length
        self.MAX_PATH_WIDTH = args.max_path_width

    def __init__(self, set_defaults: bool = False, load_from_args: bool = False, verify: bool = False, **kwargs):
        self.NUM_TRAIN_EPOCHS: int = kwargs.get("num_train_epochs", 0)
        self.SAVE_EVERY_EPOCHS: int = kwargs.get("save_every_epochs", 0)
        self.TRAIN_BATCH_SIZE: int = kwargs.get("train_batch_size", 0)
        self.TEST_BATCH_SIZE: int = kwargs.get("test_batch_size", 0)
        self.TOP_K_WORDS_CONSIDERED_DURING_PREDICTION: int = kwargs.get("top_k_words_considered_during_prediction", 0)
        self.NUM_BATCHES_TO_LOG_PROGRESS: int = kwargs.get("num_batches_to_log_progress", 0)
        self.NUM_TRAIN_BATCHES_TO_EVALUATE: int = kwargs.get("num_train_batches_to_evaluate", 0)
        self.READER_NUM_PARALLEL_BATCHES: int = kwargs.get("reader_num_parallel_batches", 0)
        self.SHUFFLE_BUFFER_SIZE: int = kwargs.get("shuffle_buffer_size", 0)
        self.CSV_BUFFER_SIZE: int = kwargs.get("csv_buffer_size", 0)
        self.MAX_TO_KEEP: int = kwargs.get("max_to_keep", 0)

        # model hyper-params
        self.MAX_CONTEXTS: int = kwargs.get("max_contexts", 0)
        self.MAX_TOKEN_VOCAB_SIZE: int = kwargs.get("max_token_vocab_size", 0)
        self.MAX_TARGET_VOCAB_SIZE: int = kwargs.get("max_target_vocab_size", 0)
        self.MAX_PATH_VOCAB_SIZE: int = kwargs.get("max_path_vocab_size", 0)
        self.DEFAULT_EMBEDDINGS_SIZE: int = kwargs.get("default_embedding_size", 0)
        self.TOKEN_EMBEDDINGS_SIZE: int = kwargs.get("token_embedding_size", 0)
        self.PATH_EMBEDDINGS_SIZE: int = kwargs.get("path_embedding_size", 0)
        self.CODE_VECTOR_SIZE: int = kwargs.get("code_vector_size", 0)
        self.TARGET_EMBEDDINGS_SIZE: int = kwargs.get("target_embedding_size", 0)
        self.DROPOUT_KEEP_RATE: float = kwargs.get("dropout_keep_rate", 0)
        self.SEPARATE_OOV_AND_PAD: bool = kwargs.get("separate_oov_and_pad", False)

        # Automatically filled by `args`.
        self.PREDICT: bool = kwargs.get("predict", False)  # TODO: update README;
        self.MODEL_SAVE_PATH: Optional[str] = kwargs.get("model_save_path", None)
        self.MODEL_LOAD_PATH: Optional[str] = kwargs.get("model_load_path", None)
        self.TRAIN_DATA_PATH_PREFIX: Optional[str] = kwargs.get("train_data_path_prefix", None)
        self.TEST_DATA_PATH: Optional[str] = kwargs.get("test_data_path", "")
        self.RELEASE: bool = kwargs.get("release", False)
        self.EXPORT_CODE_VECTORS: bool = kwargs.get("export_code_vectors", False)
        self.SAVE_W2V: Optional[str] = kwargs.get("save_w2v", None)  # TODO: update README;
        self.SAVE_T2V: Optional[str] = kwargs.get("save_t2v", None)  # TODO: update README;
        self.VERBOSE_MODE: int = kwargs.get("verbose_mode", 0)
        self.LOGS_PATH: Optional[str] = kwargs.get("logs_path", None)
        self.DL_FRAMEWORK: str = kwargs.get("dl_framework", "")  # in {'keras', 'tensorflow'}
        self.USE_TENSORBOARD: bool = kwargs.get("use_tensorboard", False)

        # Automatically filled by `Code2VecModelBase._init_num_of_examples()`.
        self.NUM_TRAIN_EXAMPLES: int = kwargs.get("num_train_examples", 0)
        self.NUM_TEST_EXAMPLES: int = kwargs.get("num_test_examples", 0)

        self.IN_DIR = kwargs.get("in_dir", None)
        self.OUT_DIR = kwargs.get("out_dir", None)
        self.JAR_PATH = kwargs.get("jar_path", None)
        self.MAX_PATH_LENGTH = kwargs.get("max_path_length", 2)
        self.MAX_PATH_WIDTH = kwargs.get("max_path_width", 2)

        self.__logger: Optional[logging.Logger] = kwargs.get("logger", None)

        # custom
        self.PREDICT = kwargs.get("predict", False)

        if set_defaults:
            self.set_defaults()
        if load_from_args:
            self.load_from_args(args=kwargs.get("args", None))
        if verify:
            self.verify()

    @property
    def context_vector_size(self) -> int:
        # The context vector is actually a concatenation of the embedded
        # source & target vectors and the embedded path vector.
        return self.PATH_EMBEDDINGS_SIZE + 2 * self.TOKEN_EMBEDDINGS_SIZE

    @property
    def is_training(self) -> bool:
        return bool(self.TRAIN_DATA_PATH_PREFIX)

    @property
    def is_loading(self) -> bool:
        return bool(self.MODEL_LOAD_PATH)

    @property
    def is_saving(self) -> bool:
        return bool(self.MODEL_SAVE_PATH)

    @property
    def is_testing(self) -> bool:
        return bool(self.TEST_DATA_PATH)

    @property
    def train_steps_per_epoch(self) -> int:
        return ceil(self.NUM_TRAIN_EXAMPLES / self.TRAIN_BATCH_SIZE) if self.TRAIN_BATCH_SIZE else 0

    @property
    def test_steps(self) -> int:
        return ceil(self.NUM_TEST_EXAMPLES / self.TEST_BATCH_SIZE) if self.TEST_BATCH_SIZE else 0

    def data_path(self, is_evaluating: bool = False):
        return self.TEST_DATA_PATH if is_evaluating else self.train_data_path

    def batch_size(self, is_evaluating: bool = False):
        return self.TEST_BATCH_SIZE if is_evaluating else self.TRAIN_BATCH_SIZE  # take min with NUM_TRAIN_EXAMPLES?

    @property
    def train_data_path(self) -> Optional[str]:
        if not self.is_training:
            return None
        return '{}.train.c2v'.format(self.TRAIN_DATA_PATH_PREFIX)

    @property
    def word_freq_dict_path(self) -> Optional[str]:
        if not self.is_training:
            return None
        return '{}.dict.c2v'.format(self.TRAIN_DATA_PATH_PREFIX)

    @classmethod
    def get_vocabularies_path_from_model_path(cls, model_file_path: str) -> str:
        vocabularies_save_file_name = "dictionaries.bin"
        return '/'.join(model_file_path.split('/')[:-1] + [vocabularies_save_file_name])

    @classmethod
    def get_entire_model_path(cls, model_path: str) -> str:
        return model_path + '__entire-model'

    @classmethod
    def get_model_weights_path(cls, model_path: str) -> str:
        return model_path + '__only-weights'

    @property
    def model_load_dir(self):
        return '/'.join(self.MODEL_LOAD_PATH.split('/')[:-1])

    @property
    def entire_model_load_path(self) -> Optional[str]:
        if not self.is_loading:
            return None
        return self.get_entire_model_path(self.MODEL_LOAD_PATH)

    @property
    def model_weights_load_path(self) -> Optional[str]:
        if not self.is_loading:
            return None
        return self.get_model_weights_path(self.MODEL_LOAD_PATH)

    @property
    def entire_model_save_path(self) -> Optional[str]:
        if not self.is_saving:
            return None
        return self.get_entire_model_path(self.MODEL_SAVE_PATH)

    @property
    def model_weights_save_path(self) -> Optional[str]:
        if not self.is_saving:
            return None
        return self.get_model_weights_path(self.MODEL_SAVE_PATH)

    def verify(self):
        if not self.is_training and not self.is_loading:
            raise ValueError("Must train or load a model.")
        if self.is_loading and not os.path.isdir(self.model_load_dir):
            raise ValueError("Model load dir `{model_load_dir}` does not exist.".format(
                model_load_dir=self.model_load_dir))
        if self.DL_FRAMEWORK not in {'tensorflow', 'keras'}:
            raise ValueError("config.DL_FRAMEWORK must be in {'tensorflow', 'keras'}.")

    def __iter__(self):
        for attr_name in dir(self):
            if attr_name.startswith("__"):
                continue
            try:
                attr_value = getattr(self, attr_name, None)
            except:
                attr_value = None
            if callable(attr_value):
                continue
            yield attr_name, attr_value

    def get_logger(self) -> logging.Logger:
        if self.__logger is None:
            self.__logger = logging.getLogger('code2vec')
            self.__logger.setLevel(logging.INFO)
            self.__logger.handlers = []
            self.__logger.propagate = 0

            formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')

            if self.VERBOSE_MODE >= 1:
                ch = logging.StreamHandler(sys.stdout)
                ch.setLevel(logging.INFO)
                ch.setFormatter(formatter)
                self.__logger.addHandler(ch)

            if self.LOGS_PATH:
                fh = logging.FileHandler(self.LOGS_PATH)
                fh.setLevel(logging.INFO)
                fh.setFormatter(formatter)
                self.__logger.addHandler(fh)

        return self.__logger

    def log(self, msg):
        self.get_logger().info(msg)
