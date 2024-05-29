import logging

import numpy as np
from extract_ast_paths import Extractor
from models.code2vec.config import Config
from models.code2vec.model_base import Code2VecModelBase
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model_dynamically(config: Config) -> Code2VecModelBase:
    assert config.DL_FRAMEWORK in {'tensorflow', 'keras'}

    if config.DL_FRAMEWORK == 'tensorflow':
        from models.code2vec.tensorflow_model import Code2VecModel
    elif config.DL_FRAMEWORK == 'keras':
        from models.code2vec.keras_model import Code2VecModel

    logger.info(f"Type Code2VecModel: {type(Code2VecModel)}")
    return Code2VecModel(config)


def prepare_data(data, config: Config, test=None):
    """Parse the tree data from a pickle file and create samples.

    Args:
        args (AttrDict): The arguments containing paths and limits for processing.
    """
    model = load_model_dynamically(config)
    path_extractor = Extractor(
        jar_path=config.JAR_PATH,
        max_contexts=config.MAX_CONTEXTS,
        max_path_length=config.MAX_PATH_LENGTH,
        max_path_width=config.MAX_PATH_WIDTH
    )
    data_with_repr, test_with_repr = [], []

    for row in tqdm(data, total=len(data)):
        try:
            lines, hash_to_string_dict = path_extractor.extract_paths(row["path"])
        except Exception as e:
            print(e)
            continue

        if lines:
            representation = model.predict(lines)

            row["representation"] = np.vstack([rp.code_vector for rp in representation])
            data_with_repr.append(row)

    if test and isinstance(test, list):
        for row in tqdm(test, total=len(test)):
            try:
                lines, hash_to_string_dict = path_extractor.extract_paths(row["path"])
            except Exception as e:
                print(e)
                # raise
                continue

            if lines:
                representation = model.predict(lines)
                row["representation"] = np.vstack([rp.code_vector for rp in representation])
                test_with_repr.append(row)

    return data_with_repr, test_with_repr
