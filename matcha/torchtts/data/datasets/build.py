import hydra
import logging
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def build_data_pipeline(dataset_config):
    logger.info("Instantiating dataset with config: \n" f"{OmegaConf.to_yaml(dataset_config, sort_keys=True)}")
    dataset_builder = hydra.utils.instantiate(dataset_config)
    dataset_builder.prepare_dataset()
    return dataset_builder.as_data_pipeline()
