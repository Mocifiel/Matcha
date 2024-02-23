from abc import ABC
from abc import abstractmethod
import logging
import os

from torchtts.data.core import dataset_info
from torchtts.data.core import datapipes
from torchtts.data.core import writers
from torchtts.utils.dict_utils import map_nested
from torchtts.utils.file_utils import incomplete_dir
from torchtts.utils.misc_utils import memoized_property
from torchtts.utils.misc_utils import temporary_assignment
from torchtts.utils.tqdm_utils import tqdm

logger = logging.getLogger(__name__)


class DatasetBuilder(ABC):
    def __init__(self, data_dir, shard_format="tar", **kwargs):
        assert shard_format in ["tar", "zip", "chunk"]
        self._data_dir = data_dir
        self._shard_format = shard_format
        self._config = kwargs

        split = self._config.get("split", "train")
        if split == "train":
            self._split = {"train": "train"}
        elif split == "train-dev":
            self._split = {"train": "train", "dev": "dev"}
        else:
            raise ValueError(f"split must be 'train' or 'train-dev', but get {split}")

        if self._shard_format == "tar":
            self._writer_cls = writers.TarWriter
            self._shard_suffix = "tar"
        elif self._shard_format == "zip":
            self._writer_cls = writers.ZipWriter
            self._shard_suffix = "zip"
        elif self._shard_format == "chunk":
            self._writer_cls = writers.ChunkWriter
            self._shard_suffix = "chunk"

    @property
    def data_dir(self):
        return self._data_dir

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def split_type(self):
        return self._split

    @memoized_property
    def info(self):
        info = self._info()
        if not isinstance(info, dataset_info.DatasetInfo):
            raise TypeError(f"DatasetBuilder._info should returns DatasetInfo, not {type(info)}")
        return info

    @abstractmethod
    def _info(self):
        raise NotImplementedError("Must be implemented in subclasses.")

    def prepare_dataset(self):
        if os.path.exists(self._data_dir):
            logger.info(f"Reusing dataset {self.name} ({self._data_dir})")
            return

        logger.info(f"Generating dataset {self.name} ({self._data_dir})")
        with incomplete_dir(self._data_dir) as tmp_data_dir:
            with temporary_assignment(self, "_data_dir", tmp_data_dir):
                self._prepare_dataset()

    @abstractmethod
    def _prepare_dataset(self):
        raise NotImplementedError("Must be implemented in subclasses.")

    def as_data_pipeline(self):
        logger.info(f"Constructing DataPipeline for split {self._split}, " f"from {self._data_dir}")
        if not os.path.exists(self._data_dir):
            raise AssertionError(
                f"Dataset {self.name}: could not find data in {self._data_dir}. "
                "Please make sure to call dataset_builder.prepare_data() before "
                "trying to access the Dataset object."
            )

        datasets = map_nested(self._build_single_dataset, self._split, map_tuple=True)
        return datasets

    def _build_single_dataset(self, split):
        shuffle = True if split == "train" else False

        # Collect shard file pathnames
        shard_masks = self._config.get("shard_masks", f"*.{self._shard_suffix}")
        pipe = datapipes.ListDirFilesIterDataPipe(
            os.path.join(self._data_dir, split), masks=shard_masks, recursive=True
        )

        # Dispatch data to different ranks (gpus) in shard-level or sample-level
        dispatch_shards = self._config.get("dispatch_shards", True)
        dispatch_samples = not dispatch_shards

        if dispatch_shards:
            # Shard level shuffling
            if shuffle:
                shard_shuffle_size = self._config.get("shard_shuffle_size", 32)
                pipe = datapipes.ShuffleIterDataPipe(pipe, buffer_size=shard_shuffle_size)

            # Apply sharding to split data according to rank_id, worker_id and world_size
            pipe = datapipes.ShardingFilterIterDataPipe(pipe)

        # Load shard files from disk
        pipe = datapipes.LoadFilesFromDiskIterDataPipe(pipe)

        # Load data from shard files
        if self._shard_format == "zip":
            pipe = datapipes.ReadFilesFromZipIterDataPipe(pipe)
        elif self._shard_format == "tar":
            pipe = datapipes.ReadFilesFromTarIterDataPipe(pipe)
        elif self._shard_format == "chunk":
            pipe = datapipes.ReadFilesFromChunkIterDataPipe(pipe)

        # Group records by key to dict
        pipe = datapipes.GroupByKeyIterDataPipe(pipe)

        if dispatch_samples:
            # Sample level shuffling
            if shuffle:
                sample_shuffle_size = self._config.get("sample_shuffle_size", 128)
                pipe = datapipes.ShuffleIterDataPipe(pipe, buffer_size=sample_shuffle_size)

            # Put ahead before actual reading and decoding to skip unnecessary loading
            pipe = datapipes.ShardingFilterIterDataPipe(pipe)

        # Decode binary data
        def decode_example(example):
            try:
                return self.info.features.decode_example(example)
            except BaseException as ex:
                logger.exception(f"Failed to decode example with __key__ == {example['__key__']}: {ex}")
                return None
        pipe = datapipes.FilterMapIterDataPipe(pipe, decode_example)

        return self._data_pipeline(pipe, shuffle)

    @abstractmethod
    def _data_pipeline(self, datapipe: datapipes.IterDataPipe, shuffle: bool):
        raise NotImplementedError("Must be implemented in subclasses.")


class GeneratorBasedBuilder(DatasetBuilder):
    DEFAULT_SHARD_SIZE = 2000
    DEFAULT_SHARD_NAME = "shards"

    @abstractmethod
    def _split_generators(self):
        raise NotImplementedError("Must be implemented in subclasses.")

    @abstractmethod
    def _raw_data_generator(self, **kwargs):
        raise NotImplementedError("Must be implemented in subclasses.")

    def _prepare_dataset(self):
        split_generators = self._split_generators()
        for split_name, generator in tqdm(split_generators.items(), unit=" splits", leave=False):
            self._build_from_generator(split_name, generator)

    def _build_from_generator(self, split_name, generator):
        split_dir = os.path.join(self._data_dir, split_name)
        os.makedirs(split_dir)

        chunk_size = self._config.get("shard_size", self.DEFAULT_SHARD_SIZE)
        shard_name = self._config.get("shard_name", self.DEFAULT_SHARD_NAME)
        with writers.ShardWriter(
            f"{split_dir}/{shard_name}-%05d.{self._shard_suffix}", writer_cls=self._writer_cls, max_count=chunk_size
        ) as writer:
            for key, example in tqdm(generator, unit=" examples", leave=False):
                example = self.info.features.encode_example(example)
                writer.write({"__key__": key, **example})
                # release memory
                example = None
