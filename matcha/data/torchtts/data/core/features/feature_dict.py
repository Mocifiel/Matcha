import collections
import logging

from torchtts.data.core.features import feature_connector
from torchtts.data.core.features import tensor_feature
from torchtts.utils.dict_utils import zip_dict
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache
def log_warning_once(*args, **kwargs):
    logger.warning(*args, **kwargs)


class FeaturesDict(collections.abc.Mapping, feature_connector.FeatureConnector):
    def __init__(self, feature_dict):
        super(FeaturesDict, self).__init__()
        self._feature_dict = {k: to_feature(v) for k, v in feature_dict.items()}

    def keys(self):
        return self._feature_dict.keys()

    def items(self):
        return self._feature_dict.items()

    def values(self):
        return self._feature_dict.values()

    def __contains__(self, k):
        return k in self._feature_dict

    def __getitem__(self, key):
        return self._feature_dict[key]

    def __len__(self):
        return len(self._feature_dict)

    def __iter__(self):
        return iter(self._feature_dict)

    def __repr__(self):
        lines = ["{}({{".format(type(self).__name__)]
        for key, feature in sorted(self._feature_dict.items()):
            feature_repr = get_inner_feature_repr(feature)
            all_sub_lines = "'{}': {},".format(key, feature_repr)
            lines.extend("    " + i for i in all_sub_lines.split("\n"))
        lines.append("})")
        return "\n".join(lines)

    def get_tensor_info(self):
        return {feature_key: feature.get_tensor_info() for feature_key, feature in self._feature_dict.items()}

    def get_serialized_info(self):
        return {feature_key: feature.get_serialized_info() for feature_key, feature in self._feature_dict.items()}

    def encode_example(self, example_dict):
        return {
            k: feature.encode_example(example_value)
            for k, (feature, example_value) in zip_dict(self._feature_dict, example_dict)
        }

    def decode_example(self, example_dict):
        decoded_example_dict = {}
        for key, example_value in example_dict.items():
            if key == "__key__":
                decoded_example_dict[key] = example_value
            elif key in self._feature_dict:
                decoded_example_dict[key] = self._feature_dict[key].decode_example(example_value)
            else:
                log_warning_once(
                    f"failed to decode example of {key} since the key is missing in DatasetInfo "
                    "of the dataset. Ignore this warning if you don't need that for training."
                )
        return decoded_example_dict


def to_feature(value):
    if isinstance(value, feature_connector.FeatureConnector):
        return value
    elif isinstance(value, dict):
        return FeaturesDict(value)
    else:
        raise ValueError("Feature not supported: {}".format(value))


def get_inner_feature_repr(feature):
    if type(feature) == tensor_feature.Tensor and feature.shape == ():
        return repr(feature.dtype)
    else:
        return repr(feature)
