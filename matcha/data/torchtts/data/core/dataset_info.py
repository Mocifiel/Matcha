from torchtts.data.core.features import feature_dict


class DatasetInfo:
    def __init__(self, *, builder, description, features):
        self._builder = builder
        self._description = description
        if not isinstance(features, feature_dict.FeaturesDict):
            raise ValueError("DatasetInfo.features only supports FeaturesDict. " f"Got {type(features)}")
        self._features = features

    @property
    def builder(self):
        return self._builder

    @property
    def description(self):
        return self._description

    @property
    def features(self):
        return self._features

    def __len__(self):
        return len(self._features)
