class TensorInfo:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    @classmethod
    def copy_from(cls, tensor_info):
        return cls(shape=tensor_info.shape, dtype=tensor_info.dtype)

    def __eq__(self, other):
        return self.shape == other.shape and self.dtype == other.dtype

    def __repr__(self):
        return "{}(shape={}, dtype={})".format(type(self).__name__, self.shape, repr(self.dtype))
