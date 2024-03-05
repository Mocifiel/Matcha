def is_exception(ex):
    """Check if the argument is of ``Exception`` type."""
    result = (ex is not None) and isinstance(ex, BaseException)
    return result


class InstantiationError(Exception):
    """Exception class for instantiation errors."""

    def __init__(self, message):
        """
        Args:
            message: exception message
        """
        super().__init__(message)
