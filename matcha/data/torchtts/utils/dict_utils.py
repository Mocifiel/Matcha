import itertools


def get_for_key_str(dictionary, key):
    """Returns value from dict by key.
    Args:
        dictionary: dict
        key: key
    Returns:
        value
    """
    return dictionary[key]


def get_for_key_list(dictionary, key):
    """Returns sub-dict from dict by list of keys.
    Args:
        dictionary: dict
        key: list of keys
    Returns:
        sub-dict
    """
    result = {name: dictionary[name] for name in key}
    return result


def get_for_key_dict(dictionary, key):
    """Returns sub-dict from dict by dict-mapping of keys.
    Args:
        dictionary: dict
        key: dict-mapping of keys
    Returns:
        sub-dict
    """
    result = {key_out: dictionary[key_in] for key_in, key_out in key.items()}
    return result


def get_for_key_none(dictionary, key):
    """Returns empty dict.
    Args:
        dictionary: dict
        key: none
    Returns:
        dict
    """
    return {}


def get_for_key_all(dictionary, key):
    """Returns whole dict.
    Args:
        dictionary: dict
        key: none
    Returns:
        dict
    """
    return dictionary


def get_universal_retrieve_fn(key):
    """Get retrieve function for dict based on the type of key.
    Args:
        key: keys
    Returns:
        function
    Raises:
        NotImplementedError: if key is out of
            `str`, `tuple`, `list`, `dict`, `None`
    """
    if isinstance(key, str):
        if key == "__all__":
            return get_for_key_all
        else:
            return get_for_key_str
    elif isinstance(key, (list, tuple)):
        return get_for_key_list
    elif isinstance(key, dict):
        return get_for_key_dict
    elif key is None:
        return get_for_key_none
    else:
        raise NotImplementedError(f"{type(key)} is not supported as a key.")


def zip_dict(*dicts):
    """Iterate over items of dictionaries grouped by their keys."""
    for key in set(itertools.chain(*dicts)):  # set merge all keys
        # Will raise KeyError if the dict don't have the same keys
        yield key, tuple(d[key] for d in dicts)


def map_nested(function, data_struct, dict_only=False, map_tuple=False):
    """Apply a function recursively to each element of a nested data struct."""

    # Could add support for more exotic data_struct, like OrderedDict
    if isinstance(data_struct, dict):
        return {k: map_nested(function, v, dict_only, map_tuple) for k, v in data_struct.items()}
    elif not dict_only:
        types_ = [list]
        if map_tuple:
            types_.append(tuple)
        if isinstance(data_struct, tuple(types_)):
            mapped = [map_nested(function, v, dict_only, map_tuple) for v in data_struct]
            if isinstance(data_struct, list):
                return mapped
            else:
                return tuple(mapped)
    # Singleton
    return function(data_struct)
