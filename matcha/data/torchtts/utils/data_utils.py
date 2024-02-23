from typing import Dict


def lowercase_dict_keys(dict_obj: Dict):
    for key in list(dict_obj.keys()):
        if key.lower() != key:
            if key.lower() in dict_obj:
                raise ValueError(f"Duplicate key [{key.lower()}] in dict")
            dict_obj[key.lower()] = dict_obj.pop(key)


def _bucket_boundaries(max_length, min_length=8, length_bucket_step=1.1):
    """A default set of length-bucket boundaries."""
    assert length_bucket_step > 1.0
    x = min_length
    boundaries = []
    while x < max_length:
        boundaries.append(x)
        x = max(x + 1, int(x * length_bucket_step))
    return boundaries


def get_bucket_scheme(max_token_size, min_length_bucket, bucket_step, length_multiplier=1):
    """A bucket scheme based on max token/frame size.

    Every batch contains a number of sequences divisible by `shard_multiplier`.

    Args:
      max_token_size: int, max number of tokens (frames) in a batch.
      min_length_bucket: int
      bucket_step: float greater than 1.0
      length_multiplier: int, used to increase the bucket boundary tolerance.

    Returns:
       A dictionary with parameters that can be passed to DynamicBatchWithBucket:
         * boundaries: list of bucket boundaries
         * batch_sizes: list of batch sizes for each length bucket
    """
    boundaries = _bucket_boundaries(max_token_size, min_length_bucket, bucket_step)
    boundaries = [boundary * length_multiplier for boundary in boundaries]

    batch_sizes = [max(1, max_token_size // length) for length in boundaries + [length_multiplier * max_token_size]]
    max_batch_size = max(batch_sizes)

    highly_composite_numbers = [
        1,
        2,
        4,
        6,
        12,
        24,
        36,
        48,
        60,
        120,
        180,
        240,
        360,
        720,
        840,
        1260,
        1680,
        2520,
        5040,
        7560,
        10080,
        15120,
        20160,
        25200,
        27720,
        45360,
        50400,
        55440,
        83160,
        110880,
        166320,
        221760,
        277200,
        332640,
        498960,
        554400,
        665280,
        720720,
        1081080,
        1441440,
        2162160,
        2882880,
        3603600,
        4324320,
        6486480,
        7207200,
        8648640,
        10810800,
        14414400,
        17297280,
        21621600,
        32432400,
        36756720,
        43243200,
        61261200,
        73513440,
        110270160,
    ]
    window_size = max([i for i in highly_composite_numbers if i <= 3 * max_batch_size])
    divisors = [i for i in range(1, window_size + 1) if window_size % i == 0]
    batch_sizes = [max([d for d in divisors if d <= bs]) for bs in batch_sizes]

    ret = {
        "boundaries": boundaries,
        "batch_sizes": batch_sizes,
    }
    return ret
