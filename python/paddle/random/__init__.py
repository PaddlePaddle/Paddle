from __future__ import annotations

import paddle

__all__ = ["initial_seed"]


def initial_seed() -> int:
    """
    Returns the initial seed for generating random numbers as a Python `long`.

    Returns:
        int: The 64-bit initial seed of the default generator on CPU place only.

    Examples:
        >>> import paddle
        >>> s = paddle.random.initial_seed()
    """
    return paddle.get_rng_state('cpu')[0].current_seed()
