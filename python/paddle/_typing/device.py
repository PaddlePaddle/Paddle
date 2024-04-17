from typing import TypeAlias, Union


class Place:
    ...


class CPUPlace(Place):
    ...


class CUDAPlace(Place):
    def __init__(self, id: int) -> None: ...


class CUDAPinnedPlace(Place):
    ...


class NPUPlace(Place):
    def __init__(self, id: int) -> None: ...


class IPUPlace(Place):
    ...


class CustomPlace(Place):
    def __init__(self, name: str, id: int) -> None: ...


class MLUPlace(Place):
    def __init__(self, id: int) -> None: ...


class XPUPlace(Place):
    def __init__(self, id: int) -> None: ...


PlaceLike: TypeAlias = Union[
    CPUPlace,
    CUDAPlace,
    CUDAPinnedPlace,
    NPUPlace,
    IPUPlace,
    CustomPlace,
    MLUPlace,
    XPUPlace,
    str,  # It seems that we cannot define the literal for dev:id in nowadays python type-hinting.
]
