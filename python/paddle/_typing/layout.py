from typing import Literal, TypeAlias, Union


# Note: Do not confrom to predefined naming style in pylint.
DataLayout0D: TypeAlias = Literal["NC"]
DataLayout1D: TypeAlias = Literal["NCL", "NLC"]
DataLayout2D: TypeAlias = Literal["NCHW", "NHCW"]
DataLayout3D: TypeAlias = Literal["NCDHW", "NDHWC"]
DataLayoutND: TypeAlias = Union[DataLayout0D, DataLayout1D, DataLayout2D, DataLayout3D]

DataLayout1DVariant: TypeAlias = Literal["NCW", "NWC"]
DataLayoutImage: TypeAlias = Literal["HWC", "CHW"]
