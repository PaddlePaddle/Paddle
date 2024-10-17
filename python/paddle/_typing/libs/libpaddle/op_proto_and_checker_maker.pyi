from __future__ import annotations
import typing
__all__ = ['OpRole', 'kOpCreationCallstackAttrName', 'kOpDeviceAttrName', 'kOpNameScopeAttrName', 'kOpRoleAttrName', 'kOpRoleVarAttrName', 'kOpWithQuantAttrName']
class OpRole:
    """
    Members:
    
      Forward
    
      Backward
    
      Optimize
    
      Loss
    
      RPC
    
      Dist
    
      LRSched
    """
    Backward: typing.ClassVar[OpRole]  # value = <OpRole.Backward: 1>
    Dist: typing.ClassVar[OpRole]  # value = <OpRole.Dist: 8>
    Forward: typing.ClassVar[OpRole]  # value = <OpRole.Forward: 0>
    LRSched: typing.ClassVar[OpRole]  # value = <OpRole.LRSched: 16>
    Loss: typing.ClassVar[OpRole]  # value = <OpRole.Loss: 256>
    Optimize: typing.ClassVar[OpRole]  # value = <OpRole.Optimize: 2>
    RPC: typing.ClassVar[OpRole]  # value = <OpRole.RPC: 4>
    __members__: typing.ClassVar[dict[str, OpRole]]  # value = {'Forward': <OpRole.Forward: 0>, 'Backward': <OpRole.Backward: 1>, 'Optimize': <OpRole.Optimize: 2>, 'Loss': <OpRole.Loss: 256>, 'RPC': <OpRole.RPC: 4>, 'Dist': <OpRole.Dist: 8>, 'LRSched': <OpRole.LRSched: 16>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
def kOpCreationCallstackAttrName() -> str:
    ...
def kOpDeviceAttrName() -> str:
    ...
def kOpNameScopeAttrName() -> str:
    ...
def kOpRoleAttrName() -> str:
    ...
def kOpRoleVarAttrName() -> str:
    ...
def kOpWithQuantAttrName() -> str:
    ...
