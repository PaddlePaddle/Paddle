from typing import TypeVar, Generic

T = TypeVar('T')

class GuardedBox(Generic[T]):
    def __init__(self, hash_key: int, value):
        self.hash_key_ = hash_key
        self.value_ = value

    def Get(self, hash_key: int):
        assert self.hash_key_ == hash_key
        return self.value_

    @property
    def value(self):
        raise NotImplementedError("please use Get() method")