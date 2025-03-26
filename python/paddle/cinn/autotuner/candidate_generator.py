from abc import ABC, abstractmethod
from typing import *  # noqa: F403
from typing_extensions import *  # type: ignore # noqa: F403
import itertools

from paddle._typing import *  # noqa: F403

from .bench_func import BaseBenchFunc

class BaseCandidateGenerator(ABC):
    @abstractmethod
    def candidates(self) -> List[List[int]]:
        pass
    @abstractmethod
    def next(self, candidate: List[int], ndim: int, step: int) -> List[int]:
        pass
    @abstractmethod
    def param_names(self) -> List[str]:
        pass

class BFGenerator(BaseCandidateGenerator):
    def __init__(
        self,
        # [Tuner_Arg0(start, end), ...]
        candidate_range:  Dict[str, Tuple[int, int]],
        # candidate: [Tuner_Arg0, Tuner_Arg1, ...]
        # constraints: [constraint0: Callable, constraint1: Callable, ...]
        # 限制candidate每个维度
        constraints: List[Callable[[List[int]], bool]] = []
    ):
        self.constraints = constraints
        self.candidates_each_dim = {
            # 每个 range 全展开
            key: list(range(start, end+1)) for key, (start, end) in candidate_range.items()
        }

    # 返回所以可选的cadidate
    def candidates(self) -> List[List[int]]:
        # 全排列所有组合
        all_combinations = itertools.product(*self.candidates_each_dim.values())
        valid_candidates = [
            list(comb) for comb in all_combinations if self.is_valid(list(comb))
        ]
        return valid_candidates

    def next(self, candidate: List[int], ndim: int, step: int) -> List[int]:
        assert len(candidate) == len(self.candidates_each_dim)
        assert 0 <= ndim < len(candidate)
        
        current_idx = self.candidates_each_dim[ndim].index(candidate[ndim])
        new_idx = (current_idx + step) % len(self.candidates_each_dim[ndim])
        new_candidate = candidate.copy()
        new_candidate[ndim] = self.candidates_each_dim[ndim][new_idx]
        return new_candidate

    def param_names(self) -> List[str]:
        return list(self.candidates_each_dim.keys())

    def is_valid(self, candidate: List[int]) -> bool:
        return all(constraint(candidate) for constraint in self.constraints)
