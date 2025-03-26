from __future__ import annotations
from abc import ABC, abstractmethod
from typing import *
from paddle.base.core import cinn
# print(cinn.autotuner.tuner_config)
from paddle import static
import random

class BaseBenchFunc(ABC):
    @abstractmethod
    def __call__(self, candidate: List[int]) -> float:
        pass

class WeightedBenchFunc(BaseBenchFunc):
    def __init__(
        self,
        program: static.Program,
        bucket_info: cinn.autotuner.tuner_config.BucketInfo,       # 算子输入变量的 shape 限制范围
        sampling_prob: float = 1.0,                 # shape 范围内采样率
        max_sampling_times: int = 65536,            # shape 范围内最大采样次数
        repeats: int = 80,                          # 每个输入参数下，算子运行次数
        weights: Optional[List[List[float]]] = None # shape 限制范围内每个可选shape选中概率分布
    ):
        self.program = program
        self.bucket_info = bucket_info
        self.measurer = cinn.autotuner.search.Measurer(program)
        self.sampling_prob = sampling_prob
        self.max_sampling_times = max_sampling_times
        self.repeats = repeats
        self.sampling_times = 0
        self.weights = weights if weights else []
        self.inputs_sampling: List[Dict[str, List[int]]] = []
        self.rand_seed = 1

        weighted_space_size = 1.0
        if not self.weights:
            for dim in self.bucket_info.space:
                count = dim.upper_bound - dim.lower_bound + 1
                self.weights.append([1.0] * count)
                weighted_space_size *= sum(self.weights[-1])
        else:
            for i, dim in enumerate(self.bucket_info.space):
                expected = dim.upper_bound - dim.lower_bound + 1
                print(f"expected = {expected}")
                print(f"weight len = {len(self.weights[i])}")
                assert len(self.weights[i]) == expected
                weighted_space_size *= sum(self.weights[i])
        
        self.sampling_times = min(int(weighted_space_size * sampling_prob), max_sampling_times)
        
        # Generate samples
        def sample() -> List[int]:
            samples = []
            for i, dim in enumerate(self.bucket_info.space):
                probs = [w / sum(self.weights[i]) for w in self.weights[i]]
                sampled = random.choices(range(len(probs)), weights=probs, k=1)[0]
                samples.append(sampled + dim.lower_bound)
            return samples
        
        for _ in range(self.sampling_times):
            self.inputs_sampling.append({"x": sample()})

    def __call__(self, candidate: List[int]) -> float:
        cinn.autotuner.tuner_config._tuner_add_config_helper(candidate, self.bucket_info)
        # tile_config_db = tuner_config.NaiveTileConfigDatabase()
        # if candidate:
        #     config = ScheduleConfig()
        #     config.warp_num = candidate[0]
        #     config.tree_reduce_num = candidate[1]
        #     config.spatial_inner_num = candidate[2]
        #     tile_config_db.AddConfig("default", self.bucket_info, config)
        #     ScheduleConfigManager.Instance().AddConfigDatabase("search", tile_config_db)
        self.measurer.compile()
        for inputs in self.inputs_sampling:
            self.measurer.run(inputs, self.repeats)
        
        return self.measurer.result().avg_kernel_execute_time

