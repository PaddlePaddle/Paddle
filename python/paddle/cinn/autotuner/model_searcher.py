from dataclasses import dataclass
import os
import json
from typing import *
import logging, json
import datetime

from paddle import static, Tensor
from paddle.base.core import cinn
# from paddle.cinn import common
from .bench_func import WeightedBenchFunc, Operator
from .candidate_generator import BaseCandidateGenerator
from .candidate_searcher import CandidateSearcher
from .config import *

def candidate_join(name, candidate):
    return [{'name': a, 'value': b} for a, b in zip(name, candidate)]

class ModelSearcher:
    def __init__(
        self, 
        name: str,
        shape: List[int|Tuple[int,int]],
        shape_name: List[str],
        layout: str,
        program_builder,
        candidate_generator: BaseCandidateGenerator,
    ):
        self.name = name
        self.shape = shape
        self.shape_name = shape_name
        self.layout = layout
        self.program_builder = program_builder
        self.candidate_generator = candidate_generator

        # Dynamic shape ranges
        self.__dynamic_window_ranges = [
            (2, 128),
            (129, 512),
            (513, 1024),
            (1025, 2048),
            (2049, 4096),
            (4097, 8192),
        ]

    def _gen_log_file(self, target):
        filename = self.layout
        for dim in self.shape:
            if isinstance(dim, int):
                filename += f"_{dim}"
            else:
                filename += f"_({dim[0]}_{dim[1]})"
        
        now_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')


        DIR_SUFFIX = "_autotune"
        dirname = self.name + DIR_SUFFIX
        filename = filename + f"__{now_time}"
        
        def check_exist(test_path):
            if not os.path.exists(test_path):
                try:
                    os.makedirs(test_path)
                except:
                    raise Exception(f"Cannot create directory: {test_path}")
                    
        root_path = os.path.join(os.getcwd(), "tune_log")
            
        target_str = f"{target.arch_str()}_{target.device_name_str()}"
        
        check_exist(root_path)
        check_exist(os.path.join(root_path, target_str))
        check_exist(os.path.join(root_path, target_str, dirname))
        
        return os.path.join(root_path, target_str, dirname, f"{filename}.csv")

    def _intent(self, intent: int):
        return " "* intent * 4
    
    def _print(self, output, intent: int, message: str):
        output.write(self._intent(intent) + message + "\n")
        output.flush()

    def _build_program(self, shape) -> Tuple[static.Program, static.Program, Tensor]:
        return self.program_builder(shape)
        
    def _create_bucket(self, shape: List[int|Tuple[int,int]]) -> cinn.autotuner.tuner_config.BucketInfo:
        bucket_info = cinn.autotuner.tuner_config.BucketInfo()
        
        space = []
        for i, dim in enumerate(shape):
            dimension = cinn.autotuner.tuner_config.Dimension(
                dim,
                dim,
                self.shape_name[i],
                False
            ) if isinstance(dim, int) else cinn.autotuner.tuner_config.Dimension(
                dim[0],
                dim[1],
                self.shape_name[i],
                True
            )
            space.append(dimension)

        bucket_info.space = space
        logging.info(bucket_info)
        return bucket_info

    def _write_bucket_info(self, output_file, bucket_info: cinn.autotuner.tuner_config.BucketInfo):
        self._print(output_file, 3, "\"shape\": {")
        for i, name in enumerate(self.shape_name):
            bucket = bucket_info.space[i]
            self._print(output_file, 4, f"\"{name}\": [{bucket.lower_bound}, {bucket.upper_bound}]" + 
                        ("," if i < len(self.shape_name) - 1 else ""))
            # output_file.write(f"\"{name}\": [{bucket.lower_bound}, {bucket.upper_bound}],")
        self._print(output_file, 3, "},")
        # output_file.write(" }\n")
        # output_file.flush()

    
    def _get_best_window_range(self, dimension_lower: int) -> Tuple[int, int]:
        for i in range(len(self.__dynamic_window_ranges)):
            if dimension_lower <= self.__dynamic_window_ranges[i][1]:
                return self.__dynamic_window_ranges[i]
        return self.__dynamic_window_ranges[-1]

    def _get_all_dynamic_ranges(self, shape: List[int|Tuple[int,int]]) -> List[List[int|Tuple[int,int]]]:
        all_ranges = []
        def dfs(index, current_range):
            if index == len(shape):
                all_ranges.append(current_range.copy())
                return
            dim = shape[index]
            if isinstance(dim, int):
                current_range[index] = dim
                dfs(index + 1, current_range)
            else:
                i = dim[0]
                while i < dim[1]:
                    best_range = self._get_best_window_range(i)
                    current_range[index] = best_range
                    dfs(index + 1, current_range)
                    i = best_range[1] + 1
        dfs(0, [None] * len(shape))
        return all_ranges

    def _search_window(
        self, 
        output_file,
        shape: List[int|Tuple[int,int]],
        weights: List[float],
        graph_num: int,
        sampling_prob: float,
        num_measure_trials: int,
        repeats: int,
    ):
        shape_weights = []
        for i ,dim in enumerate(shape):
            if isinstance(dim, int):
                shape_weights.append([weights[i]] * 1)
            else:
                shape_weights.append([weights[i]] * (dim[1] - dim[0] + 1))
        
        logging.info(f"shape_weights = {shape_weights}")
        # s_weights = [s_weight] * spatial_tile_width
        # r_weights = [r_weight] * reduce_tile_width
        
        logging.info(f"tuning shape = {shape}")

        # 测试逻辑实现
        program_bundle = self._build_program(shape)
        
        bucket_info = self._create_bucket(shape)
        
        bench_func = WeightedBenchFunc(
            program_bundle, bucket_info, sampling_prob, num_measure_trials, repeats, shape_weights)

        candidate_searcher = CandidateSearcher(
            [bench_func],
            self.candidate_generator
        )

        # write this bucket info
        self._write_bucket_info(output_file, bucket_info)

        logging.info("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        logging.info(f"Start search for shape: {shape}")
        logging.info("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")

        # Baseline
        cinn.autotuner.tuner_config._env_set_tile_config_policy("default")
        baseline_score = bench_func([])
        self._print(output_file, 3, f"\"baseline_score\": {baseline_score/graph_num},")

        # 寻找最佳性能
        cinn.autotuner.tuner_config._env_set_tile_config_policy("search")
        best = candidate_searcher.search()
        best_score = best[0] / graph_num
        best_candidate = json.dumps(candidate_join(
            self.candidate_generator.param_names(), best[1]
        ), indent=4)
                    
        # 输出结果
        logging.info(f"Best score: {best_score}")
        logging.info(f"Best candidate: {best_candidate}")
        logging.info("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        
        # 写入CSV文件
        self._print(output_file, 3, f"\"best_score\": {best_score},")
        self._print(output_file, 3, f"\"best_candidate\": {best_candidate}")
        # output_file.write(f"baseline_score: {(baseline_score/ graph_num):.3f} \n")
        # output_file.write(f"best_score: {best_score:.3f} \n")
        # output_file.write(f"best_candidate: \n{best_candidate} \n")
        # output_file.write(f"=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")
        # output_file.flush()

    def set_search_option(self, option: SearchOption):
        self.search_option = option
        # 获取配置并打开输出文件
        self.log_path = self._gen_log_file(self.search_option.target)

    def get_log_path(self):
        if hasattr(self, 'log_path'):
            return self.log_path
        else:
            raise AttributeError("Log path not set. Please call set_search_option() first.")
    
    def search(
        self, 
        search_option: SearchOption = None
    ):

        # 配置常量
        THREADS_PER_WARP = 32
        MAX_THREADS_PER_BLOCK = 1024

        SAMPLING_PROB = 1.0
        MAX_SAMPLING_TIMES = 360
        REPEATS = 5
        GRAPH_NUM = 25
        # s_weight = 1.0
        # r_weight = 1.0
        
        # iter_space_type = [
        #     ("S", "dynamic" if is_spatial_dynamic else "static"),
        #     ("R", "dynamic" if is_reduce_dynamic else "static")
        # ]
        self.set_search_option(search_option)
        # if search_option is not None:
        #     search_option = self.search_option
        # # 获取配置并打开输出文件
        # log_path = self._gen_log_file(search_option.target)
        
        with open(self.log_path, 'a') as output_file:
            # 测试主循环
            # outside `while` loop init
            
            # 枚举shape中每个动态维度（Tuple[int，int]）在dynamic range内的所有可能值
            # 静态维度（int）保持不变
            all_shapes = self._get_all_dynamic_ranges(self.shape)
            logging.info(f"all_shapes = {all_shapes}")
            self._print(output_file, 0, "{")
            self._print(output_file, 1, f"\"name\": \"{self.name}\",")
            self._print(output_file, 1, f"\"device\": \"{self.search_option.target.arch_str()}_{self.search_option.target.device_name_str()}\",")
            self._print(output_file, 1, f"\"record\": [")

            for i, partitial_shape in enumerate(all_shapes):
                self._print(output_file, 2, f"{{")
                self._search_window(
                    output_file=output_file,
                    shape=partitial_shape,
                    weights=[1.0] * len(partitial_shape),
                    graph_num=GRAPH_NUM,
                    sampling_prob=SAMPLING_PROB,
                    num_measure_trials=self.search_option.num_measure_trials,
                    repeats=self.search_option.repeat
                )
                if i < len(all_shapes) - 1:
                    self._print(output_file, 2, "},")
                else:
                    self._print(output_file, 2, "}")
            self._print(output_file, 1, "]")
            self._print(output_file, 0, "}")
            # output_file.write("]}\n")
        return self.log_path


    def apply(self, search_log: str, shape: List[int|Tuple[int,int]]):
        """
        Apply the search log to the program builder.
        This method should be implemented in subclasses.
        """
        # raise NotImplementedError("This method should be implemented in subclasses.")
        cinn.autotuner.tuner_config._env_set_tile_config_policy("search")
        with open(search_log, 'r') as f:
            results = json.load(f)

        operator_prim = None
        best_candidate = None
        shape
        # for each shape result
        for shape_result in results['record']:
            if shape_result['shape'][1] < shape[1]:
                continue
            program_bundle = self._build_program(shape_result['shape'])
            bucket_info = self._create_bucket(shape_result['shape'])
            operator_prim = Operator(
                self.name, 
                program_bundle,
                bucket_info, 
                
            )
        best_candidate = candidate_join(self.candidate_generator.param_names(), shape_result['best_candidate'])
        return operator_prim(best_candidate);



if __name__ == "__main__":
    pass
