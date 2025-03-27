from dataclasses import dataclass
import os
import math
import statistics
from typing import *
import logging, json

from paddle.base import core
from paddle.base.core import cinn
from paddle.cinn import common
from paddle.cinn import autotuner
from .bench_func import WeightedBenchFunc
from .candidate_generator import BaseCandidateGenerator
from .candidate_searcher import CandidateSearcher
import utils

class OpsSearcher:
    def __init__(
        self, 
        program_builder,
        candidate_generator: BaseCandidateGenerator,
    ):
        self.program_builder = program_builder
        self.candidate_generator = candidate_generator

    def gen_csv_file_path(self, target, iter_space_type):
        dirname = ""
        filename = ""
        for key, value in iter_space_type:
            dirname += f"{key}_"
            filename += f"{key}{value}_"
        
        DIR_SUFFIX = "_AutoTune"
        dirname = dirname[:-1] + DIR_SUFFIX
        filename = filename[:-1]
        
        def check_exist(test_path):
            if not os.path.exists(test_path):
                try:
                    os.makedirs(test_path)
                except:
                    raise Exception(f"Cannot create directory: {test_path}")
                    
        root_path = os.getcwd()
            
        target_str = f"{target.arch_str()}_{target.device_name_str()}"
        
        check_exist(root_path)
        check_exist(os.path.join(root_path, target_str))
        check_exist(os.path.join(root_path, target_str, dirname))
        
        return os.path.join(root_path, target_str, dirname, f"{filename}.csv")

    def build_program(self, is_spatial_dynamic: bool, is_reduce_dynamic: bool, s_dimension_lower: int, r_dimension_lower: int):
        return self.program_builder(
            s_dimension_lower if not is_spatial_dynamic else -1,
            r_dimension_lower if not is_reduce_dynamic else -1)[0]
        
    def create_bucket(self, s_dimension_lower, spatial_tile_width, r_dimension_lower, reduce_tile_width, is_spatial_dynamic, is_reduce_dynamic):
        bucket_info = autotuner.tuner_config.BucketInfo()
        # bucket_info.space = [] if not bucket_info.space else bucket_info.space
        space = []
        space.append(autotuner.tuner_config.Dimension(
            s_dimension_lower,
            s_dimension_lower + spatial_tile_width - 1,
            "S",
            is_spatial_dynamic
        ))
        space.append(autotuner.tuner_config.Dimension(
            r_dimension_lower,
            r_dimension_lower + reduce_tile_width - 1,
            "R",
            is_reduce_dynamic
        ))
        bucket_info.space = space
        print(bucket_info)
        return bucket_info

    def write_bucket_info(
            self,
            output_file,
            iter_space_type: List[Tuple[str, str]],
            bucket_info: cinn.autotuner.tuner_config.BucketInfo
    ):
        output_file.write(" { ")
        for i, space in enumerate(iter_space_type):
            bucket = bucket_info.space[i]
            output_file.write(f"{space[0]}_{space[1]}: {bucket.lower_bound}-{bucket.upper_bound} ")
        output_file.write(" }\n")

    def get_window_size(self, dimension_lower: int) -> int:
        if dimension_lower <= 2:
            return 126
        elif dimension_lower <= 128:
            return 384
        elif dimension_lower <= 512:
            return 512
        elif dimension_lower <= 1024:
            return 1024
        elif dimension_lower <= 2048:
            return 2048
        elif dimension_lower <= 4096:
            return 4096
        elif dimension_lower <= 8192:
            return 8192
        return 8192


    def search_window(
        self, 
        output_file,
        graph_num: int,
        s_dimension_lower: int,
        spatial_tile_width: int, 
        r_dimension_lower: int,
        reduce_tile_width: int,
        is_spatial_dynamic: bool,
        is_reduce_dynamic: bool,
        s_weight: float,
        r_weight: float,
        sampling_prob: float,
        max_sampling_times: int,
        repeats: int,
        iter_space_type: List[Tuple[str, str]]
    ):
        s_weights = [s_weight] * spatial_tile_width
        r_weights = [r_weight] * reduce_tile_width
        
        logging.info(f"spatial tile dimension lower bound = {s_dimension_lower}, "
                    f"reduce tile dimension lower bound = {r_dimension_lower}")

        # 测试逻辑实现
        program = self.build_program(is_spatial_dynamic, is_reduce_dynamic,
                            s_dimension_lower, r_dimension_lower)
        
        bucket_info = self.create_bucket(s_dimension_lower, spatial_tile_width,
                                r_dimension_lower, reduce_tile_width,
                                is_spatial_dynamic, is_reduce_dynamic)
        
        bench_func = WeightedBenchFunc(
            program, bucket_info, sampling_prob, max_sampling_times, repeats, [s_weights, r_weights])

        candidate_searcher = CandidateSearcher(
            [bench_func],
            self.candidate_generator
        )

        # write this bucket info
        self.write_bucket_info(output_file, iter_space_type, bucket_info)

        logging.info("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        logging.info(f"Start search for {s_dimension_lower}_{s_dimension_lower + spatial_tile_width - 1} and {r_dimension_lower}_{r_dimension_lower + reduce_tile_width - 1}")
        logging.info("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        # 寻找最佳性能
        autotuner.tuner_config._env_set_tile_config_policy("search")
        best = candidate_searcher.search()
        best_score = best[0] / graph_num
        best_candidate = json.dumps(utils.candidate_join(
            self.candidate_generator.param_names(), best[1]
        ), indent=4)
                    
        # 输出结果
        logging.info(f"Best score: {best_score}")
        logging.info(f"Best candidate: {best_candidate}")
        logging.info("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        
        # 写入CSV文件
        output_file.write(f"best_score: {best_score:.3f} \n")
        output_file.write(f"best_candidate: \n{best_candidate} \n")
        output_file.write(f"=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")
        output_file.flush()

    def search(
        self, 
        spatial_left_bound: int = 2,
        spatial_right_bound: int = 1024, 
        reduce_left_bound: int = 2,
        reduce_right_bound: int = 1024,
        is_spatial_dynamic: bool = True,
        is_reduce_dynamic: bool = True,
        test_single_large: bool = False
    ):

        # 配置常量
        THREADS_PER_WARP = 32
        MAX_THREADS_PER_BLOCK = 1024
        S_W = 0.05
        R_W = 0.05
        SAMPLING_PROB = 1.0
        MAX_SAMPLING_TIMES = 360
        REPEATS = 5
        GRAPH_NUM = 25
        
        
        # 设置权重
        s_weight = S_W if is_spatial_dynamic else 1.0
        r_weight = R_W if is_reduce_dynamic else 1.0
        
        iter_space_type = [
            ("S", "dynamic" if is_spatial_dynamic else "static"),
            ("R", "dynamic" if is_reduce_dynamic else "static")
        ]
        
        # 获取配置并打开输出文件
        dump_path = self.gen_csv_file_path(common.DefaultTarget(), iter_space_type)
        
        spatial_window_size = 0
        reduce_window_size = 0
        spatial_tile_width = 0
        reduce_tile_width = 0
        with open(dump_path, 'a') as output_file:
            # 测试主循环
            # outside `while` loop init
            s_dimension_lower = spatial_left_bound
            while s_dimension_lower < spatial_right_bound or \
                s_dimension_lower == spatial_right_bound and spatial_left_bound == spatial_right_bound:
                
                spatial_window_size = self.get_window_size(s_dimension_lower)
                spatial_tile_width = spatial_window_size if is_spatial_dynamic else 1
                
                # inside `while` loop init
                r_dimension_lower = reduce_left_bound
                while r_dimension_lower < reduce_right_bound or \
                    r_dimension_lower == reduce_right_bound and reduce_left_bound == reduce_right_bound:
                    
                    reduce_window_size = self.get_window_size(r_dimension_lower)
                    reduce_tile_width = reduce_window_size if is_reduce_dynamic else 1
                    
                    self.search_window(
                        output_file, GRAPH_NUM,
                        s_dimension_lower, spatial_tile_width,
                        r_dimension_lower, reduce_tile_width,
                        is_spatial_dynamic, is_reduce_dynamic,
                        s_weight, r_weight, SAMPLING_PROB,
                        MAX_SAMPLING_TIMES, REPEATS,
                        iter_space_type
                    )
                    # inside `while` variable change
                    r_dimension_lower += reduce_window_size

                # outside `while` variable change
                s_dimension_lower += spatial_window_size

            # # 测试大规模区域(如果需要)
            # if test_single_large:
            #     # 实现大规模区域测试逻辑
            #     pass


