from typing import *
import logging

from .candidate_generator import *
from .bench_func import BaseBenchFunc

class CandidateSearcher:
    def __init__(
        self,
        bench_funcs: List[BaseBenchFunc],
        candidate_generator: BaseCandidateGenerator
    ):
        self.candidate_generator = candidate_generator
        self.bench_funcs = bench_funcs
        self.records: Dict[float, List[int]] = {}

    def search(self, is_search_minimum: bool = True) -> Tuple[float, List[int]]:
        candidates = self.candidate_generator.candidates()
        print(f"candidates = {candidates}")
        for candidate in candidates:
            # print candidate info
            logging.info("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
            for i, param_name in enumerate(self.candidate_generator.param_names()):
                logging.info(f"{param_name}: {candidate[i]}")
            
            total_score = 0.0
            for bench_func in self.bench_funcs:
                total_score += bench_func(candidate)
            self.records[total_score] = candidate
            logging.info(f"    total_score = {total_score}")
        
        if not self.records:
            return (float('inf') if is_search_minimum else float('-inf'), [])
        
        sorted_scores = sorted(self.records.items(), key=lambda x: x[0])
        return sorted_scores[0] if is_search_minimum else sorted_scores[-1]