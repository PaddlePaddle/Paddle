from dataclasses import dataclass
import .dag_generator as dag_generator
import .dims_eq1_generator as dims_eq1_generator
from .dims_eq1_signature_inferer import DimsEq1SignatureInferer
from .pick_weight import PickWeight
from typing import List, Generator
from collections import namedtuple
from .defensive_list import DList


class DAGNumSourcesPatcher:
    def __init__(self, num_sources: int, dag_tag: str):
        self.num_sources = num_sources
        self.dag_tag = dag_tag

    def Patch(
        self,
        guarded_num_sources_sigs: DList["DAGGenInstruction", "NumSourcesSignature"],
    ) -> List["DAGGenInstruction"]:
        TODO()
