from dataclasses import dataclass
import .dag_generator as dag_generator
import .dims_eq1_generator as dims_eq1_generator
from .dims_eq1_signature_inferer import DimsEq1SignatureInferer
from .pick_weight import PickWeight
from typing import List, Generator
from collections import namedtuple
from .defensive_list import DList

@dataclass
class NumSourcesSignature
    current_num_sources : int

class NumSourcesSignatureInferer:
    def __init__(self):
        pass

    def Infer(
        self,
        dag_gen_instructions: List["DAGGenInstruction"],
    ) -> DList["DAGGenInstruction", NumSourcesSignature]:
        current_num_sources = 0
        def MakeNumSourcesSignature(dag_gen_instruction):
            nonlocal current_num_sources
            ret = current_num_sources
            cls = type(dag_gen_instruction)
            current_num_sources += cls.GetDeltaNumSourceTensors()
            return NumSourcesSignature(
                current_num_sources=current_num_sources
            )
        num_source_tensors = [
             MakeNumSourcesSignature(dag_gen_instruction)
            for dag_gen_instruction in dag_gen_instructions
        ]
        return DList(
            dag_gen_instructions,
            num_source_tensors,
            lambda instr: instr.GetHashValue()
        )