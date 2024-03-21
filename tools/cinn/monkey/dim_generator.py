from dataclasses import dataclass
import .dag_generator as dag_generator
from .pick_weight import PickWeight
from typing import List
import random

@dataclass
class DimGenTypePickProbability:
    jump_to_one_ratio: PickWeight

@dataclass
class DimGenRequirement:
    pick_probability: DimGenTypePickProbability

@dataclass
class Nope:
    @classmethod
    def MakeRandomInstance(cls, requirement: DimGenRequirement):
        return Nope()


@dataclass
class AddSinkTensor:
    source_tensor_dim_eq_one: bool

    @classmethod
    def MakeRandomInstance(cls, requirement: DimGenRequirement):
        return AddSinkTensor(
            source_tensor_dim_eq_one=_GetRandomBool(requirement)
        )


@dataclass
class AddUnaryOp:
    source_tensor_dim_eq_one: bool

    @classmethod
    def MakeRandomInstance(cls, requirement: DimGenRequirement):
        return AddUnaryOp(
            source_tensor_dim_eq_one=_GetRandomBool(requirement)
        )


@dataclass
class AddBinaryOp:
    lhs_source_tensor_dim_eq_one: bool
    rhs_source_tensor_dim_eq_one: bool

    @classmethod
    def MakeRandomInstance(cls, requirement: DimGenRequirement):
        return AddBinaryOp(
            lhs_source_tensor_dim_eq_one=_GetRandomBool(requirement),           rhs_source_tensor_dim_eq_one=_GetRandomBool(requirement)
        )


@dataclass
class InsertBinaryOp:
    rhs_source_tensor_dim_eq_one: bool

    @classmethod
    def MakeRandomInstance(cls, requirement: DimGenRequirement):
        return InsertBinaryOp(
            rhs_source_tensor_dim_eq_one=_GetRandomBool(requirement)
        )


@dataclass
class AddBinaryClone:

    @classmethod
    def MakeRandomInstance(cls, requirement: DimGenRequirement):
        return AddBinaryClone()


@dataclass
class AddSourceOp:

    @classmethod
    def MakeRandomInstance(cls, requirement: DimGenRequirement):
        return AddSourceOp()


DimGenInstruction = ( Nope
                    | AddSinkTensor
                    | AddUnaryOp
                    | AddBinaryOp
                    | InsertBinaryOp
                    | AddBinaryClone
                    | AddSourceOp
                    )

kDAGGenClassToDimGenClassMap = {
    dag_generator.Nope: Nope,
    dag_generator.AddSinkTensor: AddSinkTensor,
    dag_generator.AddUnaryOp: AddUnaryOp,
    dag_generator.AddBinaryOp: AddBinaryOp,
    dag_generator.InsertBinaryOp: InsertBinaryOp,
    dag_generator.AddBinaryClone: AddBinaryClone,
    dag_generator.AddSourceOp: AddSourceOp,
}

class DimGenerator:
    def __init__(self, requirement: DimGenRequirement):
        self.requirement = requirement
    
    # Instructions generating sink nodes of DAG are on put the front of list.
    def Generate(
        self,
        dag_gen_instructions: List[dag_generator.DAGGenInstruction]
    ) -> List[DimGenInstruction]:
        def CreateDimGenInstruction(dag_gen_instruction):
            dag_gen_class = type(dag_gen_instruction)
            dim_gen_class = kDAGGenClassToDimGenClassMap[dag_gen_class]
            return dim_gen_class.MakeRandomInstance(self.requirement)
        return [
            CreateDimGenInstruction(dag_gen_instruction)
            for dag_gen_instruction in dag_gen_instructions
        ]

def _GetRandomBool(requirement: DimGenRequirement):
    ratio = requirement.jump_to_one_ratio.weight
    kRangeMax = 10000
    random_int = random.randomint(0, kRangeMax)
    return random_int < ratio * kRangeMax