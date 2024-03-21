from dataclasses import dataclass
import .dag_generator as dag_generator
from .pick_weight import PickWeight
from typing import List
import random

@dataclass
class DimEq1GenTypePickProbability:
    jump_to_one_ratio: PickWeight

@dataclass
class DimEq1GenRequirement:
    pick_probability: DimEq1GenTypePickProbability

@dataclass
class Nope:
    @classmethod
    def MakeRandomInstance(cls, requirement: DimEq1GenRequirement):
        return Nope()


@dataclass
class AddSinkTensor:
    source_tensor_dim_eq1: bool

    @classmethod
    def MakeRandomInstance(cls, requirement: DimEq1GenRequirement):
        return AddSinkTensor(
            source_tensor_dim_eq1=_GetRandomBool(requirement)
        )


@dataclass
class AddUnaryOp:
    source_tensor_dim_eq1: bool

    @classmethod
    def MakeRandomInstance(cls, requirement: DimEq1GenRequirement):
        return AddUnaryOp(
            source_tensor_dim_eq1=_GetRandomBool(requirement)
        )


@dataclass
class AddBinaryOp:
    lhs_source_tensor_dim_eq1: bool
    rhs_source_tensor_dim_eq1: bool

    @classmethod
    def MakeRandomInstance(cls, requirement: DimEq1GenRequirement):
        return AddBinaryOp(
            lhs_source_tensor_dim_eq1=_GetRandomBool(requirement),           rhs_source_tensor_dim_eq1=_GetRandomBool(requirement)
        )


@dataclass
class InsertBinaryOp:
    rhs_source_tensor_dim_eq1: bool

    @classmethod
    def MakeRandomInstance(cls, requirement: DimEq1GenRequirement):
        return InsertBinaryOp(
            rhs_source_tensor_dim_eq1=_GetRandomBool(requirement)
        )


@dataclass
class AddBinaryClone:

    @classmethod
    def MakeRandomInstance(cls, requirement: DimEq1GenRequirement):
        return AddBinaryClone()


@dataclass
class AddSourceOp:

    @classmethod
    def MakeRandomInstance(cls, requirement: DimEq1GenRequirement):
        return AddSourceOp()


DimEq1GenInstruction = ( Nope
                    | AddSinkTensor
                    | AddUnaryOp
                    | AddBinaryOp
                    | InsertBinaryOp
                    | AddBinaryClone
                    | AddSourceOp
                    )

kDAGGenClassToDimEq1GenClassMap = {
    dag_generator.Nope: Nope,
    dag_generator.AddSinkTensor: AddSinkTensor,
    dag_generator.AddUnaryOp: AddUnaryOp,
    dag_generator.AddBinaryOp: AddBinaryOp,
    dag_generator.InsertBinaryOp: InsertBinaryOp,
    dag_generator.AddBinaryClone: AddBinaryClone,
    dag_generator.AddSourceOp: AddSourceOp,
}

class DimEq1Generator:
    def __init__(self, requirement: DimEq1GenRequirement):
        self.requirement = requirement
    
    # Instructions generating sink nodes of DAG are on put the front of list.
    def Generate(
        self,
        dag_gen_instructions: List[dag_generator.DAGGenInstruction]
    ) -> List[DimEq1GenInstruction]:
        def CreateDimEq1GenInstruction(dag_gen_instruction):
            dag_gen_class = type(dag_gen_instruction)
            dim_eq1_gen_class = kDAGGenClassToDimEq1GenClassMap[dag_gen_class]
            return dim_eq1_gen_class.MakeRandomInstance(self.requirement)
        return [
            CreateDimEq1GenInstruction(dag_gen_instruction)
            for dag_gen_instruction in dag_gen_instructions
        ]

def _GetRandomBool(requirement: DimEq1GenRequirement):
    ratio = requirement.jump_to_one_ratio.weight
    kRangeMax = 10000
    random_int = random.randomint(0, kRangeMax)
    return random_int < ratio * kRangeMax