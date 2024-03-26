from dataclasses import dataclass
import .dag_generator as dag_generator
from .pick_weight import PickWeight
from typing import List
import random

@dataclass
class DimEq1GenTypePickProbability:
    dim_eq1_probability: PickWeight

@dataclass
class DimEq1GenRequirement:
    pick_probability: DimEq1GenTypePickProbability

@dataclass
class DimEq1GenInstruction:
    
    @classmethod
    def GetDAGGenClassToDerivedClassMap(cls):
        return kDAGGenClassToDerivedClass


@dataclass
class Nope(DimEq1GenInstruction):
    @classmethod
    def MakeRandomInstance(cls, requirement: DimEq1GenRequirement):
        return Nope()


@dataclass
class AddSinkTensor(DimEq1GenInstruction):
    sink_tensor_dim_eq1: bool

    @classmethod
    def MakeRandomInstance(cls, requirement: DimEq1GenRequirement):
        return AddSinkTensor(
            sink_tensor_dim_eq1=_GetRandomBool(requirement)
        )


@dataclass
class AddUnaryOp(DimEq1GenInstruction):
    source_tensor_dim_eq1: bool

    @classmethod
    def MakeRandomInstance(cls, requirement: DimEq1GenRequirement):
        return AddUnaryOp(
            source_tensor_dim_eq1=_GetRandomBool(requirement)
        )


@dataclass
class AddBinaryOp(DimEq1GenInstruction):
    lhs_source_tensor_dim_eq1: bool
    rhs_source_tensor_dim_eq1: bool

    @classmethod
    def MakeRandomInstance(cls, requirement: DimEq1GenRequirement):
        return AddBinaryOp(
            lhs_source_tensor_dim_eq1=_GetRandomBool(requirement),           rhs_source_tensor_dim_eq1=_GetRandomBool(requirement)
        )


@dataclass
class AddBinaryClone(DimEq1GenInstruction):

    @classmethod
    def MakeRandomInstance(cls, requirement: DimEq1GenRequirement):
        return AddBinaryClone()


@dataclass
class AddSourceOp(DimEq1GenInstruction):

    @classmethod
    def MakeRandomInstance(cls, requirement: DimEq1GenRequirement):
        return AddSourceOp()


kDAGGenClassToDerivedClass = {
    dag_generator.Nope: Nope,
    dag_generator.AddSinkTensor: AddSinkTensor,
    dag_generator.AddUnaryOp: AddUnaryOp,
    dag_generator.AddBinaryOp: AddBinaryOp,
    dag_generator.AddBinaryClone: AddBinaryClone,
    dag_generator.AddSourceOp: AddSourceOp,
}

class DimEq1Generator:
    def __init__(self, requirement: DimEq1GenRequirement):
        self.requirement = requirement
    
    # Instructions generating sink nodes of DAG are on put the front of list.
    def Generate(
        self,
        dag_gen_instructions: List["DAGGenInstruction"]
    ) -> Dict["DAGGenInstruction", "DimEq1GenInstruction"]:
        def CreateDimEq1GenInstruction(dag_gen_instruction):
            dag_gen_class = type(dag_gen_instruction)
            dim_eq1_gen_class = kDAGGenClassToDerivedClass[dag_gen_class]
            return dim_eq1_gen_class.MakeRandomInstance(self.requirement)
        return {
            x: CreateDimEq1GenInstruction(x)
            for x in dag_gen_instructions
        }

def _GetRandomBool(requirement: DimEq1GenRequirement):
    ratio = requirement.dim_eq1_probability.weight
    kRangeMax = 10000
    random_int = random.randomint(0, kRangeMax)
    return random_int < ratio * kRangeMax