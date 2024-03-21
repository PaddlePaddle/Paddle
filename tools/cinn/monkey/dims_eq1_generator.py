from dataclasses import dataclass
import .dag_generator as dag_generator
import .dim_eq1_generator as dim_eq1_generator
from .pick_weight import PickWeight
from typing import List

@dataclass
class DimsEq1GenRequirement:
    jump_to_one_ratios: List[float]

@dataclass
class Nope:
    @classmethod
    def Merge(cls, dim_eq1_gen_instructions: List[dim_eq1_generator.DimEq1GenInstruction]):
        return Nope()


@dataclass
class AddSinkTensor:
    source_tensor_dim_eq1: List[bool]

    @classmethod
    def Merge(cls, dim_eq1_gen_instructions: List[dim_eq1_generator.DimEq1GenInstruction]):
        return AddSinkTensor(
            source_tensor_dim_eq1=tuple(
                dim_eq1_gen_instance.source_tensor_dim_eq1
                for dim_eq1_gen_instance in dim_eq1_gen_instructions
            )
        )


@dataclass
class AddUnaryOp:
    source_tensor_dim_eq1: List[bool]

    @classmethod
    def Merge(cls, dim_eq1_gen_instructions: List[dim_eq1_generator.DimEq1GenInstruction]):
        return AddUnaryOp(
            source_tensor_dim_eq1=tuple(
                dim_eq1_gen_instance.source_tensor_dim_eq1
                for dim_eq1_gen_instance in dim_eq1_gen_instructions
            )
        )


@dataclass
class AddBinaryOp:
    lhs_source_tensor_dim_eq1: List[bool]
    rhs_source_tensor_dim_eq1: List[bool]

    @classmethod
    def Merge(cls, dim_eq1_gen_instructions: List[dim_eq1_generator.DimEq1GenInstruction]):
        return AddBinaryOp(
            lhs_source_tensor_dim_eq1=tuple(
                dim_eq1_gen_instance.lhs_source_tensor_dim_eq1
                for dim_eq1_gen_instance in dim_eq1_gen_instructions
            ),
            rhs_source_tensor_dim_eq1=tuple(
                dim_eq1_gen_instance.rhs_source_tensor_dim_eq1
                for dim_eq1_gen_instance in dim_eq1_gen_instructions
            )
        )


@dataclass
class InsertBinaryOp:
    rhs_source_tensor_dim_eq1: List[bool]

    @classmethod
    def Merge(cls, dim_eq1_gen_instructions: List[dim_eq1_generator.DimEq1GenInstruction]):
        return InsertBinaryOp(
            rhs_source_tensor_dim_eq1=tuple(
                dim_eq1_gen_instance.rhs_source_tensor_dim_eq1
                for dim_eq1_gen_instance in dim_eq1_gen_instructions
            )
        )


@dataclass
class AddBinaryClone:

    @classmethod
    def Merge(cls, dim_eq1_gen_instructions: List[dim_eq1_generator.DimEq1GenInstruction]):
        return AddBinaryClone()


@dataclass
class AddSourceOp:

    @classmethod
    def Merge(cls, dim_eq1_gen_instructions: List[dim_eq1_generator.DimEq1GenInstruction]):
        return AddSourceOp()


DimsEq1GenInstruction = ( Nope
                     | AddSinkTensor
                     | AddUnaryOp
                     | AddBinaryOp
                     | InsertBinaryOp
                     | AddBinaryClone
                     | AddSourceOp
                     )

kDAGGenClassToDimsEq1GenClassMap = {
    dag_generator.Nope: Nope,
    dag_generator.AddSinkTensor: AddSinkTensor,
    dag_generator.AddUnaryOp: AddUnaryOp,
    dag_generator.AddBinaryOp: AddBinaryOp,
    dag_generator.InsertBinaryOp: InsertBinaryOp,
    dag_generator.AddBinaryClone: AddBinaryClone,
    dag_generator.AddSourceOp: AddSourceOp,
}

class DimsEq1Generator:
    def __init__(
        self,
        requirement: DimsEq1GenRequirement
    ):
        self.requirement = requirement
        def MakeDimEq1Generator(ratio):
            return type(self)._MakeDimEq1Generator(ratio)
        self.dim_eq1_generators = [
            MakeDimEq1Generator(ratio)
            for ratio in requirement.jump_to_one_ratios
        ]

    # Instructions generating sink nodes of DAG are on put the front of list.
    def Generate(
        self,
        dag_gen_instructions: List[dag_generator.DAGGenInstruction]
    ) -> List[DimsEq1GenInstruction]:
        dim_eq1_gen_instructions = [
            dim_eq1_generator.Generate(dag_gen_instructions)
            for dim_eq1_generator in dim_eq1_generators
        ]
        instructions = zip(
            dag_gen_instructions,
            zip(dim_eq1_gen_instructions)
        )
        def MakeDimsEq1GenInstruction(dag_gen_instruction, dim_eq1_gen_instructions):
            dag_gen_class = type(dag_gen_instruction)
            return dag_gen_class.Merge(dim_eq1_gen_instructions)
        return [
            MakeDimsEq1GenInstruction(dag_gen_instruction, dim_eq1_gen_instructions)
            for dag_gen_instruction, dim_eq1_gen_instructions in instructions
        ]

    @classmethod
    def _MakeDimEq1Generator(
        cls,
        jump_to_one_ratio
    ):
        pick_probability = dim_eq1_generator.DimEq1GenTypePickProbability(
            jump_to_one_ratio=PickWeight(jump_to_one_ratio)
        )
        requirement = dim_eq1_generator.DimEq1GenRequirement(
            pick_probability
        )
        return dim_eq1_generator.DimEq1Generator(requirement)
