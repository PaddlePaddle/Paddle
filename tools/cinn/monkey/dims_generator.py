from dataclasses import dataclass
import .dag_generator as dag_generator
import .dim_generator as dim_generator
from .pick_weight import PickWeight
from typing import List

@dataclass
class DimsGenRequirement:
    jump_to_one_ratios: List[float]

@dataclass
class Nope:
    @classmethod
    def Merge(cls, dim_gen_instructions):
        return Nope()


@dataclass
class AddSinkTensor:
    source_tensor_dim_eq_one: List[bool]

    @classmethod
    def Merge(cls, dim_gen_instructions):
        return AddSinkTensor(
            source_tensor_dim_eq_one=tuple(
                dim_gen_instance.source_tensor_dim_eq_one
                for dim_gen_instance in dim_gen_instructions
            )
        )


@dataclass
class AddUnaryOp:
    source_tensor_dim_eq_one: List[bool]

    @classmethod
    def Merge(cls, dim_gen_instructions):
        return AddUnaryOp(
            source_tensor_dim_eq_one=tuple(
                dim_gen_instance.source_tensor_dim_eq_one
                for dim_gen_instance in dim_gen_instructions
            )
        )


@dataclass
class AddBinaryOp:
    lhs_source_tensor_dim_eq_one: List[bool]
    rhs_source_tensor_dim_eq_one: List[bool]

    @classmethod
    def Merge(cls, dim_gen_instructions):
        return AddBinaryOp(
            lhs_source_tensor_dim_eq_one=tuple(
                dim_gen_instance.lhs_source_tensor_dim_eq_one
                for dim_gen_instance in dim_gen_instructions
            ),
            rhs_source_tensor_dim_eq_one=tuple(
                dim_gen_instance.rhs_source_tensor_dim_eq_one
                for dim_gen_instance in dim_gen_instructions
            )
        )


@dataclass
class InsertBinaryOp:
    rhs_source_tensor_dim_eq_one: List[bool]

    @classmethod
    def Merge(cls, dim_gen_instructions):
        return InsertBinaryOp(
            rhs_source_tensor_dim_eq_one=tuple(
                dim_gen_instance.rhs_source_tensor_dim_eq_one
                for dim_gen_instance in dim_gen_instructions
            )
        )


@dataclass
class AddBinaryClone:

    @classmethod
    def Merge(cls, dim_gen_instructions):
        return AddBinaryClone()


@dataclass
class AddSourceOp:

    @classmethod
    def Merge(cls, dim_gen_instructions):
        return AddSourceOp()


DimsGenInstruction = ( Nope
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

class DimsGenerator:
    def __init__(
        self,
        requirement: DimsGenRequirement
    ):
        self.requirement = requirement
        def MakeDimGenerator(ratio):
            return type(self)._MakeDimGenerator(ratio)
        self.dim_generators = [
            MakeDimGenerator(ratio)
            for ratio in requirement.jump_to_one_ratios
        ]

    # Instructions generating sink nodes of DAG are on put the front of list.
    def Generate(
        self,
        dag_gen_instructions: List[dag_generator.DAGGenInstruction]
    ) -> List[DimsGenInstruction]:
        dim_gen_instructions = [
            dim_generator.Generate(dag_gen_instructions)
            for dim_generator in dim_generators
        ]
        instructions = zip(
            dag_gen_instructions,
            zip(dim_gen_instructions)
        )
        def MakeDimsGenInstruction(dag_gen_instruction, dim_gen_instructions):
            dag_gen_class = type(dag_gen_instruction)
            return dag_gen_class.Merge(dim_gen_instructions)
        return [
            MakeDimsGenInstruction(dag_gen_instruction, dim_gen_instructions)
            for dag_gen_instruction, dim_gen_instructions in instructions
        ]

    @classmethod
    def _MakeDimGenerator(
        cls,
        jump_to_one_ratio
    ):
        pick_probability = dim_generator.DimGenTypePickProbability(
            jump_to_one_ratio=PickWeight(jump_to_one_ratio)
        )
        requirement = dim_generator.DimGenRequirement(
            pick_probability
        )
        return dim_generator.DimGenerator(requirement)
