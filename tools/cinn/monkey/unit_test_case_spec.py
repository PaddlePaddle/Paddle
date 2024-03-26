from dataclasses import dataclass
from collections import namedtuple
from typing import Generator, Union
from .defensive_list import DList
import .dag_generator as dag_generator
import .dim_eq1_generator as dim_eq1_generator
import .dims_eq1_generator as dims_eq1_generator
import .op_name_generator as op_name_generator
import .tensor_name_generator as tensor_name_generator
import .shape_signature_inferer as shape_signature_inferer
import .instruction_util as instruction_util
import .op_call_code_gen as op_call_code_gen


@dataclass
class UnitTestCaseRequirement:
    dag_gen_requirement: dag_generator.DAGGenRequirement
    dims_eq1_gen_requirement: dims_eq1_generator.DimsEq1GenRequirement
    op_name_gen_requirement: op_name_generator.OpNameGenRequirement
    tensor_name_gen_requirement: tensor_name_generator.TensorNameGenRequirement
    dim_size_requirement: shape_signature_inferer.DimSizeRequirement
    code_gen_requirement: op_call_code_gen.CodeGenRequirement


@dataclass
class UnitTestCaseSpec:
    instructions: List["Instruction"]
    pached_instruction_code_gen_spec: DList["Instruction", "CodeGenSpec"]


def GenerateRandomUnitTestCaseSpec(
    requirement: UnitTestCaseRequirement
) -> UnitTestCaseSpec:
    instructions = instruction_util.GenerateInstructions(
        dag_gen_requirement=requirement.dag_gen_requirement,
        dims_eq1_gen_requirement=requirement.dims_eq1_gen_requirement,
        op_name_gen_requirement=requirement.op_name_gen_requirement,
    )
    patched_instructions = instruction_util.PatchInstructions(
        instructions=instructions,
        op_name_gen_requirement=requirement.op_name_gen_requirement
    )
    pached_instruction_code_gen_spec = instruction_util.InferCodeGenSpecs(
        instructions=patched_instructions,
        dim_size_requirement=requirement.dim_size_requirement
    )
    return UnitTestCaseSpec(
        instructions=instructions,
        pached_instruction_code_gen_spec=pached_instruction_code_gen_spec
    )
