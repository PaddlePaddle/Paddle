from typing as List
import .dag_gen_generator as dag_gen_generator
import .dim_eq1_generator as dim_eq1_generator
import .dims_eq1_generator as dims_eq1_generator
import .op_name_generator as op_name_generator
import .code_gen_spec_inferer as code_gen_spec_inferer
import .dims_eq1_signature_inferer as dims_eq1_signature_inferer
from .defensive_list import DList
from .instruction import IrGenerator
from .instruction_id import MakeUniqueInstructionId

def GenerateInstructions(
    dag_gen_requirement: dag_gen_generator.DAGGenRequirement,
    dims_eq1_gen_requirement: dims_eq1_generator.DimsEq1GenRequirement,
    op_name_gen_requirement: op_name_generator.OpNameGenRequirement,
    shape_rank: int
) -> List["Instruction"]:
    # generators
    dag_gen = dag_gen_generator.DAGGenerator(requirement=dag_gen_requirement)
    dims_eq1_gen = dims_eq1_generator.DimsEq1Generator(
        requirement=dims_eq1_gen_requirement
    )
    op_name_gen = op_name_generator.OpNameGenerator(
        requirement=op_name_gen_requirement
    )
    ir_gen = IrGenerator()
    # inferers
    dims_eq1_sig_inferer = dims_eq1_signature_inferer.DimsEq1SignatureInferer()
    # instructions
    dag_gen_instructions = dag_gen.Generate()
    instruction_ids = [
        MakeUniqueInstructionId() for _ in range(len(dag_gen_instructions))
    ]
    dims_eq1_gen_instructions = dims_eq1_gen.Generate(dag_gen_instructions)
    op_name_gen_instructions = op_name_gen.Generate(dag_gen_instructions)
    return ir_gen.Generate(
        dag_gen_instructions=dag_gen_instructions,
        instruction_ids=instruction_ids,
        dims_eq1_gen_instructions=dims_eq1_gen_instructions,
        op_name_gen_instructions=op_name_gen_instructions
    )


def PatchInstructions(
    instructions: List["Instruction"]
) -> List["Instruction"]:
    TODO()


def InferCodeGenSpecs(
    instructions: List["Instruction"]
) -> DList["Instruction", "CodeGenSpec"]:
    TODO()