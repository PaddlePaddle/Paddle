from typing as List
import .dag_gen_generator as dag_gen_generator
import .dim_eq1_generator as dim_eq1_generator
import .dims_eq1_generator as dims_eq1_generator
import .op_name_generator as op_name_generator
import .code_gen_spec_inferer as code_gen_spec_inferer
import .dims_eq1_signature_inferer as dims_eq1_signature_inferer
from .defensive_list import DList
from .instruction import IrGenerator
from .instruction_id import MakeUniqueInstructionId,InstructionId
from .dag_dims_eq1_patcher import DAGDimsEq1Patcher
from .op_name_patcher import OpNamePatcher

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
    instructions: List["Instruction"],
    op_name_gen_requirement: op_name_generator.OpNameGenRequirement
) -> List["Instruction"]:
    def GetComponentList(base_cls):
        return [
            instruction.GetComponent(base_cls)
            for instruction in instructions
        ]
    dag_gen_instructions = GetComponentList(dag_gen_generator.DAGGenInstruction)
    instruction_ids = GetComponentList(InstructionId)
    dims_eq1_instructions = GetComponentList(dims_eq1_generator.DimsEq1GenInstruction)
    op_name_instructions = GetComponentList(op_name_generator.OpNameGenInstruction)
    PatchDAGDimsEq1 = DAGDimsEq1Patcher().Patch
    new_dag_gen_instrs, new_instr_ids, new_dims_eq1_instrs = PatchDAGDimsEq1(
        dag_gen_instructions=dag_gen_instructions,
        instruction_ids=instruction_ids,
        dims_eq1_gen_instructions=dims_eq1_instructions
    )
    instruction_id2existed_op_name = {
        instruction_id:existed_op_name_instr
        for instruction_id, existed_op_name_instr in zip(
            instruction_ids, op_name_instructions
        )
    }
    PatchOpName = OpNameGenerator(op_name_gen_requirement).Patch
    new_op_name_instrs = PatchOpName(
        dag_gen_instructions=new_dag_gen_instrs,
        instruction_ids=new_instr_ids,
        instruction_id2existed_op_name=instruction_id2existed_op_name
    )
    GenerateInstructions = IrGenerator().Generate
    return GenerateInstructions(
        dag_gen_instructions=new_dag_gen_instrs,
        instruction_ids=new_instr_ids,
        dims_eq1_gen_instructions=new_dims_eq1_instrs,
        op_name_gen_instructions=new_op_name_instrs
    )

def InferCodeGenSpecs(
    instructions: List["Instruction"]
) -> DList["Instruction", "CodeGenSpec"]:
    TODO()