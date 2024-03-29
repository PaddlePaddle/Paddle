from dataclasses import dataclass
from unit_test_case_spec import UnitTestCaseSpec
from instruction import Instruction
from code_gen_spec_inferer import CodeGenSpec
import op_call_code_gen
from op_call_code_gen import OpCallCodeGenRequirement,CodeGenContext
from script import Script
import dag_generator
from typing import List


class Nope:
    @classmethod
    def CodeGen(cls, ctx: CodeGenContext) -> str:
        return ""


class AddSourceTensor:
    @classmethod
    def CodeGen(cls, ctx: CodeGenContext) -> str:
        return ""


class AddSinkTensor:
    @classmethod
    def CodeGen(cls, ctx: CodeGenContext) -> str:
        return ""


class AddUnaryOp:
    @classmethod
    def CodeGen(cls, ctx: CodeGenContext) -> str:
        op_name = ctx.instruction.op_name
        return getattr(op_call_code_gen, op_name).Paddle(ctx)

class AddBinaryOp:
    @classmethod
    def CodeGen(cls, ctx: CodeGenContext) -> str:
        op_name = ctx.instruction.op_name
        return getattr(op_call_code_gen, op_name).Paddle(ctx)


class AddBinaryClone:
    @classmethod
    def CodeGen(cls, ctx: CodeGenContext) -> str:
        input_tensor_name = ctx.code_gen_spec.input_tensor_name
        lhs_output_tensor_name = ctx.code_gen_spec.lhs_output_tensor_name
        rhs_output_tensor_name = ctx.code_gen_spec.rhs_output_tensor_name
        def AllSame():
            return (
                input_tensor_name == lhs_output_tensor_name
                and input_tensor_name == rhs_output_tensor_name
            )
        if AllSame():
            return ""
        return f"{lhs_output_tensor_name}, {rhs_output_tensor_name} = ({input_tensor_name}, {input_tensor_name})"


class AddSourceOp:
    @classmethod
    def CodeGen(cls, ctx: CodeGenContext) -> str:
        op_name = ctx.instruction.op_name
        return getattr(op_call_code_gen, op_name).Paddle(ctx)


kDAGGenClassToCodeGenClass = {
    dag_generator.Nope: Nope,
    dag_generator.AddSourceTensor: AddSourceTensor,
    dag_generator.AddSinkTensor: AddSinkTensor,
    dag_generator.AddUnaryOp: AddUnaryOp,
    dag_generator.AddBinaryOp: AddBinaryOp,
    dag_generator.AddBinaryClone: AddBinaryClone,
    dag_generator.AddSourceOp: AddSourceOp,
}

class PaddleStmtGenerator:
    def __init__(self, requirement: OpCallCodeGenRequirement):
        self.requirement = requirement

    def Generate(
        self,
        unit_test_case_spec: UnitTestCaseSpec
    ) -> List[str]:
        def GenerateInstructionCodeStr(instruction, code_gen_spec):
            dag_gen_class = instruction.GetDAGGenClass()
            cls = kDAGGenClassToCodeGenClass[dag_gen_class]
            return cls.CodeGen(CodeGenContext(
                requirement=self.requirement,
                instruction=instruction,
                code_gen_spec=code_gen_spec
            ))

        instruction_code_strs = [
            GenerateInstructionCodeStr(*pair)
            for pair in unit_test_case_spec.pached_instruction_code_gen_spec.Unguard()
        ]
        return instruction_code_strs
