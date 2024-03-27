from dataclasses import dataclass
from unit_test_case_spec import UnitTestCaseSpec
from instruction import Instruction
from code_gen_spec_inferer import CodeGenSpec
import op_call_code_gen
from op_call_code_gen import OpCallCodeGenRequirement,CodeGenContext
from script import Script
import dag_generator


@dataclass
class PaddleEagerScript(Script):
    pass

class Nope:
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
        rhs_output_tensor_name = ctx.code_gen_spec.rhs_output_tensor_name
        lhs_output_tensor_name = ctx.code_gen_spec.lhs_output_tensor_name
        return f"{rhs_output_tensor_name} = {lhs_output_tensor_name}"


class AddSourceOp:
    @classmethod
    def CodeGen(cls, ctx: CodeGenContext) -> str:
        op_name = ctx.instruction.op_name
        return getattr(op_call_code_gen, op_name).Paddle(ctx)


kDAGGenClassToCodeGenClass = {
    dag_generator.Nope: Nope,
    dag_generator.AddSinkTensor: AddSinkTensor,
    dag_generator.AddUnaryOp: AddUnaryOp,
    dag_generator.AddBinaryOp: AddBinaryOp,
    dag_generator.AddBinaryClone: AddBinaryClone,
    dag_generator.AddSourceOp: AddSourceOp,
}

class PaddleEagerGenerator:
    def __init__(self, requirement: OpCallCodeGenRequirement):
        self.requirement = requirement

    def Generate(
        self,
        unit_test_case_spec: UnitTestCaseSpec
    ) -> PaddleEagerScript:
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
        instruction_code_strs = reversed(instruction_code_strs)
        file_content = "\n".join(instruction_code_strs)
        return PaddleEagerScript(
            file_content=file_content
        )