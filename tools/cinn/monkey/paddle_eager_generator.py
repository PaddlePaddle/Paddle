from dataclasses import dataclass
from unit_test_case_spec import UnitTestCaseSpec
from op_call_code_gen import OpCallCodeGenRequirement
from script import Script
from paddle_stmt_generator import PaddleStmtGenerator


@dataclass
class PaddleEagerScript(Script):
    pass

class PaddleEagerGenerator:
    def __init__(self, requirement: OpCallCodeGenRequirement):
        self.requirement = requirement
        self.stmt_generator = PaddleStmtGenerator(requirement)

    def Generate(
        self,
        unit_test_case_spec: UnitTestCaseSpec
    ) -> PaddleEagerScript:
        stmts = self.stmt_generator.Generate(unit_test_case_spec)
        instruction_code_strs = reversed(stmts)
        file_content = "\n".join(instruction_code_strs)
        return PaddleEagerScript(
            file_content=file_content
        )