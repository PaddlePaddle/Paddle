from dataclasses import dataclass
from collections import namedtuple
from typing import Generator, Union
from .defensive_list import DList
import .dag_gen_generator as dag_gen_generator
import .dim_eq1_generator as dim_eq1_generator
import .dims_eq1_generator as dims_eq1_generator
import .op_name_generator as op_name_generator
import .shape_signature_inferer as shape_signature_inferer
import .instruction_util as instruction_util

@dataclass
class SearchSpec:
    total_try_cnt: int = 1024
    dag_gen_requirement: dag_gen_generator.DAGGenRequirement
    dims_eq1_gen_requirement: dims_eq1_generator.DimsEq1GenRequirement
    op_name_gen_requirement: op_name_generator.OpNameGenRequirement
    dim_size_requirement: shape_signature_inferer.DimSizeRequirement

@dataclass
class UnitTestCaseSpec:
    instructions: List["Instruction"]
    patched_instructions: List["Instruction"]
    code_gen_spec: DList["Instruction", "CodeGenSpec"]

@dataclass
class ExecResult:
    is_error: bool

def GenerateUnitTestCaseSpec(
    search_spec: SearchSpec
) -> UnitTestCaseSpec:
    
    instructions = instruction_util.GenerateInstructions(
        dag_gen_requirement=search_spec.dag_gen_requirement,
        dims_eq1_gen_requirement=search_spec.dims_eq1_gen_requirement,
        op_name_gen_requirement=search_spec.op_name_gen_requirement,
    )
    patched_patched_instructions = instruction_util.PatchInstructions(
        instructions=instructions
    )
    code_gen_spec = instruction_util.InferCodeGenSpecs(
        instructions=patched_patched_instructions
    )
    return UnitTestCaseSpec(
        instructions=instructions,
        patched_patched_instructions=patched_patched_instructions,
        code_gen_spec=code_gen_spec
    )

def CodeGen(
    unit_test_case_spec: UnitTestCaseSpec
) -> str:
    TODO()

def ExecCode(code: str) -> ExecResult:
    TODO()

def GenerateAndRun(
    search_spec: SearchSpec
) -> Tuple[UnitTestCaseSpec, ExecResult]:
    unit_test_case_spec = GenerateUnitTestCaseSpec(search_spec)
    code = CodeGen(unit_test_case_spec)
    return unit_test_case_spec, ExecCode(code)

@dataclass
class BinarySearchResult:
    bad_case: UnitTestCaseSpec,
    simplier_case: UnitTestCaseSpec,
    simplier_case_exec_result: ExecResult

def BinarySearchBug(
    unit_test_case_spec: UnitTestCaseSpec
) -> BinarySearchResult:
    TODO()

@dataclass
class SmallDifferedGoodAndBadCases:
    good_case: UnitTestCaseSpec
    bad_case: UnitTestCaseSpec


@dataclass
class FailedSimplestCase:
    failed_case: UnitTestCaseSpec

def GenerateBadCasesAndBinarySearchBug(
    search_spec: SearchSpec
) -> Generator[Union[SmallDifferedGoodAndBadCases, FailedSimplestCase]]:
    for i in range(search_spec.total_try_cnt):
        unit_test_case_spec, exec_result = GenerateAndRun(search_spec)
        if not exec_result.is_error:
            continue
        bad_case, simplier_case, simplier_case_exec_result = BinarySearchBug(
            unit_test_case_spec
        )
        if simplier_case_exec_result.is_error:
            yield FailedSimplestCase(
                failed_case=simplier_case
            )
        else:
            yield SmallDifferedGoodAndBadCases(
                good_case=simplier_case,
                bad_case=bad_case
            )