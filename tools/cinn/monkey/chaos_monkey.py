from dataclasses import dataclass
from collections import namedtuple
from typing import Generator, Union
from defensive_list import DList
from dag_generator import PickWeight
from script import Script
import dag_generator
import dim_eq1_generator as dim_eq1_generator
import dims_eq1_generator as dims_eq1_generator
import op_name_generator as op_name_generator
import tensor_name_generator as tensor_name_generator
import shape_signature_inferer as shape_signature_inferer
from shape_signature_inferer import StaticDim
import instruction_util as instruction_util
import op_call_code_gen
from paddle_eager_generator import PaddleEagerGenerator 
from unit_test_case_spec import (
    UnitTestCaseRequirement,
    UnitTestCaseSpec,
    GenerateRandomUnitTestCaseSpec
)

@dataclass
class ChaosMonkeySpec:
    unit_test_case_requirement: UnitTestCaseRequirement


def GenerateUnitTestCaseSpec(
    unit_test_case_requirement: UnitTestCaseRequirement
) -> UnitTestCaseSpec:
    return GenerateRandomUnitTestCaseSpec(
        requirement=unit_test_case_requirement
    )

def CodeGen(
    unit_test_case_spec: UnitTestCaseSpec,
    op_call_code_gen_requirement: op_call_code_gen.OpCallCodeGenRequirement
) -> Script:
    generator = PaddleEagerGenerator(op_call_code_gen_requirement)
    return generator.Generate(unit_test_case_spec)

def Generate(
    chaos_monkey_spec: ChaosMonkeySpec
) -> Script:
    requirement = chaos_monkey_spec.unit_test_case_requirement
    unit_test_case_spec = GenerateUnitTestCaseSpec(requirement)
    return CodeGen(
        unit_test_case_spec=unit_test_case_spec,
        op_call_code_gen_requirement=requirement.op_call_code_gen_requirement
    )

if __name__ == '__main__':
    unit_test_case_requirement=UnitTestCaseRequirement(
        dag_gen_requirement=dag_generator.DAGGenRequirement(
            min_num_sources=0,
            max_num_sources=0,
            pick_probability=dag_generator.DAGGenTypePickProbability()
        ),
        dims_eq1_gen_requirement=dims_eq1_generator.DimsEq1GenRequirement(
            dims_eq1_probability=[0.1, 0.15, 0.2]
        ),
        op_name_gen_requirement=op_name_generator.OpNameGenRequirement(),
        tensor_name_gen_requirement=tensor_name_generator.TensorNameGenRequirement(),
        dim_size_requirement=shape_signature_inferer.DimSizeRequirement(
            dim_size=[StaticDim(128), StaticDim(64), StaticDim(32)]
        ),
        op_call_code_gen_requirement=op_call_code_gen.OpCallCodeGenRequirement(
            module_name="paddle"
        )
    )
    script = Generate(
        ChaosMonkeySpec(
            unit_test_case_requirement=unit_test_case_requirement
        )
    )
    print(script.file_content)