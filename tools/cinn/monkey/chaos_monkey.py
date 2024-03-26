from dataclasses import dataclass
from collections import namedtuple
from typing import Generator, Union
from .defensive_list import DList
from .dag_generator import PickWeight
from .script import Script
import .dag_generator as dag_generator
import .dim_eq1_generator as dim_eq1_generator
import .dims_eq1_generator as dims_eq1_generator
import .op_name_generator as op_name_generator
import .tensor_name_generator as tensor_name_generator
import .shape_signature_inferer as shape_signature_inferer
from .shape_signature_inferer import StaticDim
import .instruction_util as instruction_util
from .paddle_eager_generator import PaddleEagerGenerator 
from unit_test_case_spec import (
    UnitTestCaseRequirement,
    UnitTestCaseSpec,
    GenerateRandomUnitTestCaseSpec
)

@dataclass
class ChaosMonkeySpec:
    unit_test_case_requirement: UnitTestCaseRequirement


def GenerateUnitTestCaseSpec(
    chaos_monkey_spec: ChaosMonkeySpec
) -> UnitTestCaseSpec:
    return GenerateRandomUnitTestCaseSpec(
        requirement=chaos_monkey_spec.unit_test_case_requirement
    )

def CodeGen(
    unit_test_case_spec: UnitTestCaseSpec
) -> Script:
    generator = PaddleEagerGenerator(unit_test_case_spec.code_gen_requirement)
    return generator.Generate(unit_test_case_spec)

def Generate(
    chaos_monkey_spec: ChaosMonkeySpec
) -> Script:
    unit_test_case_spec = GenerateUnitTestCaseSpec(search_requirement)
    return CodeGen(unit_test_case_spec)

if __name__ == '__main__':
    unit_test_case_spec=UnitTestCaseSpec(
        dag_gen_requirement=dag_generator.DAGGenRequirement(
            pick_probability=dag_generator.DAGGenTypePickProbability(
                nope=PickWeight(0),
                add_sink_tensor=PickWeight(1),
                add_unary_op=PickWeight(1),
                add_binary_op=PickWeight(1),
                insert_binary_op=PickWeight(1),
                add_binary_clone=PickWeight(1),
                add_source_op=PickWeight(1)
            )
        ),
        dims_eq1_gen_requirement=dims_eq1_generator.DimsEq1GenRequirement(
            dims_eq1_probability=[0.1, 0.2, 0.3]
        ),
        op_name_gen_requirement=op_name_generator.OpNameGenRequirement(),
        tensor_name_gen_requirement=tensor_name_generator.TensorNameGenRequirement(),
        dim_size_requirement=shape_signature_inferer.DimSizeRequirement(
            dim_size=[StaticDim(128), StaticDim(64), StaticDim(32)]
        ),
        code_gen_requirement=op_call_code_gen.CodeGenRequirement(
            module_name="paddle"
        )
    )
    script = Generate(
        ChaosMonkeySpec(
            unit_test_case_spec=unit_test_case_spec
        )
    )
    print(script.file_content)