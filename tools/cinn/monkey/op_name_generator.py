from dataclasses import dataclass, field
import .dag_generator as dag_generator
import .dag_dims_generator as dag_dims_generator
from .pick_weight import PickWeight
from typing import List, Set
import random

@dataclass
class OpNameGenRequirement:
    unary_elementwise_op_names: List[str] = field(
        default_factory=lambda: ["add"]
    )
    binary_elementwise_op_names: List[str] = field(
        default_factory=lambda: ["negative"]
    )
    broadcast_op_names: List[str] = field(
        default_factory=lambda: ["expand"]
    )
    reduce_op_names: List[str] = field(
        default_factory=lambda: ["reduce_sum"]
    )
    reshape_op_names: List[str] = field(
        default_factory=lambda: ["reshape"]
    )
    permute_op_names: List[str] = field(
        default_factory=lambda: ["transpose"]
    )

@dataclass
class Nope:
    @classmethod
    def MakeRandomInstance(
        cls,
        requirement: OpNameGenRequirement,
        dag_dims_gen_instruction: dag_dims_generator.DAGDimsGenInstruction,
        infer_ctx: dag_dims_generator.DimsIsEqOneInferContext
    ):
        return Nope()


@dataclass
class AddSinkTensor:

    @classmethod
    def MakeRandomInstance(
        cls,
        requirement: OpNameGenRequirement,
        dag_dims_gen_instruction: dag_dims_generator.DAGDimsGenInstruction,
        infer_ctx: dag_dims_generator.DimsIsEqOneInferContext
    ):
        return AddSinkTensor()

@dataclass
class AddUnaryUpstreamOp:
    op_name: str

    @classmethod
    def MakeRandomInstance(
        cls,
        requirement: OpNameGenRequirement,
        dag_dims_gen_instruction: dag_dims_generator.DAGDimsGenInstruction,
        infer_ctx: dag_dims_generator.DimsIsEqOneInferContext
    ):
        input_dims_eq_one = dag_dims_gen_instruction.dims.source_tensor_dim_eq_one
        output_idx = dag_dims_gen_instruction.dag.source_tensor_index
        output_dims_eq_one = infer_ctx.current_source_tensor_dim_eq_one[output_idx]
        if _IsReduceOp(input_dims_eq_one, output_dims_eq_one):
            return AddUnaryUpstreamOp(
                op_name=_GetRandomReduceOpName(requirement)
            )
        if _IsBroadcastOp(input_dims_eq_one, output_dims_eq_one):
            return AddUnaryUpstreamOp(
                op_name=_GetRandomBroadcastOpName(requirement)
            )
        return AddUnaryUpstreamOp(
            op_name=_GetRandomUnaryOpName(requirement)
        )


@dataclass
class AddBinaryUpstreamOp:
    op_name: str

    @classmethod
    def MakeRandomInstance(
        cls,
        requirement: OpNameGenRequirement,
        dag_dims_gen_instruction: dag_dims_generator.DAGDimsGenInstruction,
        infer_ctx: dag_dims_generator.DimsIsEqOneInferContext
    ):
        return AddBinaryUpstreamOp(
            op_name=_GetRandomBinaryOpName(requirement)
        )


@dataclass
class InsertBinaryUpstreamOp:
    op_name: str

    @classmethod
    def MakeRandomInstance(
        cls,
        requirement: OpNameGenRequirement,
        dag_dims_gen_instruction: dag_dims_generator.DAGDimsGenInstruction,
        infer_ctx: dag_dims_generator.DimsIsEqOneInferContext
    ):
        return InsertBinaryUpstreamOp(
            op_name=_GetRandomBinaryOpName(requirement)
        )


@dataclass
class AddBinaryCloneUpstream:

    @classmethod
    def MakeRandomInstance(
        cls,
        requirement: OpNameGenRequirement,
        dag_dims_gen_instruction: dag_dims_generator.DAGDimsGenInstruction,
        infer_ctx: dag_dims_generator.DimsIsEqOneInferContext
    ):
        return AddBinaryCloneUpstream()


@dataclass
class MarkFinalSourceTensor:

    @classmethod
    def MakeRandomInstance(
        cls,
        requirement: OpNameGenRequirement,
        dag_dims_gen_instruction: dag_dims_generator.DAGDimsGenInstruction,
        infer_ctx: dag_dims_generator.DimsIsEqOneInferContext
    ):
        return MarkFinalSourceTensor()


OpNameGenInstruction = ( Nope
                       | AddSinkTensor
                       | AddUnaryUpstreamOp
                       | AddBinaryUpstreamOp
                       | InsertBinaryUpstreamOp
                       | AddBinaryCloneUpstream
                       | MarkFinalSourceTensor
                       )

kDAGGenClassToOpNameGenClassMap = {
    dag_generator.Nope: Nope,
    dag_generator.AddSinkTensor: AddSinkTensor,
    dag_generator.AddUnaryUpstreamOp: AddUnaryUpstreamOp,
    dag_generator.AddBinaryUpstreamOp: AddBinaryUpstreamOp,
    dag_generator.InsertBinaryUpstreamOp: InsertBinaryUpstreamOp,
    dag_generator.AddBinaryCloneUpstream: AddBinaryCloneUpstream,
    dag_generator.MarkFinalSourceTensor: MarkFinalSourceTensor,
}

def _IsReduceOp(input_dims_eq_one: List[bool], output_dims_eq_one:  List[bool]):
    return _IsLhsGreaterThanRhs(input_dims_eq_one, output_dims_eq_one)

def _IsBroadcastOp(input_dims_eq_one: List[bool], output_dims_eq_one:  List[bool]):
    return _IsLhsGreaterThanRhs(output_dims_eq_one, input_dims_eq_one)

def _IsLhsGreaterThanRhs(lhs: List[bool], rhs:  List[bool]):
    assert len(lhs) == len(rhs)
    for i in range(len(lhs)):
        if not (lhs[i] > rhs[i]):
            return False
    return True

def _GetRandomReduceOpName(requirement: DimGenRequirement):
    op_names = requirement.reduce_op_names
    kRangeMax = len(op_names) - 1
    assert 0 <= kRangeMax
    random_int = random.randomint(0, kRangeMax)
    return op_names[random_int]

def _GetRandomBroadcastOpName(requirement: DimGenRequirement):
    op_names = requirement.broadcast_op_names
    kRangeMax = len(op_names) - 1
    assert 0 <= kRangeMax
    random_int = random.randomint(0, kRangeMax)
    return op_names[random_int]

def _GetRandomUnaryOpName(requirement: DimGenRequirement):
    op_names = requirement.unary_elementwise_op_names
    kRangeMax = len(op_names) - 1
    assert 0 <= kRangeMax
    random_int = random.randomint(0, kRangeMax)
    return op_names[random_int]

def _GetRandomBinaryOpName(requirement: DimGenRequirement):
    op_names = requirement.binary_elementwise_op_names
    kRangeMax = len(op_names) - 1
    assert 0 <= kRangeMax
    random_int = random.randomint(0, kRangeMax)
    return op_names[random_int]

class OpNameGenerator:
    def __init__(self, requirement: DimGenRequirement):
        self.requirement = requirement
    
    # Instructions generating sink nodes of DAG are on put the front of list.
    def Generate(
        self,
        dag_dims_gen_instructions: List[dag_dims_generator.DAGDimsGenInstruction]
    ) -> List[OpNameGenInstruction]:
        infer_ctx = dag_dims_generator.DimsIsEqOneInferContext()
        def CreateOpNameGenInstruction(dag_dims_gen_instruction):
            dag_gen_class = type(dag_dims_gen_instruction.dag)
            op_gen_class = kDAGGenClassToOpNameGenClassMap[dag_gen_class]
            op_name_gen_instruction = op_gen_class.MakeRandomInstance(
                self.requirement,
                dag_dims_gen_instruction,
                infer_ctx
            )
            infer_ctx.InferInputsDimsEqOne(dag_dims_gen_instruction)
            return op_name_gen_instruction
        return [
            CreateOpNameGenInstruction(dag_dims_gen_instruction)
            for dag_dims_gen_instruction in dag_dims_gen_instructions
        ]
