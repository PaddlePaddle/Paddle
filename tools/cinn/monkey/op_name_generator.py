from dataclasses import dataclass, field
import .dag_generator as dag_generator
from .defensive_list import DList
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
    source_op_names: List[str] = field(
        default_factory=lambda: ["zeros", "ones"]
    )

@dataclass
class Nope:
    @classmethod
    def MakeRandomInstance(
        cls,
        requirement: OpNameGenRequirement,
        dag_gen_instruction: "DAGGenInstruction",
        dims_eq1_signature: "DimsEq1Signature"
    ) -> "OpNameGenInstruction":
        return Nope()


@dataclass
class AddSinkTensor:

    @classmethod
    def MakeRandomInstance(
        cls,
        requirement: OpNameGenRequirement,
        dag_gen_instruction: "DAGGenInstruction",
        dims_eq1_signature: "DimsEq1Signature"
    ) -> "OpNameGenInstruction":
        return AddSinkTensor()

@dataclass
class AddUnaryOp:
    op_name: str

    @classmethod
    def MakeRandomInstance(
        cls,
        requirement: OpNameGenRequirement,
        dag_gen_instruction: "DAGGenInstruction",
        dims_eq1_signature: "DimsEq1Signature"
    ) -> "OpNameGenInstruction":
        input_dims_eq1 = dims_eq1_signature.input_dims_eq1
        output_idx = dag_gen_instruction.source_tensor_index
        output_dims_eq1 = dims_eq1_signature.output_dims_eq1
        if _IsReduceOp(input_dims_eq1, output_dims_eq1):
            return AddUnaryOp(
                op_name=_GetRandomReduceOpName(requirement)
            )
        if _IsBroadcastOp(input_dims_eq1, output_dims_eq1):
            return AddUnaryOp(
                op_name=_GetRandomBroadcastOpName(requirement)
            )
        assert input_dims_eq1 == output_dims_eq1
        return AddUnaryOp(
            op_name=_GetRandomUnaryOpName(requirement)
        )


@dataclass
class AddBinaryOp:
    op_name: str

    @classmethod
    def MakeRandomInstance(
        cls,
        requirement: OpNameGenRequirement,
        dag_gen_instruction: "DAGGenInstruction",
        dims_eq1_signature: "DimsEq1Signature"
    ) -> "OpNameGenInstruction":
        lhs_input_dims_eq1 = dims_eq1_signature.lhs_input_dims_eq1
        rhs_input_dims_eq1 = dims_eq1_signature.rhs_input_dims_eq1
        output_dims_eq1 = dims_eq1_signature.output_dims_eq1
        assert _IsLhsGreaterThanRhs(output_dims_eq1, lhs_input_dims_eq1)
        assert _IsLhsGreaterThanRhs(output_dims_eq1, rhs_input_dims_eq1)
        return AddBinaryOp(
            op_name=_GetRandomBinaryOpName(requirement)
        )


@dataclass
class InsertBinaryOp:
    op_name: str

    @classmethod
    def MakeRandomInstance(
        cls,
        requirement: OpNameGenRequirement,
        dag_gen_instruction: "DAGGenInstruction",
        dims_eq1_signature: "DimsEq1Signature"
    ) -> "OpNameGenInstruction":
        rhs_input_dims_eq1 = dims_eq1_signature.rhs_input_dims_eq1
        output_dims_eq1 = dims_eq1_signature.output_dims_eq1
        assert _IsLhsGreaterThanRhs(output_dims_eq1, rhs_input_dims_eq1)
        return InsertBinaryOp(
            op_name=_GetRandomBinaryOpName(requirement)
        )


@dataclass
class AddBinaryClone:

    @classmethod
    def MakeRandomInstance(
        cls,
        requirement: OpNameGenRequirement,
        dag_gen_instruction: "DAGGenInstruction",
        dims_eq1_signature: "DimsEq1Signature"
    ) -> "OpNameGenInstruction":
        lhs_output_dims_eq1 = dims_eq1_signature.lhs_output_dims_eq1
        rhs_output_dims_eq1 = dims_eq1_signature.rhs_output_dims_eq1
        assert lhs_output_dims_eq1 == rhs_output_dims_eq1
        return AddBinaryClone()


@dataclass
class AddSourceOp:
    op_name: str

    @classmethod
    def MakeRandomInstance(
        cls,
        requirement: OpNameGenRequirement,
        dag_gen_instruction: "DAGGenInstruction",
        dims_eq1_signature: "DimsEq1Signature"
    ) -> "OpNameGenInstruction":
        return AddSourceOp(op_name=_GetRandomSourceOpName(requirement))


# OpNameGenInstruction = ( Nope
#                        | AddSinkTensor
#                        | AddUnaryOp
#                        | AddBinaryOp
#                        | InsertBinaryOp
#                        | AddBinaryClone
#                        | AddSourceOp
#                        )

kDAGGenClassToOpNameGenClassMap = {
    dag_generator.Nope: Nope,
    dag_generator.AddSinkTensor: AddSinkTensor,
    dag_generator.AddUnaryOp: AddUnaryOp,
    dag_generator.AddBinaryOp: AddBinaryOp,
    dag_generator.InsertBinaryOp: InsertBinaryOp,
    dag_generator.AddBinaryClone: AddBinaryClone,
    dag_generator.AddSourceOp: AddSourceOp,
}

def _IsReduceOp(input_dims_eq1: List[bool], output_dims_eq1:  List[bool]):
    return _IsLhsGreaterThanRhs(input_dims_eq1, output_dims_eq1)

def _IsBroadcastOp(input_dims_eq1: List[bool], output_dims_eq1:  List[bool]):
    return _IsLhsGreaterThanRhs(output_dims_eq1, input_dims_eq1)

def _IsLhsGreaterThanRhs(lhs: List[bool], rhs:  List[bool]):
    assert len(lhs) == len(rhs)
    for i in range(len(lhs)):
        if not (lhs[i] > rhs[i]):
            return False
    return True

def _GetRandomReduceOpName(requirement: DimEq1GenRequirement):
    return _GetRandomOpName(requirement.reduce_op_names)

def _GetRandomBroadcastOpName(requirement: DimEq1GenRequirement):
    return _GetRandomOpName(requirement.broadcast_op_names)

def _GetRandomUnaryOpName(requirement: DimEq1GenRequirement):
    return _GetRandomOpName(requirement.unary_elementwise_op_names)

def _GetRandomBinaryOpName(requirement: DimEq1GenRequirement):
    return _GetRandomOpName(requirement.binary_elementwise_op_names)

def _GetRandomSourceOpName(requirement: DimEq1GenRequirement):
    return _GetRandomOpName(requirement.source_op_names)

def _GetRandomOpName(op_names: List[str]):
    kRangeMax = len(op_names) - 1
    assert 0 <= kRangeMax
    random_int = random.randomint(0, kRangeMax)
    return op_names[random_int]

class OpNameGenerator:
    def __init__(self, requirement: DimEq1GenRequirement):
        self.requirement = requirement
    
    # Instructions generating sink nodes of DAG are on put the front of list.
    def Generate(
        self,
        guarded_dims_eq1_sigs: DList["DAGGenInstruction", "DimsEq1Signature"]
    ) -> List["OpNameGenInstruction"]:
        def CreateOpNameGenInstruction(dag_instr, dims_eq1_sig):
            cls = kDAGGenClassToOpNameGenClassMap[type(dag_instr)]
            return cls.MakeRandomInstance(
                self.requirement,
                dag_instr,
                dims_eq1_sig
            )
        return [
            CreateOpNameGenInstruction(*x)
            for x in guarded_dims_eq1_sigs.Unguard(
                lambda key: key.GetHashValue()
            )
        ]
