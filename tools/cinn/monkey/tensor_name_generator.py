from dataclasses import dataclass, field
import dag_generator as dag_generator
from defensive_list import DList
from pick_weight import PickWeight
from typing import List, Set
from hash_combine import HashCombine
import random

@dataclass
class TensorNameGenRequirement:
    tensor_name_prefix: str = "t"

@dataclass
class TensorSeqCounter:
    counter: int = 0

    def AutoIncrementalCounter(self):
        self.counter += 1
        return self.counter

@dataclass
class SourceTensorNames:
    names: List[str] = field(
        default_factory=list
    )

@dataclass
class TensorNameGenContext:
    requirement: TensorNameGenRequirement
    dag_gen_instruction: "DAGGenInstruction"
    tensor_seq_counter: TensorSeqCounter

    def NewTensorName(self):
        prefix = self.requirement.tensor_name_prefix
        seq_no = self.tensor_seq_counter.AutoIncrementalCounter()
        return "%s%d" % (prefix, seq_no)


@dataclass
class TensorNameGenInstruction:
    @classmethod
    def GetDerivedClassByDAGGenClass(cls, dag_gen_class):
        return kDAGGenClassToDerivedClass[dag_gen_class]


@dataclass
class Nope(TensorNameGenInstruction):

    def __hash__(self):
        return hash(id(Nope))

    @classmethod
    def GenerateTensorNames(
        cls,
        ctx: TensorNameGenContext
    ) -> TensorNameGenInstruction:
        return Nope()


@dataclass
class AddSinkTensor(TensorNameGenInstruction):
    sink_tensor_name: str

    def __hash__(self):
        return hash(id(AddSinkTensor))

    @classmethod
    def GenerateTensorNames(
        cls,
        ctx: TensorNameGenContext
    ) -> TensorNameGenInstruction:
        return AddSinkTensor(
            sink_tensor_name=ctx.NewTensorName()
        )


@dataclass
class AddUnaryOp(TensorNameGenInstruction):
    input_tensor_name: str

    def __hash__(self):
        return hash(self.input_tensor_name)

    @classmethod
    def GenerateTensorNames(
        cls,
        ctx: TensorNameGenContext
    ) -> TensorNameGenInstruction:
        return AddUnaryOp(
            input_tensor_name=ctx.NewTensorName()
        )


@dataclass
class AddBinaryOp(TensorNameGenInstruction):
    lhs_input_tensor_name: str
    rhs_input_tensor_name: str

    def __hash__(self):
        hash_value = 0
        hash_value = HashCombine(hash_value, hash(self.lhs_input_tensor_name))
        hash_value = HashCombine(hash_value, hash(self.rhs_input_tensor_name))
        return hash_value

    @classmethod
    def GenerateTensorNames(
        cls,
        ctx: TensorNameGenContext
    ) -> TensorNameGenInstruction:
        return AddBinaryOp(
            lhs_input_tensor_name=ctx.NewTensorName(),
            rhs_input_tensor_name=ctx.NewTensorName()
        )


@dataclass
class AddBinaryClone(TensorNameGenInstruction):

    def __hash__(self):
        return hash(id(AddBinaryClone))

    @classmethod
    def GenerateTensorNames(
        cls,
        ctx: TensorNameGenContext
    ) -> TensorNameGenInstruction:
        return AddBinaryClone()


@dataclass
class AddSourceOp(TensorNameGenInstruction):

    def __hash__(self):
        return hash(id(AddSourceOp))

    @classmethod
    def GenerateTensorNames(
        cls,
        ctx: TensorNameGenContext
    ) -> TensorNameGenInstruction:
        return AddSourceOp()


kDAGGenClassToDerivedClass = {
    dag_generator.Nope: Nope,
    dag_generator.AddSinkTensor: AddSinkTensor,
    dag_generator.AddUnaryOp: AddUnaryOp,
    dag_generator.AddBinaryOp: AddBinaryOp,
    dag_generator.AddBinaryClone: AddBinaryClone,
    dag_generator.AddSourceOp: AddSourceOp,
}

class TensorNameGenerator:
    def __init__(self, requirement: TensorNameGenRequirement):
        self.requirement = requirement
    
    # Instructions generating sink nodes of DAG are on put the front of list.
    def Generate(
        self,
        dag_gen_instructions: List["DAGGenInstruction"]
    ) -> List["TensorNameGenInstruction"]:
        def CreateTensorNameGenInstruction(dag_gen_instruction):
            cls = kDAGGenClassToDerivedClass[type(dag_gen_instruction)]
            return cls.GenerateTensorNames(
                TensorNameGenContext(
                    requirement=self.requirement,
                    dag_gen_instruction=dag_gen_instruction,
                    tensor_seq_counter=_global_tensor_seq_counter
                )
            )
        return reversed([
            CreateTensorNameGenInstruction(x)
            for x in reversed(dag_gen_instructions)
        ])

_global_tensor_seq_counter = TensorSeqCounter()