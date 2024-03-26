from dataclasses import dataclass, field
import .dag_generator as dag_generator
from .defensive_list import DList
from .pick_weight import PickWeight
from typing import List, Set
import random


@dataclass
class SourceTensorNames:
    names: List[str] = field(
        default_factory=list
    )

@dataclass
class InferContext:
    dag_gen_instruction: "DAGGenInstruction"
    tensor_name_gen_instruction: "TensorNameGenInstruction"
    source_tensor_names: SourceTensorNames

    def GetSourceTensorName(self, source_tensor_idx):
        assert source_tensor_idx >= 0
        assert source_tensor_idx < len(source_tensor_names.names)
        return source_tensor_names.names[source_tensor_idx]

    def SetSourceTensorName(self, source_tensor_idx, tensor_name):
        assert source_tensor_idx >= 0
        assert source_tensor_idx < len(source_tensor_names.names)
        source_tensor_names.names[source_tensor_idx] = tensor_name

    def AddSourceTensorName(self, tensor_name):
        source_tensor_names.names.append(tensor_name)

    def EraseSourceTensorName(self, source_tensor_idx):
        assert source_tensor_idx >= 0
        assert source_tensor_idx < len(source_tensor_names.names)
        source_tensor_names.names.pop(source_tensor_idx)

@dataclass
class TensorNameSignature:
    @classmethod
    def GetDAGGenClassToDerivedClassMap(cls):
        return kDAGGenClassToDerivedClass


@dataclass
class Nope(TensorNameSignature):

    @classmethod
    def Infer(
        cls,
        ctx: InferContext
    ) -> TensorNameSignature:
        return Nope()


@dataclass
class AddSinkTensor(TensorNameSignature):
    sink_tensor_name: str

    @classmethod
    def Infer(
        cls,
        ctx: InferContext
    ) -> TensorNameSignature:
        sink_tensor_name = ctx.tensor_name_gen_instruction.sink_tensor_name
        ctx.AddSourceTensorName(sink_tensor_name)
        return AddSinkTensor(
            sink_tensor_name=sink_tensor_name
        )


@dataclass
class AddUnaryOp(TensorNameSignature):
    input_tensor_name: str
    output_tensor_name: str

    @classmethod
    def Infer(
        cls,
        ctx: InferContext
    ) -> TensorNameSignature:
        input_tensor_name = ctx.tensor_name_gen_instruction.input_tensor_name
        source_tensor_index = ctx.dag_gen_instruction.source_tensor_index
        output_tensor_name = ctx.GetSourceTensorName(source_tensor_index)
        ctx.SetSourceTensorName(source_tensor_index, input_tensor_name)
        return AddUnaryOp(
            input_tensor_name=input_tensor_name,
            output_tensor_name=output_tensor_name
        )


@dataclass
class AddBinaryOp(TensorNameSignature):
    lhs_input_tensor_name: str
    rhs_input_tensor_name: str
    output_tensor_name: str

    @classmethod
    def Infer(
        cls,
        ctx: InferContext
    ) -> TensorNameSignature:
        lhs_input_tensor_name = ctx.tensor_name_gen_instruction.lhs_input_tensor_name
        rhs_input_tensor_name = ctx.tensor_name_gen_instruction.rhs_input_tensor_name
        source_tensor_index = ctx.dag_gen_instruction.source_tensor_index
        output_tensor_name=ctx.GetSourceTensorName(source_tensor_index)
        ctx.SetSourceTensorName(source_tensor_index, lhs_input_tensor_name)
        ctx.AddSourceTensorName(rhs_input_tensor_name)
        return AddBinaryOp(
            lhs_input_tensor_name=lhs_input_tensor_name,
            rhs_input_tensor_name=rhs_input_tensor_name,
            output_tensor_name=output_tensor_name
        )


@dataclass
class AddBinaryClone(TensorNameSignature):
    lhs_output_tensor_name: int
    rhs_output_tensor_name: int

    @classmethod
    def Infer(
        cls,
        ctx: InferContext
    ) -> TensorNameSignature:
        lhs_source_tensor_index = ctx.dag_gen_instruction.lhs_source_tensor_index
        rhs_source_tensor_index = ctx.dag_gen_instruction.rhs_source_tensor_index
        lhs_output_tensor_name = ctx.GetSourceTensorName(lhs_source_tensor_index)
        rhs_output_tensor_name = ctx.GetSourceTensorName(rhs_source_tensor_index)
        ctx.EraseSourceTensorName(rhs_source_tensor_index)
        return AddBinaryClone(
            lhs_output_tensor_name=lhs_output_tensor_name,
            rhs_output_tensor_name=rhs_output_tensor_name
        )


@dataclass
class AddSourceOp(TensorNameSignature):
    output_tensor_name: str

    @classmethod
    def Infer(
        cls,
        ctx: InferContext
    ) -> TensorNameSignature:
        source_tensor_index = ctx.dag_gen_instruction.source_tensor_index
        output_tensor_name = ctx.GetSourceTensorName(source_tensor_index)
        ctx.EraseSourceTensorName(source_tensor_index)
        return AddSourceOp(
            output_tensor_name=output_tensor_name
        )


kDAGGenClassToDerivedClass = {
    dag_generator.Nope: Nope,
    dag_generator.AddSinkTensor: AddSinkTensor,
    dag_generator.AddUnaryOp: AddUnaryOp,
    dag_generator.AddBinaryOp: AddBinaryOp,
    dag_generator.AddBinaryClone: AddBinaryClone,
    dag_generator.AddSourceOp: AddSourceOp,
}

class TensorNameGenerator:
    def __init__(self):
        pass
    
    # Instructions generating sink nodes of DAG are on put the front of list.
    def Generate(
        self,
        dag_gen_instructions: List[DAGGenInstruction],
        tensor_name_gen_instructions: List["TensorNameGenInstruction"]
    ) -> List["TensorNameSignature"]:
        source_tensor_names = SourceTensorNames()
        def CreateTensorNameSignature(pair):
            dag_gen_instruction, tensor_name_gen_instruction = *pair
            cls = kDAGGenClassToDerivedClass[type(dag_gen_instruction)]
            return cls.Infer(
                InferContext(
                    dag_gen_instruction=dag_gen_instruction,
                    tensor_name_gen_instruction=tensor_name_gen_instruction
                    source_tensor_names=source_tensor_names,
                )
            )
        return [
            CreateTensorNameSignature(x)
            for x in zip(
                dag_gen_instructions,
                tensor_name_gen_instructions
            )
        ]
