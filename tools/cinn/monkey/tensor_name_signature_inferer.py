from dataclasses import dataclass, field
import .dag_generator as dag_generator
from .defensive_list import DList
from .pick_weight import PickWeight
from typing import List, Set
import random
from .signature_constructor import NaiveSignatureConstructor
from .signature_inferer import SignatureInferer, InputIdx

@dataclass
class InferContext:
    dag_gen_instruction: "DAGGenInstruction"
    tensor_name_gen_instruction: "TensorNameGenInstruction"


@dataclass
class TensorNameSignature:
    @classmethod
    def GetDAGGenClassToDerivedClassMap(cls):
        return kDAGGenClassToDerivedClass


@dataclass
class Nope(TensorNameSignature):
    
    @classmethod
    def Make(
        cls,
        ctx: InferContext
    ):
        return Nope()


@dataclass
class AddSinkTensor(TensorNameSignature):
    sink_tensor_name: str = InputIdx(0)

    @classmethod
    def Make(
        cls,
        ctx: InferContext
    ):
        sink_tensor_name = ctx.tensor_name_gen_instruction.sink_tensor_name
        return AddSinkTensor(
            sink_tensor_name=sink_tensor_name
        )

@dataclass
class AddUnaryOp(TensorNameSignature):
    input_tensor_name: str = InputIdx(0)
    output_tensor_name: str

    @classmethod
    def Make(
        cls,
        ctx: InferContext,
        output_tensor_name: str
    ):
        input_tensor_name = ctx.tensor_name_gen_instruction.input_tensor_name
        return AddUnaryOp(
            input_tensor_name=input_tensor_name,
            output_tensor_name=output_tensor_name
        )


@dataclass
class AddBinaryOp(TensorNameSignature):
    lhs_input_tensor_name: str = InputIdx(0)
    rhs_input_tensor_name: str = InputIdx(1)
    output_tensor_name: str

    @classmethod
    def Make(
        cls,
        ctx: InferContext,
        output_tensor_name: str
    ):
        lhs_input_tensor_name = ctx.tensor_name_gen_instruction.lhs_input_tensor_name
        rhs_input_tensor_name = ctx.tensor_name_gen_instruction.rhs_input_tensor_name
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
    def Make(
        cls,
        ctx: InferContext,
        lhs_output_tensor_name: str,
        lhs_output_tensor_name: str
    ):
        return AddBinaryClone(
            lhs_output_tensor_name=lhs_output_tensor_name,
            rhs_output_tensor_name=rhs_output_tensor_name
        )


@dataclass
class AddSourceOp(TensorNameSignature):
    output_tensor_name: str

    @classmethod
    def Make(
        cls,
        ctx: InferContext,
        output_tensor_name: str
    ):
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
        signature_inferer = SignatureInferer()
        def CreateTensorNameSignature(pair):
            dag_gen_instruction, tensor_name_gen_instruction = *pair
            ctx = InferContext(
                dag_gen_instruction=dag_gen_instruction,
                tensor_name_gen_instruction=tensor_name_gen_instruction
            )
            cls = kDAGGenClassToDerivedClass[type(dag_gen_instruction)]
            return signature_inferer.Infer(
                NaiveSignatureConstructor(
                    dag_gen_instruction=dag_gen_instruction,
                    constructor=lambda *args: cls.Make(ctx, *args),
                )
            )
        return [
            CreateTensorNameSignature(x)
            for x in zip(
                dag_gen_instructions,
                tensor_name_gen_instructions
            )
        ]
