from typing import List
from dataclasses import dataclass, field
from .signature_constructor import SignatureConstructor

def InputIdx(idx):
    return field(metadata=dict(input_idx=idx))

class SignatureInferer:
    def __init__(self):
        self.current_sources = []

    def Infer(self, signature_constructor: SignatureConstructor):
        dag_gen_instruction = signature_constructor.dag_gen_instruction
        cls = globals()[type(dag_gen_instruction).__name__]
        return cls.InferAndConstructSignature(self, signature_constructor)


class Nope:
    @classmethod
    def InferAndConstructSignature(
        cls,
        signature_inferer: SignatureInferer,
        signature_constructor: SignatureConstructor
    ) -> "Signature":
        return signature_inferer.signature_constructor.Nope()

class AddSinkTensor:
    @classmethod
    def InferAndConstructSignature(
        cls,
        signature_inferer: SignatureInferer,
        signature_constructor: SignatureConstructor
    ) -> "Signature":
        signature = signature_constructor.AddSinkTensor()
        input, = _GetInputs(signature)
        signature_inferer.current_sources.append(input)
        return signature

class AddUnaryOp:
    @classmethod
    def InferAndConstructSignature(
        cls,
        signature_inferer: SignatureInferer,
        signature_constructor: SignatureConstructor
    ) -> "Signature":
        dag_gen_instruction = signature_constructor.dag_gen_instruction
        source_tensor_index = dag_gen_instruction.source_tensor_index
        output = signature_inferer.current_sources[source_tensor_index]
        signature = signature_constructor.AddUnaryOp(output)
        input, = _GetInputs(signature)
        signature_inferer.current_sources[source_tensor_index] = input
        return signature


class AddBinaryOp:
    @classmethod
    def InferAndConstructSignature(
        cls,
        signature_inferer: SignatureInferer,
        signature_constructor: SignatureConstructor
    ) -> "Signature":
        dag_gen_instruction = signature_constructor.dag_gen_instruction
        source_tensor_index = dag_gen_instruction.source_tensor_index
        output = signature_inferer.current_sources[source_tensor_index]
        signature = signature_constructor.AddBinaryOp(output)
        lhs_input, rhs_input = _GetInputs(signature)
        signature_inferer.current_sources[source_tensor_index] = lhs_input
        signature_inferer.current_sources.append(rhs_input)
        return signature


class AddBinaryClone:
    @classmethod
    def InferAndConstructSignature(
        cls,
        signature_inferer: SignatureInferer,
        signature_constructor: SignatureConstructor
    ) -> "Signature":
        dag_gen_instruction = signature_constructor.dag_gen_instruction
        lhs_source_tensor_index = dag_gen_instruction.lhs_source_tensor_index
        rhs_source_tensor_index = dag_gen_instruction.rhs_source_tensor_index
        lhs_output = signature_inferer.current_sources[lhs_source_tensor_index]
        rhs_output = signature_inferer.current_sources[rhs_source_tensor_index]
        signature = signature_constructor.AddBinaryClone(lhs_output, rhs_output)
        signature_inferer.current_sources.pop(rhs_source_tensor_index)
        return signature

class AddSourceOp:
    @classmethod
    def InferAndConstructSignature(
        cls,
        signature_inferer: SignatureInferer,
        signature_constructor: SignatureConstructor
    ) -> "Signature":
        dag_gen_instruction = signature_constructor.dag_gen_instruction
        source_tensor_index = dag_gen_instruction.source_tensor_index
        output = signature_inferer.current_sources[source_tensor_index]
        signature = signature_constructor.AddSourceOp(output)
        signature_inferer.current_sources.pop(source_tensor_index)
        return signature

def _GetInputs(signature):
    signature_fields = fields(signature)
    input_fields = [
        f
        for f in signature_fields
        if ('input_idx' in f.metadata) and (type(f.metadata['input_idx']) is int)
    ]
    input_fields = sorted(input_fields, key=lambda f: f.metadata['input_idx'])
    return [
        getattr(signature, f.name)
        for f in input_fields
    ]