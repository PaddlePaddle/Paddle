from dataclasses import dataclass

class SignatureConstructor:

    def Nope(self):
        return self.Make()

    def AddSinkTensor(self):
        return self.Make()

    def AddUnaryOp(self, output):
        return self.Make(output)

    def AddBinaryOp(self, output):
        return self.Make(output)

    def AddBinaryClone(self, lhs_output, rhs_output):
        return self.Make(lhs_output, rhs_output)

    def AddSourceOp(self, output):
        return self.Make(output)

    def Make(self, *args):
        raise NotImplementedError("no overrided self.Make() method")

class NaiveSignatureConstructor(SignatureConstructor):
    def __init__(self, dag_gen_instruction, constructor):
        self.dag_gen_instruction=dag_gen_instruction
        self.constructor = constructor

    def Make(self, *args):
        return self.constructor(*args)


@dataclass
class NoneSignature:
    pass


@dataclass
class NoInputNoneSignature(NoneSignature):
    pass

@dataclass
class UnaryInputNoneSignature(NoneSignature):
    input: None = field(
        default_factory=lambda: None,
        metadata=dict(input_idx=0)
    )


@dataclass
class BinaryInputNoneSignature(NoneSignature):
    lhs_input: None = field(
        default_factory=lambda: None,
        metadata=dict(input_idx=0)
    )
    rhs_input: None = field(
        default_factory=lambda: None,
        metadata=dict(input_idx=1)
    )

class NoneSignatureConstructor:

    def __init__(self, dag_gen_instruction: "DAGGenInstruction"):
        self.dag_gen_instruction = dag_gen_instruction

    def Nope(self):
        return NoInputNoneSignature()

    def AddSinkTensor(self):
        return UnaryInputNoneSignature()

    def AddUnaryOp(self, output):
        return UnaryInputNoneSignature()

    def AddBinaryOp(self, output):
        return BinaryInputNoneSignature()

    def AddBinaryClone(self, lhs_output, rhs_output):
        return UnaryInputNoneSignature()

    def AddSourceOp(self, output):
        return NoInputNoneSignature()
