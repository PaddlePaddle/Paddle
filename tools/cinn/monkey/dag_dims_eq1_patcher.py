from dataclasses import dataclass
import .dag_generator as dag_generator
import .dims_eq1_generator as dims_eq1_generator
import .dims_eq1_signature_inferer as DimsEq1SignatureInferer
from .pick_weight import PickWeight
from typing import List, Generator
from collections import namedtuple
from .defensive_list import DList


@dataclass
class Nope:

    @classmethod
    def GetPatchedDAGDimsEq1GenInstruction(
        cls,
        dag_gen_instruction: "DAGGenInstruction",
        dims_eq1_signature: "DimsEq1Signature"
    ) -> Generator["DAGGenInstruction", "DimsEq1GenInstruction"]:
        yield (
            dag_generator.Nope(),
            dims_eq1_generator.Nope()
        )


@dataclass
class AddSinkTensor:

    @classmethod
    def GetPatchedDAGDimsEq1GenInstruction(
        cls,
        dag_gen_instruction: "DAGGenInstruction",
        dims_eq1_signature: "DimsEq1Signature"
    ) -> Generator["DAGGenInstruction", "DimsEq1GenInstruction"]:
        yield (
            dag_generator.AddSinkTensor(),
            dims_eq1_generator.AddSinkTensor()
        )


@dataclass
class AddUnaryOp:

    @classmethod
    def GetPatchedDAGDimsEq1GenInstruction(
        cls,
        dag_gen_instruction: "DAGGenInstruction",
        dims_eq1_signature: "DimsEq1Signature"
    ) -> Generator["DAGGenInstruction", "DimsEq1GenInstruction"]:
        input_dims_eq1 = dims_eq1_signature.input_dims_eq1
        output_dims_eq1 = dims_eq1_signature.output_dims_eq1
        middle_dims_eq1 = [
            x and y
            for x, y in zip(input_dims_eq1, output_dims_eq1)
        ]
        if middle_dims_eq1 != output_dims_eq1:
            yield (
                dag_gen_instruction,
                dims_eq1_generator.AddUnaryOp(
                    source_tensor_dim_eq1=middle_dims_eq1
                )
            )
        yield (
            dag_gen_instruction,
            dims_eq1_generator.AddUnaryOp(
                source_tensor_dim_eq1=input_dims_eq1
            )
        )


@dataclass
class AddBinaryOp:

    @classmethod
    def GetPatchedDAGDimsEq1GenInstruction(
        cls,
        dag_gen_instruction: "DAGGenInstruction",
        dims_eq1_signature: "DimsEq1Signature"
    ) -> Generator["DAGGenInstruction", "DimsEq1GenInstruction"]:
        lhs_input_dims_eq1 = dims_eq1_signature.lhs_input_dims_eq1
        rhs_input_dims_eq1 = dims_eq1_signature.rhs_input_dims_eq1
        output_dims_eq1 = dims_eq1_signature.output_dims_eq1
        output_idx = dag_gen_instruction.source_tensor_index
        broadcast_dims_eq1 = [
            x or y
            for x, y in zip(lhs_input_dims_eq1, rhs_input_dims_eq1)
        ]
        if broadcast_dims_eq1 != output_dims_eq1:
            yield (
                dag_generator.AddUnaryOp(
                    source_tensor_index=output_idx,
                    dag_tag=dims_eq1_signature.dag_tag
                ),
                dims_eq1_generator.AddUnaryOp(
                    source_tensor_dim_eq1=broadcast_dims_eq1
                )
            )
        yield (
            dag_gen_instruction,
            dim_eq1_generator.AddBinaryOp(
                lhs_source_tensor_dim_eq1=lhs_input_dims_eq1,
                rhs_source_tensor_dim_eq1=rhs_input_dims_eq1
            )
        )

    
@dataclass
class InsertBinaryOp:

    @classmethod
    def GetPatchedDAGDimsEq1GenInstruction(
        cls,
        dag_gen_instruction: "DAGGenInstruction",
        dims_eq1_signature: "DimsEq1Signature"
    ) -> Generator["DAGGenInstruction", "DimsEq1GenInstruction"]:
        lhs_input_dims_eq1 = dims_eq1_signature.lhs_input_dims_eq1
        rhs_input_dims_eq1 = dims_eq1_signature.rhs_input_dims_eq1
        output_dims_eq1 = dims_eq1_signature.output_dims_eq1
        output_idx = dag_gen_instruction.source_tensor_index
        middle_dims_eq1 = [
            x and y
            for x, y in zip(output_dims_eq1, rhs_input_dims_eq1)
        ]
        yield (
            dag_gen_instruction,
            dims_eq1_gen_instructions.InsertBinaryOp(
                rhs_source_tensor_dim_eq1=middle_dims_eq1
            )
        )
        new_output_idx = dims_eq1_signature.rhs_input_source_tensor_index
        if rhs_input_dims_eq1 != middle_dims_eq1:
            yield (
                dag_generator.AddUnaryOp(
                    source_tensor_index=new_output_idx,
                    dag_tag=dag_gen_instruction.dag_tag
                ),
                dims_eq1_generator.AddUnaryOp(
                    source_tensor_dim_eq1=rhs_input_dims_eq1
                )
            )
 

@dataclass
class AddBinaryClone:

    @classmethod
    def GetPatchedDAGDimsEq1GenInstruction(
        cls,
        dag_gen_instruction: "DAGGenInstruction",
        dims_eq1_signature: "DimsEq1Signature"
    ) -> Generator["DAGGenInstruction", "DimsEq1GenInstruction"]:
        lhs_input_dims_eq1 = dims_eq1_signature.lhs_input_dims_eq1
        rhs_input_dix = dag_gen_instruction.rhs_source_tensor_index
        rhs_input_dims_eq1 = dims_eq1_signature.rhs_input_dims_eq1
        if lhs_input_dims_eq1 != rhs_input_dims_eq1:
            yield (
                dag_generator.AddUnaryOp(
                    source_tensor_index=rhs_input_dix,
                    dag_tag=dag_gen_instruction.dag_tag
                ),
                dims_eq1_generator.AddUnaryOp(
                    source_tensor_dim_eq1=lhs_input_dims_eq1
                )
            )
        yield (
            dag_gen_instruction,
            dims_eq1_generator.AddBinaryClone()
        )


@dataclass
class AddSourceOp:

    @classmethod
    def GetPatchedDAGDimsEq1GenInstruction(
        cls,
        dag_gen_instruction: "DAGGenInstruction",
        dims_eq1_signature: "DimsEq1Signature"
    ) -> Generator["DAGGenInstruction", "DimsEq1GenInstruction"]:
        yield dag_gen_instruction, dims_eq1_generator.AddSourceOp()


kDAGGenClassToDAGDimsEq1GenClassMap = {
    dag_generator.Nope: Nope,
    dag_generator.AddSinkTensor: AddSinkTensor,
    dag_generator.AddUnaryOp: AddUnaryOp,
    dag_generator.AddBinaryOp: AddBinaryOp,
    dag_generator.InsertBinaryOp: InsertBinaryOp,
    dag_generator.AddBinaryClone: AddBinaryClone,
    dag_generator.AddSourceOp: AddSourceOp,
}


class DAGDimsEq1Patcher:
    def __init__(self):
        pass

    def Patch(
        self,
        dag_gen_instruction: List["DAGGenInstruction"],
        dims_eq1_gen_instructions: List["DimsEq1GenInstruction"]
    ) -> Tuple[List["DAGGenInstruction"], List["DimsEq1GenInstruction"]]:
        # first patching
        guarded_dims_eq1_sigs = DimsEq1SignatureInferer(
            dag_gen_instruction, dims_eq1_gen_instructions
        )
        dag_gen_instruction, dims_eq1_gen_instructions = self.PatchOnce(
            guarded_dims_eq1_sigs
        )
        # second patching
        guarded_dims_eq1_sigs = DimsEq1SignatureInferer(
            dag_gen_instruction, dims_eq1_gen_instructions
        )
        dag_gen_instruction, dims_eq1_gen_instructions = self.PatchOnce(
            guarded_dims_eq1_sigs
        )
        # thrid patching
        guarded_dims_eq1_sigs = DimsEq1SignatureInferer(
            dag_gen_instruction, dims_eq1_gen_instructions
        )
        ret_dag_gen_instruction, ret_dims_eq1_gen_instructions = self.PatchOnce(
            guarded_dims_eq1_sigs
        )
        assert len(dag_gen_instruction) == len(ret_dag_gen_instruction)
        assert len(dims_eq1_gen_instructions) == len(ret_dims_eq1_gen_instructions)
        return ret_dag_gen_instruction, ret_dims_eq1_gen_instructions



    def PatchOnce(
        self,
        guarded_dims_eq1_sigs: DList["DAGGenInstruction", "DimsEq1Signature"]
    ) -> Tuple[List["DAGGenInstruction"], List["DimsEq1GenInstruction"]]:
        def CreateDAGDimsEq1GenInstructions(dag_instr, dims_eq1_sig):
            cls = kDAGGenClassToDAGDimsEq1GenClassMap[type(dag_instr.dag)]
            yield from cls.GetPatchedDAGDimsEq1GenInstruction(
                dag_instr, dims_eq1_sig,
            )

        pairs = [
            pair
            for pair in CreateDAGDimsEq1GenInstructions(*x)
            for x in guarded_dims_eq1_sigs.Unguard(
                lambda key: key.GetHashValue()
            )
        ]
        return [x for x,_ in pairs], [y for _,y for pairs]
