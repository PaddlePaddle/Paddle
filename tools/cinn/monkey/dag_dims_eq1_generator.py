from dataclasses import dataclass
import .dag_generator as dag_generator
import .dims_eq1_generator as dims_eq1_generator
from .pick_weight import PickWeight
from typing import List
from collections import namedtuple

DAGDimsEq1GenInstruction = namedtuple("DAGDimsEq1GenInstruction", ["dag", "dims_eq1"])

class DimsIsEqOneInferContext:
    def __init__(self):
        self.current_source_tensor_dim_eq1 = []
        self.current_num_source_tensors = 0

    def InferInputsDimsEqOne(
        self,
        dag_dims_eq1_gen_instruction: DAGDimsEq1GenInstruction
    ):
        dag_gen_class = type(dag_dims_eq1_gen_instruction.dag)
        cls = kDAGGenClassToDAGDimsEq1GenClassMap[dag_gen_class]
        cls.InferInputsDimsEqOne(dag_dims_eq1_gen_instruction, self)
        self._InferAndCheckCurrentNumSourceTensors(dag_gen_class)

    def _InferAndCheckCurrentNumSourceTensors(self, dag_gen_class):
        self.current_num_source_tensors += dag_gen_class.GetDeltaNumSourceTensors()
        assert (
            len(self.current_source_tensor_dim_eq1)
            == self.current_num_source_tensors
        )


class Nope:

    @classmethod
    def GetPatchedDAGDimsEq1GenInstruction(
        cls,
        dag_dims_eq1_gen_instruction: DAGDimsEq1GenInstruction,
        infer_ctx: DimsIsEqOneInferContext
    ):
        yield dag_dims_eq1_gen_instruction

    @classmethod
    def InferInputsDimsEqOne(
        cls,
        dag_dims_eq1_gen_instruction: DAGDimsEq1GenInstruction,
        infer_ctx: DimsIsEqOneInferContext
    ):
        pass


class AddSinkTensor:

    @classmethod
    def GetPatchedDAGDimsEq1GenInstruction(
        cls,
        dag_dims_eq1_gen_instruction: DAGDimsEq1GenInstruction,
        infer_ctx: DimsIsEqOneInferContext
    ):
        yield dag_dims_eq1_gen_instruction

    @classmethod
    def InferInputsDimsEqOne(
        cls,
        dag_dims_eq1_gen_instruction: DAGDimsEq1GenInstruction,
        infer_ctx: DimsIsEqOneInferContext
    ):
        infer_ctx.current_source_tensor_dim_eq1.append(
            dag_dims_eq1_gen_instruction.dims_eq1.source_tensor_dim_eq1
        )


class AddUnaryOp:

    @classmethod
    def GetPatchedDAGDimsEq1GenInstruction(
        cls,
        dag_dims_eq1_gen_instruction: DAGDimsEq1GenInstruction,
        infer_ctx: DimsIsEqOneInferContext
    ):
        idx = dag_dims_eq1_gen_instruction.dag.source_tensor_index
        output_dims_eq1 = infer_ctx.current_source_tensor_dim_eq1[idx]
        input_dims_eq1 = dag_dims_eq1_gen_instruction.dims_eq1.source_tensor_dim_eq1
        middle_dims_eq1 = [
            x and y
            for x, y in zip(input_dims_eq1, output_dims_eq1)
        ]
        if middle_dims_eq1 != output_dims_eq1:
            yield DAGDimsEq1GenInstruction(
                dag=dag_generator.AddUnaryOp(
                    source_tensor_index=idx,
                    dag_tag=dag_dims_eq1_gen_instruction.dag.dag_tag
                ),
                dims_eq1=dims_eq1_generator.AddUnaryOp(
                    source_tensor_dim_eq1=middle_dims_eq1
                )
            )
        if input_dims_eq1 != middle_dims_eq1:
            yield DAGDimsEq1GenInstruction(
                dag=dag_generator.AddUnaryOp(
                    source_tensor_index=idx,
                    dag_tag=dag_dims_eq1_gen_instruction.dag.dag_tag
                ),
                dims_eq1=dims_eq1_generator.AddUnaryOp(
                    source_tensor_dim_eq1=input_dims_eq1
                )
            )
        yield dag_dims_eq1_gen_instruction

    @classmethod
    def InferInputsDimsEqOne(
        cls,
        dag_dims_eq1_gen_instruction: DAGDimsEq1GenInstruction,
        infer_ctx: DimsIsEqOneInferContext
    ):
        idx = dag_dims_eq1_gen_instruction.dag.source_tensor_index
        input_dims_eq1 = dag_dims_eq1_gen_instruction.dims_eq1.source_tensor_dim_eq1
        infer_ctx.current_source_tensor_dim_eq1[idx] = input_dims_eq1
       

class AddBinaryOp:

    @classmethod
    def GetPatchedDAGDimsEq1GenInstruction(
        cls,
        dag_dims_eq1_gen_instruction: DAGDimsEq1GenInstruction,
        infer_ctx: DimsIsEqOneInferContext
    ):
        lhs_input_dims_eq1 = (
            dag_dims_eq1_gen_instruction.dims_eq1.lhs_source_tensor_dim_eq1
        )
        rhs_input_dims_eq1 = (
            dag_dims_eq1_gen_instruction.dims_eq1.rhs_source_tensor_dim_eq1
        )
        output_idx = dag_dims_eq1_gen_instruction.dag.source_tensor_index
        output_dims_eq1 = infer_ctx.current_source_tensor_dim_eq1[output_idx]
        broadcast_dims_eq1 = [
            x or y
            for x, y in zip(lhs_input_dims_eq1, rhs_input_dims_eq1)
        ]
        middle_dims_eq1 = [
            x and y
            for x, y in zip(broadcast_dims_eq1, output_dims_eq1)
        ]
        if middle_dims_eq1 != output_dims_eq1:
            yield DAGDimsEq1GenInstruction(
                dag=dag_generator.AddUnaryOp(
                    source_tensor_index=output_idx,
                    dag_tag=dag_dims_eq1_gen_instruction.dag.dag_tag
                ),
                dims_eq1=dims_eq1_generator.AddUnaryOp(
                    source_tensor_dim_eq1=middle_dims_eq1
                )
            )
        if broadcast_dims_eq1 != middle_dims_eq1:
            yield DAGDimsEq1GenInstruction(
                dag=dag_generator.AddUnaryOp(
                    source_tensor_index=output_idx,
                    dag_tag=dag_dims_eq1_gen_instruction.dag.dag_tag
                ),
                dims_eq1=dims_eq1_generator.AddUnaryOp(
                    source_tensor_dim_eq1=broadcast_dims_eq1
                )
            )
        yield dag_dims_eq1_gen_instruction


    @classmethod
    def InferInputsDimsEqOne(
        cls,
        dag_dims_eq1_gen_instruction: DAGDimsEq1GenInstruction,
        infer_ctx: DimsIsEqOneInferContext
    ):
        lhs_input_idx = dag_dims_eq1_gen_instruction.dag.source_tensor_index
        infer_ctx.current_source_tensor_dim_eq1[lhs_input_idx] = (
            dag_dims_eq1_gen_instruction.dims_eq1.lhs_source_tensor_dim_eq1
        )
        infer_ctx.current_source_tensor_dim_eq1.append(
            dag_dims_eq1_gen_instruction.dims_eq1.rhs_source_tensor_dim_eq1
        )
    
class InsertBinaryOp:

    @classmethod
    def GetPatchedDAGDimsEq1GenInstruction(
        cls,
        dag_dims_eq1_gen_instruction: DAGDimsEq1GenInstruction,
        infer_ctx: DimsIsEqOneInferContext
    ):
        output_idx = dag_dims_eq1_gen_instruction.dag.source_tensor_index
        output_dims_eq1 = infer_ctx.current_source_tensor_dim_eq1[output_idx]
        rhs_input_dims_eq1 = (
            dag_dims_eq1_gen_instruction.dims_eq1.rhs_source_tensor_dim_eq1
        )
        middle_dims_eq1 = [
            x and y
            for x, y in zip(output_dims_eq1, rhs_input_dims_eq1)
        ]
        yield dag_dims_eq1_gen_instruction
        new_output_idx = len(infer_ctx.current_source_tensor_dim_eq1)
        if middle_dims_eq1 != output_dims_eq1:
            yield DAGDimsEq1GenInstruction(
                dag=dag_generator.AddUnaryOp(
                    source_tensor_index=new_output_idx,
                    dag_tag=dag_dims_eq1_gen_instruction.dag.dag_tag
                ),
                dims_eq1=dims_eq1_generator.AddUnaryOp(
                    source_tensor_dim_eq1=middle_dims_eq1
                )
            )
        if rhs_input_dims_eq1 != middle_dims_eq1:
            yield DAGDimsEq1GenInstruction(
                dag=dag_generator.AddUnaryOp(
                    source_tensor_index=new_output_idx,
                    dag_tag=dag_dims_eq1_gen_instruction.dag.dag_tag
                ),
                dims_eq1=dims_eq1_generator.AddUnaryOp(
                    source_tensor_dim_eq1=rhs_input_dims_eq1
                )
            )
 
    @classmethod
    def InferInputsDimsEqOne(
        cls,
        dag_dims_eq1_gen_instruction: DAGDimsEq1GenInstruction,
        infer_ctx: DimsIsEqOneInferContext
    ):
        idx = dag_dims_eq1_gen_instruction.dag.source_tensor_index
        input_dims_eq1 = dag_dims_eq1_gen_instruction.dims_eq1.source_tensor_dim_eq1
        infer_ctx.current_source_tensor_dim_eq1[idx] = input_dims_eq1
       

    @classmethod
    def InferInputsDimsEqOne(
        cls,
        dag_dims_eq1_gen_instruction: DAGDimsEq1GenInstruction,
        infer_ctx: DimsIsEqOneInferContext
    ):
        rhs_input_dims_eq1 = (
            dag_dims_eq1_gen_instruction.dims_eq1.rhs_source_tensor_dim_eq1
        )
        infer_ctx.current_source_tensor_dim_eq1.append(rhs_input_dims_eq1)
    

class AddBinaryClone:

    @classmethod
    def GetPatchedDAGDimsEq1GenInstruction(
        cls,
        dag_dims_eq1_gen_instruction: DAGDimsEq1GenInstruction,
        infer_ctx: DimsIsEqOneInferContext
    ):
        lhs_input_dims_eq1 = infer_ctx.current_source_tensor_dim_eq1[
            dag_dims_eq1_gen_instruction.dag.lhs_source_tensor_index
        ]
        rhs_input_dix = dag_dims_eq1_gen_instruction.dag.rhs_source_tensor_index
        rhs_input_dims_eq1 = infer_ctx.current_source_tensor_dim_eq1[
            rhs_input_dix
        ]
        middle_dims_eq1 = [
            x and y
            for x, y in zip(lhs_input_dims_eq1, rhs_input_dims_eq1)
        ]
        if middle_dims_eq1 != rhs_input_dims_eq1:
            yield DAGDimsEq1GenInstruction(
                dag=dag_generator.AddUnaryOp(
                    source_tensor_index=rhs_input_dix,
                    dag_tag=dag_dims_eq1_gen_instruction.dag.dag_tag
                ),
                dims_eq1=dims_eq1_generator.AddUnaryOp(
                    source_tensor_dim_eq1=middle_dims_eq1
                )
            )
        if lhs_input_dims_eq1 != middle_dims_eq1:
            yield DAGDimsEq1GenInstruction(
                dag=dag_generator.AddUnaryOp(
                    source_tensor_index=rhs_input_dix,
                    dag_tag=dag_dims_eq1_gen_instruction.dag.dag_tag
                ),
                dims_eq1=dims_eq1_generator.AddUnaryOp(
                    source_tensor_dim_eq1=lhs_input_dims_eq1
                )
            )
        yield dag_dims_eq1_gen_instruction

    @classmethod
    def InferInputsDimsEqOne(
        cls,
        dag_dims_eq1_gen_instruction: DAGDimsEq1GenInstruction,
        infer_ctx: DimsIsEqOneInferContext
    ):
        rhs_output_idx = dag_dims_eq1_gen_instruction.dag.rhs_source_tensor_index
        infer_ctx.current_source_tensor_dim_eq1.pop(rhs_output_idx)


class AddSourceOp:

    @classmethod
    def GetPatchedDAGDimsEq1GenInstruction(
        cls,
        dag_dims_eq1_gen_instruction: DAGDimsEq1GenInstruction,
        infer_ctx: DimsIsEqOneInferContext
    ):
        yield dag_dims_eq1_gen_instruction

    @classmethod
    def InferInputsDimsEqOne(
        cls,
        dag_dims_eq1_gen_instruction: DAGDimsEq1GenInstruction,
        infer_ctx: DimsIsEqOneInferContext
    ):
        output_idx = dag_dims_eq1_gen_instruction.dag.source_tensor_index
        infer_ctx.current_source_tensor_dim_eq1.pop(output_idx)


kDAGGenClassToDAGDimsEq1GenClassMap = {
    dag_generator.Nope: Nope,
    dag_generator.AddSinkTensor: AddSinkTensor,
    dag_generator.AddUnaryOp: AddUnaryOp,
    dag_generator.AddBinaryOp: AddBinaryOp,
    dag_generator.InsertBinaryOp: InsertBinaryOp,
    dag_generator.AddBinaryClone: AddBinaryClone,
    dag_generator.AddSourceOp: AddSourceOp,
}


class DAGDimsEq1Generator:
    def __init__(self):
        pass

    def Generate(
        self,
        dag_gen_instructions: List["DAGGenInstruction"],
        dims_eq1_gen_instructions: List["DimsEq1GenInstruction"]
    ) -> List[dag_generator.DAGDimsEq1GenInstruction]:
        infer_ctx = DimsIsEqOneInferContext()
        def CreateDAGDimsEq1GenInstructions(dag_dims_eq1_gen_instruction):
            dag_gen_class = type(dag_dims_eq1_gen_instruction.dag)
            cls = kDAGGenClassToDAGDimsEq1GenClassMap[dag_gen_class]
            assert (
                len(infer_ctx.current_source_tensor_dim_eq1)
                == infer_ctx.current_num_source_tensors
            )
            yield from cls.GetPatchedDAGDimsEq1GenInstruction(
                dag_dims_eq1_gen_instruction,
                infer_ctx
            )
            cls.InferInputsDimsEqOne(dag_dims_eq1_gen_instruction, infer_ctx)

        return [
            dag_dims_eq1_gen_instruction
            for dag_dims_eq1_gen_instruction in CreateDAGDimsEq1GenInstructions(x)
            for x in _ZipDAGDimsInstr(dag_gen_instructions, dims_eq1_gen_instructions)
        ]

def _ZipDAGDimsInstr(dag_gen_instructions, dims_eq1_gen_instructions):
    return [
        DAGDimsEq1GenInstruction(*instruction_tuple)
        for instruction_tuple in zip(dag_gen_instructions, dims_eq1_gen_instructions)
    ]
