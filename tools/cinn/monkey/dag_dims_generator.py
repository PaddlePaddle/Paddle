from dataclasses import dataclass
import .dag_generator as dag_generator
import .dims_generator as dims_generator
from .pick_weight import PickWeight
from typing import List
from collections import namedtuple

DAGDimsGenInstruction = namedtuple("DAGDimsGenInstruction", ["dag", "dims"])

class DimsIsEqOneInferContext:
    def __init__(self):
        self.current_source_tensor_dim_eq_one = []
        self.current_num_source_tensors = 0

    def InferInputsDimsEqOne(
        self,
        dag_dims_gen_instruction: DAGDimsGenInstruction
    ):
        dag_gen_class = type(dag_dims_gen_instruction.dag)
        cls = kDAGGenClassToDAGDimsGenClassMap[dag_gen_class]
        cls.InferInputsDimsEqOne(dag_dims_gen_instruction, self)
        self._InferAndCheckCurrentNumSourceTensors(dag_gen_class)

    def _InferAndCheckCurrentNumSourceTensors(self, dag_gen_class):
        self.current_num_source_tensors += dag_gen_class.GetDeltaNumSourceTensors()
        assert (
            len(self.current_source_tensor_dim_eq_one)
            == self.current_num_source_tensors
        )


class Nope:

    @classmethod
    def GetPatchedDAGDimsGenInstruction(
        cls,
        dag_dims_gen_instruction: DAGDimsGenInstruction,
        infer_ctx: DimsIsEqOneInferContext
    ):
        yield dag_dims_gen_instruction

    @classmethod
    def InferInputsDimsEqOne(
        cls,
        dag_dims_gen_instruction: DAGDimsGenInstruction,
        infer_ctx: DimsIsEqOneInferContext
    ):
        pass


class AddSinkTensor:

    @classmethod
    def GetPatchedDAGDimsGenInstruction(
        cls,
        dag_dims_gen_instruction: DAGDimsGenInstruction,
        infer_ctx: DimsIsEqOneInferContext
    ):
        yield dag_dims_gen_instruction

    @classmethod
    def InferInputsDimsEqOne(
        cls,
        dag_dims_gen_instruction: DAGDimsGenInstruction,
        infer_ctx: DimsIsEqOneInferContext
    ):
        infer_ctx.current_source_tensor_dim_eq_one.append(
            dag_dims_gen_instruction.dims.source_tensor_dim_eq_one
        )


class AddUnaryUpstreamOp:

    @classmethod
    def GetPatchedDAGDimsGenInstruction(
        cls,
        dag_dims_gen_instruction: DAGDimsGenInstruction,
        infer_ctx: DimsIsEqOneInferContext
    ):
        idx = dag_dims_gen_instruction.dag.source_tensor_index
        output_dims_eq_one = infer_ctx.current_source_tensor_dim_eq_one[idx]
        input_dims_eq_one = dag_dims_gen_instruction.dims.source_tensor_dim_eq_one
        middle_dims_eq_one = [
            x and y
            for x, y in zip(input_dims_eq_one, output_dims_eq_one)
        ]
        if middle_dims_eq_one != output_dims_eq_one:
            yield DAGDimsGenInstruction(
                dag=dag_generator.AddUnaryUpstreamOp(
                    source_tensor_index=idx,
                    dag_tag=dag_dims_gen_instruction.dag.dag_tag
                ),
                dims=dims_generator.AddUnaryUpstreamOp(
                    source_tensor_dim_eq_one=middle_dims_eq_one
                )
            )
        if input_dims_eq_one != middle_dims_eq_one:
            yield DAGDimsGenInstruction(
                dag=dag_generator.AddUnaryUpstreamOp(
                    source_tensor_index=idx,
                    dag_tag=dag_dims_gen_instruction.dag.dag_tag
                ),
                dims=dims_generator.AddUnaryUpstreamOp(
                    source_tensor_dim_eq_one=input_dims_eq_one
                )
            )
        yield dag_dims_gen_instruction

    @classmethod
    def InferInputsDimsEqOne(
        cls,
        dag_dims_gen_instruction: DAGDimsGenInstruction,
        infer_ctx: DimsIsEqOneInferContext
    ):
        idx = dag_dims_gen_instruction.dag.source_tensor_index
        input_dims_eq_one = dag_dims_gen_instruction.dims.source_tensor_dim_eq_one
        infer_ctx.current_source_tensor_dim_eq_one[idx] = input_dims_eq_one
       

class AddBinaryUpstreamOp:

    @classmethod
    def GetPatchedDAGDimsGenInstruction(
        cls,
        dag_dims_gen_instruction: DAGDimsGenInstruction,
        infer_ctx: DimsIsEqOneInferContext
    ):
        lhs_input_dims_eq_one = (
            dag_dims_gen_instruction.dims.lhs_source_tensor_dim_eq_one
        )
        rhs_input_dims_eq_one = (
            dag_dims_gen_instruction.dims.rhs_source_tensor_dim_eq_one
        )
        output_idx = dag_dims_gen_instruction.dag.source_tensor_index
        output_dims_eq_one = infer_ctx.current_source_tensor_dim_eq_one[output_idx]
        broadcast_dims_eq_one = [
            x or y
            for x, y in zip(lhs_input_dims_eq_one, rhs_input_dims_eq_one)
        ]
        middle_dims_eq_one = [
            x and y
            for x, y in zip(broadcast_dims_eq_one, output_dims_eq_one)
        ]
        if middle_dims_eq_one != output_dims_eq_one:
            yield DAGDimsGenInstruction(
                dag=dag_generator.AddUnaryUpstreamOp(
                    source_tensor_index=output_idx,
                    dag_tag=dag_dims_gen_instruction.dag.dag_tag
                ),
                dims=dims_generator.AddUnaryUpstreamOp(
                    source_tensor_dim_eq_one=middle_dims_eq_one
                )
            )
        if broadcast_dims_eq_one != middle_dims_eq_one:
            yield DAGDimsGenInstruction(
                dag=dag_generator.AddUnaryUpstreamOp(
                    source_tensor_index=output_idx,
                    dag_tag=dag_dims_gen_instruction.dag.dag_tag
                ),
                dims=dims_generator.AddUnaryUpstreamOp(
                    source_tensor_dim_eq_one=broadcast_dims_eq_one
                )
            )
        yield dag_dims_gen_instruction


    @classmethod
    def InferInputsDimsEqOne(
        cls,
        dag_dims_gen_instruction: DAGDimsGenInstruction,
        infer_ctx: DimsIsEqOneInferContext
    ):
        lhs_input_idx = dag_dims_gen_instruction.dag.source_tensor_index
        infer_ctx.current_source_tensor_dim_eq_one[lhs_input_idx] = (
            dag_dims_gen_instruction.dims.lhs_source_tensor_dim_eq_one
        )
        infer_ctx.current_source_tensor_dim_eq_one.append(
            dag_dims_gen_instruction.dims.rhs_source_tensor_dim_eq_one
        )
    
class InsertBinaryUpstreamOp:

    @classmethod
    def GetPatchedDAGDimsGenInstruction(
        cls,
        dag_dims_gen_instruction: DAGDimsGenInstruction,
        infer_ctx: DimsIsEqOneInferContext
    ):
        output_idx = dag_dims_gen_instruction.dag.source_tensor_index
        output_dims_eq_one = infer_ctx.current_source_tensor_dim_eq_one[output_idx]
        rhs_input_dims_eq_one = (
            dag_dims_gen_instruction.dims.rhs_source_tensor_dim_eq_one
        )
        middle_dims_eq_one = [
            x and y
            for x, y in zip(output_dims_eq_one, rhs_input_dims_eq_one)
        ]
        yield dag_dims_gen_instruction
        new_output_idx = len(infer_ctx.current_source_tensor_dim_eq_one)
        if middle_dims_eq_one != output_dims_eq_one:
            yield DAGDimsGenInstruction(
                dag=dag_generator.AddUnaryUpstreamOp(
                    source_tensor_index=new_output_idx,
                    dag_tag=dag_dims_gen_instruction.dag.dag_tag
                ),
                dims=dims_generator.AddUnaryUpstreamOp(
                    source_tensor_dim_eq_one=middle_dims_eq_one
                )
            )
        if rhs_input_dims_eq_one != middle_dims_eq_one:
            yield DAGDimsGenInstruction(
                dag=dag_generator.AddUnaryUpstreamOp(
                    source_tensor_index=new_output_idx,
                    dag_tag=dag_dims_gen_instruction.dag.dag_tag
                ),
                dims=dims_generator.AddUnaryUpstreamOp(
                    source_tensor_dim_eq_one=rhs_input_dims_eq_one
                )
            )
 
    @classmethod
    def InferInputsDimsEqOne(
        cls,
        dag_dims_gen_instruction: DAGDimsGenInstruction,
        infer_ctx: DimsIsEqOneInferContext
    ):
        idx = dag_dims_gen_instruction.dag.source_tensor_index
        input_dims_eq_one = dag_dims_gen_instruction.dims.source_tensor_dim_eq_one
        infer_ctx.current_source_tensor_dim_eq_one[idx] = input_dims_eq_one
       

    @classmethod
    def InferInputsDimsEqOne(
        cls,
        dag_dims_gen_instruction: DAGDimsGenInstruction,
        infer_ctx: DimsIsEqOneInferContext
    ):
        rhs_input_dims_eq_one = (
            dag_dims_gen_instruction.dims.rhs_source_tensor_dim_eq_one
        )
        infer_ctx.current_source_tensor_dim_eq_one.append(rhs_input_dims_eq_one)
    

class AddBinaryCloneUpstream:

    @classmethod
    def GetPatchedDAGDimsGenInstruction(
        cls,
        dag_dims_gen_instruction: DAGDimsGenInstruction,
        infer_ctx: DimsIsEqOneInferContext
    ):
        lhs_input_dims_eq_one = infer_ctx.current_source_tensor_dim_eq_one[
            dag_dims_gen_instruction.dag.lhs_source_tensor_index
        ]
        rhs_input_dix = dag_dims_gen_instruction.dag.rhs_source_tensor_index
        rhs_input_dims_eq_one = infer_ctx.current_source_tensor_dim_eq_one[
            rhs_input_dix
        ]
        middle_dims_eq_one = [
            x and y
            for x, y in zip(lhs_input_dims_eq_one, rhs_input_dims_eq_one)
        ]
        if middle_dims_eq_one != rhs_input_dims_eq_one:
            yield DAGDimsGenInstruction(
                dag=dag_generator.AddUnaryUpstreamOp(
                    source_tensor_index=rhs_input_dix,
                    dag_tag=dag_dims_gen_instruction.dag.dag_tag
                ),
                dims=dims_generator.AddUnaryUpstreamOp(
                    source_tensor_dim_eq_one=middle_dims_eq_one
                )
            )
        if lhs_input_dims_eq_one != middle_dims_eq_one:
            yield DAGDimsGenInstruction(
                dag=dag_generator.AddUnaryUpstreamOp(
                    source_tensor_index=rhs_input_dix,
                    dag_tag=dag_dims_gen_instruction.dag.dag_tag
                ),
                dims=dims_generator.AddUnaryUpstreamOp(
                    source_tensor_dim_eq_one=lhs_input_dims_eq_one
                )
            )
        yield dag_dims_gen_instruction

    @classmethod
    def InferInputsDimsEqOne(
        cls,
        dag_dims_gen_instruction: DAGDimsGenInstruction,
        infer_ctx: DimsIsEqOneInferContext
    ):
        rhs_output_idx = dag_dims_gen_instruction.dag.rhs_source_tensor_index
        infer_ctx.current_source_tensor_dim_eq_one.pop(rhs_output_idx)


class MarkFinalSourceTensor:

    @classmethod
    def GetPatchedDAGDimsGenInstruction(
        cls,
        dag_dims_gen_instruction: DAGDimsGenInstruction,
        infer_ctx: DimsIsEqOneInferContext
    ):
        yield dag_dims_gen_instruction

    @classmethod
    def InferInputsDimsEqOne(
        cls,
        dag_dims_gen_instruction: DAGDimsGenInstruction,
        infer_ctx: DimsIsEqOneInferContext
    ):
        output_idx = dag_dims_gen_instruction.dag.source_tensor_index
        infer_ctx.current_source_tensor_dim_eq_one.pop(output_idx)


kDAGGenClassToDAGDimsGenClassMap = {
    dag_generator.Nope: Nope,
    dag_generator.AddSinkTensor: AddSinkTensor,
    dag_generator.AddUnaryUpstreamOp: AddUnaryUpstreamOp,
    dag_generator.AddBinaryUpstreamOp: AddBinaryUpstreamOp,
    dag_generator.InsertBinaryUpstreamOp: InsertBinaryUpstreamOp,
    dag_generator.AddBinaryCloneUpstream: AddBinaryCloneUpstream,
    dag_generator.MarkFinalSourceTensor: MarkFinalSourceTensor,
}


class DAGDimsGenerator:
    def __init__(self):
        pass

    def Generate(
        self,
        dag_gen_instructions: List[dag_generator.DAGGenInstruction],
        dims_gen_instructions: List[dims_generator.DimsGenInstruction]
    ) -> List[dag_generator.DAGDimsGenInstruction]:
        infer_ctx = DimsIsEqOneInferContext()
        def CreateDAGDimsGenInstructions(dag_dims_gen_instruction):
            dag_gen_class = type(dag_dims_gen_instruction.dag)
            cls = kDAGGenClassToDAGDimsGenClassMap[dag_gen_class]
            assert (
                len(infer_ctx.current_source_tensor_dim_eq_one)
                == infer_ctx.current_num_source_tensors
            )
            yield from cls.GetPatchedDAGDimsGenInstruction(
                dag_dims_gen_instruction,
                infer_ctx
            )
            cls.InferInputsDimsEqOne(dag_dims_gen_instruction, infer_ctx)

        return [
            dag_dims_gen_instruction
            for dag_dims_gen_instruction in CreateDAGDimsGenInstructions(x)
            for x in _ZipDAGDimsInstr(dag_gen_instructions, dims_gen_instructions)
        ]

def _ZipDAGDimsInstr(dag_gen_instructions, dims_gen_instructions):
    return [
        DAGDimsGenInstruction(*instruction_tuple)
        for instruction_tuple in zip(dag_gen_instructions, dims_gen_instructions)
    ]
