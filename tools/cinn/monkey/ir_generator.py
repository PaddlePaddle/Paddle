from dataclasses import dataclass
import .dag_generator as dag_generator
import .dims_generator as dims_generator
import .op_name_generator as op_name_generator
from typing import List

@dataclass
class IrGenDimDescriptor:
    static_dim_size: List[int]

@dataclass
class IrGenRequirement:
    dims_descriptor: IrGenDimDescriptor

@dataclass
class Nope:
    dag: dag_generator.Nope
    dims: dims_generator.Nope
    op_name: op_name_generator.Nope
    dims_descriptor: IrGenDimDescriptor

    def CheckNumDims(self):
        pass

@dataclass
class AddSinkTensor:
    dag: dag_generator.AddSinkTensor
    dims: dims_generator.AddSinkTensor
    op_name: op_name_generator.AddSinkTensor
    dims_descriptor: IrGenDimDescriptor

    def CheckNumDims(self):
        pass


@dataclass
class AddUnaryOp:
    dag: dag_generator.AddUnaryOp
    dims: dims_generator.AddUnaryOp
    op_name: op_name_generator.AddUnaryOp
    dims_descriptor: IrGenDimDescriptor

    def CheckNumDims(self):
        assert (
            len(dims_descriptor.static_dim_size)
            == len(dims.source_tensor_dim_eq_one)
        )


@dataclass
class AddBinaryOp:
    dag: dag_generator.AddBinaryOp
    dims: dims_generator.AddBinaryOp
    op_name: op_name_generator.AddBinaryOp
    dims_descriptor: IrGenDimDescriptor

    def CheckNumDims(self):
        assert (
            len(dims_descriptor.static_dim_size)
            == len(dims.lhs_source_tensor_dim_eq_one)
        )
        assert (
            len(dims_descriptor.static_dim_size)
            == len(dims.rhs_source_tensor_dim_eq_one)
        )


@dataclass
class InsertBinaryOp:
    dag: dag_generator.InsertBinaryOp
    dims: dims_generator.InsertBinaryOp
    op_name: op_name_generator.InsertBinaryOp
    dims_descriptor: IrGenDimDescriptor

    def CheckNumDims(self):
        assert (
            len(dims_descriptor.static_dim_size)
            == len(dims.rhs_source_tensor_dim_eq_one)
        )


@dataclass
class AddBinaryClone:
    dag: dag_generator.AddBinaryClone
    dims: dims_generator.AddBinaryClone
    op_name: op_name_generator.AddBinaryClone
    dims_descriptor: IrGenDimDescriptor


@dataclass
class AddSourceOp:
    dag: dag_generator.AddSourceOp
    dims: dims_generator.AddSourceOp
    op_name: op_name_generator.AddSourceOp
    dims_descriptor: IrGenDimDescriptor


IrGenInstruction = ( Nope
                   | AddSinkTensor
                   | AddUnaryOp
                   | AddBinaryOp
                   | InsertBinaryOp
                   | AddBinaryClone
                   | AddSourceOp
                   )


kDAGGenClassToIrGenClassMap = {
    dag_generator.Nope: Nope,
    dag_generator.AddSinkTensor: AddSinkTensor,
    dag_generator.AddUnaryOp: AddUnaryOp,
    dag_generator.AddBinaryOp: AddBinaryOp,
    dag_generator.InsertBinaryOp: InsertBinaryOp,
    dag_generator.AddBinaryClone: AddBinaryClone,
    dag_generator.AddSourceOp: AddSourceOp,
}

class IrGenGenerator:
    def __init__(self, requirement: IrGenRequirement):
        self.requirement = requirement
    
    def Generate(
        self,
        dag_gen_instructions: List[dag_generator.DAGGenInstruction],
        dims_gen_instructions: List[dims_generator.DimsGenInstruction],
        op_name_gen_instructions: List[op_name_generator.OpNameGenInstruction]
    ) -> List[IrGenInstruction]:
        def CreateIrGenInstruction(triple):
            dag_gen_instr, dims_gen_instr, op_name_gen_instr = triple
            dag_gen_class = type(dag_gen_instr)
            ir_gen_class = kDAGGenClassToIrGenClassMap[dag_gen_class]
            instruction = ir_gen_class(
                dag=dag_gen_instr,
                dims=dims_gen_instr,
                op_name=op_name_gen_instr,
                dims_descriptor=self.requirement.dims_descriptor
            )
            instruction.CheckNumDims()
            return instruction
        return [
            CreateIrGenInstruction(triple)
            for triple in zip(
                dag_gen_instructions,
                dims_gen_instructions,
                op_name_gen_instructions
            )
        ]
