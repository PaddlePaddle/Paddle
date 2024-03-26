from dataclasses import dataclass
import .dag_generator as dag_generator
import .dims_eq1_generator as dims_eq1_generator
import .op_name_generator as op_name_generator
from typing import List
from .hash_combine import HashCombine
from .instruction_id import InstructionId, MakeUniqueInstructionId

kBaseClassModules = (dag_generator, dims_eq1_generator, op_name_generator)

def GetHashValue(class_name):
    def Func(self):
        hash_value = 0
        for m in kBaseClassModules:
            cls = getattr(m, class_name)
            hash_func = getattr(cls, "__hash__")
            hash_value = HashCombine(hash_value, hash_func(self))
    return Func

def MergeInstructionClass(class_name, modules):
    base_classes = (InstructionId,)
    base_classes += tuple(getattr(m, class_name) for m in modules)
    methods = dict(
        __hash__=GetHashValue(class_name)
    )
    return dataclass(type(class_name, base_classes, methods))

Nope = MergeInstructionClass("Nope", kBaseClassModules)
AddSinkTensor = MergeInstructionClass("AddSinkTensor", kBaseClassModules)
AddUnaryOp = MergeInstructionClass("AddUnaryOp", kBaseClassModules)
AddBinaryOp = MergeInstructionClass("AddBinaryOp", kBaseClassModules)
InsertBinaryOp = MergeInstructionClass("InsertBinaryOp", kBaseClassModules)
AddBinaryClone = MergeInstructionClass("AddBinaryClone", kBaseClassModules)
AddSourceOp = MergeInstructionClass("AddSourceOp", kBaseClassModules)


Instruction = Union[
    Nope,
    AddSinkTensor,
    AddUnaryOp,
    AddBinaryOp,
    InsertBinaryOp,
    AddBinaryClone,
    AddSourceOp
]

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
    def __init__(self):
        pass

    def Generate(
        self,
        dag_gen_instructions: List["DAGGenInstruction"],
        instruction_ids: List[InstructionId]
        dims_eq1_gen_instructions: List["DimsEq1GenInstruction"],
        op_name_gen_instructions: List["OpNameGenInstruction"]
    ) -> List["Instruction"]:
        def CreateIrGenInstruction(dag_gen_instr, *other_instr):
            for instr in [dag_gen_instr, *other_instr]:
                params.update(vars(instr))
            ir_gen_class = kDAGGenClassToIrGenClassMap[type(dag_gen_instr)]
            return ir_gen_class(*params)
        return [
            CreateIrGenInstruction(*triple)
            for triple in zip(
                dag_gen_instructions,
                instruction_ids,
                dims_eq1_gen_instructions,
                op_name_gen_instructions
            )
        ]
