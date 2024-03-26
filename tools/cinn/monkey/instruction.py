from dataclasses import dataclass
import .dag_generator as dag_generator
import .dims_eq1_generator as dims_eq1_generator
import .op_name_generator as op_name_generator
from typing import List
from .hash_combine import HashCombine
from .instruction_id import InstructionId, MakeUniqueInstructionId

kBaseClassModules = (dag_generator, dims_eq1_generator, op_name_generator)

@dataclass
class Instruction:
    @classmethod
    def GetDAGGenClassToDerivedClassMap(cls):
        return kDAGGenClassToDerivedClass


def GetHashValue(class_name):
    def Func(self):
        hash_value = 0
        for m in kBaseClassModules:
            cls = getattr(m, class_name)
            hash_func = getattr(cls, "__hash__")
            hash_value = HashCombine(hash_value, hash_func(self))
    return Func

def GetComponent(self, base_class):
    dag_gen_class = self.GetDAGGenClass()
    cls = base_class.GetDAGGenClassToDerivedClassMap()[dag_gen_class]
    params = {
        field_name: getattr(self, field_name)
        for field_name in cls.__annotations__.keys()
    }
    return cls(*params)

def MergeClass(dag_gen_class, modules):
    class_name = dag_gen_class.__name__
    base_classes = (Instruction, InstructionId,)
    base_classes += tuple(getattr(m, class_name) for m in modules)
    methods = dict(
        __hash__=GetHashValue(class_name),
        GetComponent=GetComponent,
        GetDAGGenClass=lambda self:dag_gen_class
    )
    return dataclass(type(class_name, base_classes, methods))

Nope = MergeClass(dag_generator.Nope, kBaseClassModules)
AddSinkTensor = MergeClass(dag_generator.AddSinkTensor, kBaseClassModules)
AddUnaryOp = MergeClass(dag_generator.AddUnaryOp, kBaseClassModules)
AddBinaryOp = MergeClass(dag_generator.AddBinaryOp, kBaseClassModules)
AddBinaryClone = MergeClass(dag_generator.AddBinaryClone, kBaseClassModules)
AddSourceOp = MergeClass(dag_generator.AddSourceOp, kBaseClassModules)


kDAGGenClassToDerivedClass = {
    dag_generator.Nope: Nope,
    dag_generator.AddSinkTensor: AddSinkTensor,
    dag_generator.AddUnaryOp: AddUnaryOp,
    dag_generator.AddBinaryOp: AddBinaryOp,
    dag_generator.AddBinaryClone: AddBinaryClone,
    dag_generator.AddSourceOp: AddSourceOp,
}

class IrGenerator:
    def __init__(self):
        pass

    def Generate(
        self,
        dag_gen_instructions: List["DAGGenInstruction"],
        instruction_ids: List[InstructionId],
        dims_eq1_gen_instructions: List["DimsEq1GenInstruction"],
        op_name_gen_instructions: List["OpNameGenInstruction"]
    ) -> List["Instruction"]:
        def CreateIrGenInstruction(dag_gen_instr, *other_instr):
            for instr in [dag_gen_instr, *other_instr]:
                params.update(vars(instr))
            ir_gen_class = kDAGGenClassToDerivedClass[type(dag_gen_instr)]
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
