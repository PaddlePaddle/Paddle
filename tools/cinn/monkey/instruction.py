from dataclasses import dataclass
import .dag_generator as dag_generator
import .dims_eq1_generator as dims_eq1_generator
import .op_name_generator as op_name_generator
import .tensor_name_generator as tensor_name_generator
from typing import List
from .hash_combine import HashCombine
from .instruction_id import InstructionId, MakeUniqueInstructionId


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

kComponentClasses = (
    dag_generator.DAGGenInstruction,
    dims_eq1_generator.DimsEq1GenInstruction,
    op_name_generator.OpNameGenInstruction,
    tensor_name_generator.TensorNameGenInstruction
)

def MergeClass(dag_gen_class, component_classes):
    class_name = dag_gen_class.__name__
    base_classes = (Instruction, InstructionId,)
    base_classes += tuple(
        component_class.GetDAGGenClassToDerivedClassMap()[dag_gen_cls]
        for component_class in component_classes
    )
    methods = dict(
        __hash__=GetHashValue(class_name),
        GetComponent=GetComponent,
        GetDAGGenClass=lambda self:dag_gen_class
    )
    return dataclass(type(class_name, base_classes, methods))

Nope = MergeClass(dag_generator.Nope, kComponentClasses)
AddSinkTensor = MergeClass(dag_generator.AddSinkTensor, kComponentClasses)
AddUnaryOp = MergeClass(dag_generator.AddUnaryOp, kComponentClasses)
AddBinaryOp = MergeClass(dag_generator.AddBinaryOp, kComponentClasses)
AddBinaryClone = MergeClass(dag_generator.AddBinaryClone, kComponentClasses)
AddSourceOp = MergeClass(dag_generator.AddSourceOp, kComponentClasses)


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
        op_name_gen_instructions: List["OpNameGenInstruction"],
        tensor_name_gen_instructions: List["TensorNameGenInstruction"]
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
                op_name_gen_instructions,
                tensor_name_gen_instructions
            )
        ]
