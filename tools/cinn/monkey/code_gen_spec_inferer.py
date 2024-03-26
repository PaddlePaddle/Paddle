from dataclasses import dataclass
import .dag_generator as dag_generator
import .dims_eq1_signature_inferer as dims_eq1_signature_inferer
import .shape_signature_inferer as shape_signature_inferer
from typing import List
from .hash_combine import HashCombine

kBaseClassModules = (dims_eq1_signature_inferer, shape_signature_inferer)

@dataclass
class CodeGenSpec:
    
    @classmethod
    def GetDAGGenClassToDerivedClassMap(cls):
        return kDAGGenClassToDerivedClass


def MergeCodeGenSpecClßass(class_name, modules):
    base_classes = (CodeGenSpec,)
    base_classes += tuple(getattr(m, class_name) for m in modules)
    return dataclass(type(class_name, base_classes, {}))

Nope = MergeCodeGenSpecClass("Nope", kBaseClassModules)
AddSinkTensor = MergeCodeGenSpecClass("AddSinkTensor", kBaseClassModules)
AddUnaryOp = MergeCodeGenSpecClass("AddUnaryOp", kBaseClassModules)
AddBinaryOp = MergeCodeGenSpecClass("AddBinaryOp", kBaseClassModules)
AddBinaryClone = MergeCodeGenSpecClass("AddBinaryClone", kBaseClassModules)
AddSourceOp = MergeCodeGenSpecClass("AddSourceOp", kBaseClassModules)


kDAGGenClassToDerivedClass = {
    dag_generator.Nope: Nope,
    dag_generator.AddSinkTensor: AddSinkTensor,
    dag_generator.AddUnaryOp: AddUnaryOp,
    dag_generator.AddBinaryOp: AddBinaryOp,
    dag_generator.AddBinaryClone: AddBinaryClone,
    dag_generator.AddSourceOp: AddSourceOp,
}

class CodeGenSpecInferer:
    def __init__(self):
        pass
    
    def Generate(
        self,
        dag_gen_instructions: List["DAGGenInstruction"],
        dims_eq1_signatures: List["DimsEq1Signature"],
        shape_signatures: List["ShapeSignature"]
    ) -> List["Instruction"]:
        def CreateCodeGenSpec(dag_gen_instr, *signatures):
            params = dict()
            for signature in signatures:
                params.update(vars(signature))
            dag_gen_cls = type(dag_gen_instr)
            code_gen_spec_class = kDAGGenClassToDerivedClass[dag_gen_cls]
            return code_gen_spec_class(*params)
        return [
            CreateCodeGenSpec(*x)
            for x in zip(dag_gen_instructions, dims_eq1_signatures, shape_signatures)
        ]
