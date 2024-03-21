from dataclasses import dataclass
from .pick_weight import PickWeight
from typing import List
from collections import namedtuple
import functools
import random

@dataclass
class DAGGenTypePickProbability:
    nope: PickWeight
    add_sink_tensor: PickWeight
    add_unary_op: PickWeight
    # append to core DAG.
    add_binary_op: PickWeight
    # modify core DAG
    insert_binary_op: PickWeight
    add_binary_clone: PickWeight
    add_source_op: PickWeight

@dataclass
class DAGGenRequirement:
    max_width: int
    max_instructions: int
    dag_tag: str
    pick_probability: DAGGenTypePickProbability

@dataclass
class Nope:

    def IsValidSourceTensorIndex(
        self,
        num_core_source_tensors: int,
        num_source_tensors: int
    ) -> bool:
        return True
    
    @classmethod
    def GetDeltaNumSourceTensors(cls):
        return 0

    @classmethod
    def RandomGenerate(
        cls,
        requirement: DAGGenRequirement,
        num_core_source_tensors: int,
        num_source_tensors: int
    ):
        return Nope()

    @classmethod
    def GetWeight(
        cls,
        pick_probability: DAGGenTypePickProbability
    ) -> float:
        return pick_probability.nope.weight

    @classmethod
    def IsValidNumSourceTensors(
        cls,
        num_core_source_tensors: int,
        num_source_tensors: int,
        requirement: DAGGenRequirement
    ) -> bool:
        return True

@dataclass
class AddSinkTensor:
    
    def IsValidSourceTensorIndex(
        self,
        num_core_source_tensors: int,
        num_source_tensors: int
    ) -> bool:
        return True
    
    @classmethod
    def GetDeltaNumSourceTensors(cls):
        return 1

    @classmethod
    def RandomGenerate(
        cls,
        requirement: DAGGenRequirement,
        num_core_source_tensors: int,
        num_source_tensors: int
    ):
        return AddSinkTensor()

    @classmethod
    def GetWeight(
        cls,
        pick_probability: DAGGenTypePickProbability
    ) -> float:
        return pick_probability.add_sink_tensor.weight

    @classmethod
    def IsValidNumSourceTensors(
        cls,
        num_core_source_tensors: int,
        num_source_tensors: int,
        requirement: DAGGenRequirement
    ) -> bool:
        return num_source_tensors <= requirement.max_width

@dataclass
class AddUnaryOp:
    source_tensor_index: int
    dag_tag: str

    def IsValidSourceTensorIndex(
        self,
        num_core_source_tensors: int,
        num_source_tensors: int
    ) -> bool:
        return (self.source_tensor_index >= num_core_source_tensors
            and self.source_tensor_index < num_source_tensors
        )
        
    @classmethod
    def GetDeltaNumSourceTensors(cls):
        return 0

    @classmethod
    def RandomGenerate(
        cls,
        requirement: DAGGenRequirement,
        num_core_source_tensors: int,
        num_source_tensors: int
    ):
        random_int = random.randomint(
            num_core_source_tensors,
            num_source_tensors - 1
        )
        return AddUnaryOp(
            source_tensor_index=source_tensor_index,
            dag_tag=requirement.dag_tag
        )

    @classmethod
    def GetWeight(
        cls,
        pick_probability: DAGGenTypePickProbability
    ) -> float:
        return pick_probability.add_unary_op.weight

    @classmethod
    def IsValidNumSourceTensors(
        cls,
        num_core_source_tensors: int,
        num_source_tensors: int,
        requirement: DAGGenRequirement
    ) -> bool:
        return num_core_source_tensors < num_source_tensors

@dataclass
class AddBinaryOp:
    source_tensor_index: int
    dag_tag: str

    def IsValidSourceTensorIndex(
        self,
        num_core_source_tensors: int,
        num_source_tensors: int
    ) -> bool:
        return (self.source_tensor_index >= num_core_source_tensors
            and self.source_tensor_index < num_source_tensors
        )
        
    @classmethod
    def GetDeltaNumSourceTensors(cls):
        return 1

    @classmethod
    def RandomGenerate(
        cls,
        requirement: DAGGenRequirement,
        num_core_source_tensors: int,
        num_source_tensors: int
    ):
        random_int = random.randomint(
            num_core_source_tensors,
            num_source_tensors - 1
        )
        return AddBinaryOp(
            source_tensor_index=source_tensor_index,
            dag_tag=requirement.dag_tag
        )

    @classmethod
    def GetWeight(
        cls,
        pick_probability: DAGGenTypePickProbability
    ) -> float:
        return pick_probability.add_binary_op.weight

    @classmethod
    def IsValidNumSourceTensors(
        cls,
        num_core_source_tensors: int,
        num_source_tensors: int,
        requirement: DAGGenRequirement
    ) -> bool:
        return (num_source_tensors <= requirement.max_width
            and num_core_source_tensors < num_source_tensors
        )

@dataclass
class InsertBinaryOp:
    source_tensor_index: int
    dag_tag: str

    def IsValidSourceTensorIndex(
        self,
        num_core_source_tensors: int,
        num_source_tensors: int
    ) -> bool:
        return (self.source_tensor_index >= 0
            and self.source_tensor_index < num_core_source_tensors
        )
    
    @classmethod
    def GetDeltaNumSourceTensors(cls):
        return 1

    @classmethod
    def RandomGenerate(
        cls,
        requirement: DAGGenRequirement,
        num_core_source_tensors: int,
        num_source_tensors: int
    ):
        random_int = random.randomint(
            0,
            num_core_source_tensors - 1
        )
        return InsertBinaryOp(
            source_tensor_index=source_tensor_index,
            dag_tag=requirement.dag_tag
        )

    @classmethod
    def GetWeight(
        cls,
        pick_probability: DAGGenTypePickProbability
    ) -> float:
        return pick_probability.insert_binary_op.weight

    @classmethod
    def IsValidNumSourceTensors(
        cls,
        num_core_source_tensors: int,
        num_source_tensors: int,
        requirement: DAGGenRequirement
    ) -> bool:
        return (num_source_tensors <= requirement.max_width
            and num_core_source_tensors > 0
        )

@dataclass
class AddBinaryClone:
    lhs_source_tensor_index: int
    rhs_source_tensor_index: int

    def IsValidSourceTensorIndex(
        self,
        num_core_source_tensors: int,
        num_source_tensors: int
    ) -> bool:
        return (self.lhs_source_tensor_index >= 0
            and self.lhs_source_tensor_index < num_source_tensors
            and self.rhs_source_tensor_index >= num_core_source_tensors
            and self.rhs_source_tensor_index < num_source_tensors
        )

    @classmethod
    def GetDeltaNumSourceTensors(cls):
        return -1

    @classmethod
    def RandomGenerate(
        cls,
        requirement: DAGGenRequirement,
        num_core_source_tensors: int,
        num_source_tensors: int
    ):
        lhs_random_int = random.randomint(
            0, num_source_tensors - 1
        )
        rhs_random_int = random.randomint(
            num_core_source_tensors,
            num_source_tensors - 1
        )
        return AddBinaryClone(
            lhs_source_tensor_index=lhs_random_int,
            rhs_source_tensor_index=rhs_random_int,
        )

    @classmethod
    def GetWeight(
        cls,
        pick_probability: DAGGenTypePickProbability
    ) -> float:
        return pick_probability.add_binary_clone.weight

    @classmethod
    def IsValidNumSourceTensors(
        cls,
        num_core_source_tensors: int,
        num_source_tensors: int,
        requirement: DAGGenRequirement
    ) -> bool:
        return num_core_source_tensors < num_source_tensors

@dataclass
class AddSourceOp:
    source_tensor_index: int

    def IsValidSourceTensorIndex(
        self,
        num_core_source_tensors: int,
        num_source_tensors: int
    ) -> bool:
        return (self.source_tensor_index >= num_core_source_tensors
            and self.source_tensor_index < num_source_tensors
        )

    @classmethod
    def GetDeltaNumSourceTensors(cls):
        return -1

    @classmethod
    def RandomGenerate(
        cls,
        requirement: DAGGenRequirement,
        num_core_source_tensors: int,
        num_source_tensors: int
    ):
        random_int = random.randomint(
            num_core_source_tensors,
            num_source_tensors - 1
        )
        return AddSourceOp(
            source_tensor_index=source_tensor_index,
            dag_tag=requirement.dag_tag
        )

    @classmethod
    def GetWeight(
        cls,
        pick_probability: DAGGenTypePickProbability
    ) -> float:
        return pick_probability.add_source_op.weight

    @classmethod
    def IsValidNumSourceTensors(
        cls,
        num_core_source_tensors: int,
        num_source_tensors: int,
        requirement: DAGGenRequirement
    ) -> bool:
        return num_core_source_tensors < num_source_tensors

# DAGGenInstruction = ( Nope
#                     | AddSinkTensor
#                     | AddUnaryOp
#                     | AddBinaryOp
#                     | InsertBinaryOp
#                     | AddBinaryClone
#                     | AddSourceOp
#                     )

kDAGGenInstructionClasses = [
    Nope,
    AddSinkTensor,
    AddUnaryOp,
    AddBinaryOp,
    InsertBinaryOp,
    AddBinaryClone,
    AddSourceOp,
]

def GetTotalWeight(
    pick_probability: DAGGenTypePickProbability,
    dag_gen_classes: List[type]
) -> float:
    def GetWeight(dag_gen_class):
        return dag_gen_class.GetWeight(pick_probability)
    return functools.reduce(
        lambda a, b: a + b,
        functools.map(GetWeight, dag_gen_classes)
    )

class DAGGenClassGenerator:
    def __init__(
        self,
        pick_probability: DAGGenTypePickProbability
    ):
        self.pick_probability = pick_probability
    
    def GetRandomDAGGenClass(
        self,
        num_core_source_tensors: int,
        num_source_tensors: int,
        requirement: DAGGenRequirement
    ) -> type:
        def IsValidNumSources(dag_gen_class):
            return dag_gen_class.IsValidNumSourceTensors(
                num_core_source_tensors,
                num_source_tensors,
                requirement
            )
        rolling_ranges = type(self)._MakeRollingRange(
            self.pick_probability,
            [x for x in kDAGGenInstructionClasses if IsValidNumSources(x)]
        )
        def Roll():
            random_int = random.randomint(0, type(self)._RollingLimit())
            for start, end, dag_gen_class in rolling_ranges:
                if random_int >= start and random_int < end:
                    return dag_gen_class
            return None
        kTryCnt = 10
        for i in range(kTryCnt):
            cls = Roll()
            if cls is not None:
                return cls
        return Nope()

    DAGGenClassRollingRange = namedtuple(
        "DAGGenClassRollingRange",
        ["start", "end", "dag_gen_class"]
    )

    @classmethod
    def _MakeRollingRange(
        cls,
        pick_probability: DAGGenTypePickProbability,
        dag_gen_classes: List[type]
    ) -> List[DAGGenClassRollingRange]:
        total_weight = GetTotalWeight(pick_probability, dag_gen_classes)
        start = 0
        def GetRange(dag_gen_class):
            nonlocal start
            current_start = start
            current_end = current_start
            weight = dag_gen_class.GetWeight(pick_probability)
            start += weight * cls._RollingLimit() / total_weight
            return DAGGenClassRollingRange(
                .start=current_start,
                .end=current_end,
                .dag_gen_class=dag_gen_class,
            )
        return [GetRange(cls) for cls in dag_gen_classes]

    @classmethod
    def _RollingLimit(cls):
        return 10000


class ConstDAGGenInstructions:
    def __init__(
        self,
        instructions: List["DAGGenInstruction"]
    ):
        self.instructions = instructions[:]
        self.current_num_source_tensors = 0
    
    def TryPop(self):
        if len(self.instructions) == 0:
            return
        top = self.instructions[0]
        self.current_num_source_tensors += type(top).GetDeltaNumSourceTensors()
        self.instructions.pop(0)

class MutDAGGenInstructions:
    def __init__(self):
        self.instructions = []
        self.current_num_source_tensors = 0

    def Push(
        self,
        instruction: "DAGGenInstruction"
    ):
        self.instructions.insert(0, instruction)
        top = instruction
        self.current_num_source_tensors += type(top).GetDeltaNumSourceTensors()

class DAGGenContext:
    def __init__(
        self,
        requirement: DAGGenRequirement,
        core_dag_gen_instructions: List["DAGGenInstruction"]
    ):
        self.requirement = requirement
        self.core_dag_gen_instructions = ConstDAGGenInstructions(
            core_dag_gen_instructions
        )
        self.result_dag_gen_instructions = MutDAGGenInstructions()

    def result_instructions(self):
        return self.result_dag_gen_instructions.instructions

    def GenerateOneInstruction(self, Converter):
        self.core_dag_gen_instructions.TryPop()
        num_core_source_tensors = (
            self.core_dag_gen_instructions.current_num_source_tensors
        )
        num_source_tensors = (
            self.result_dag_gen_instructions.current_num_source_tensors
        )
        new_instruction = Converter(num_core_source_tensors, num_source_tensors)
        is_valid = new_instruction.IsValidSourceTensorIndex(
            num_core_source_tensors,
            num_source_tensors
        )
        if is_valid:
            ctx.result_dag_gen_instructions.Push(new_instruction)

class DAGGenerator:
    def __init__(self, requirement: DAGGenRequirement):
        self.requirement = requirement
        self.dag_gen_class_generator = DAGGenClassGenerator(
            requirement.pick_probability
        )
    
    # Instructions generating sink nodes of DAG are on the front of list.
    def Generate(
            self,
            core_instructions: List["DAGGenInstruction"]
        ) -> List["DAGGenInstruction"]:
        core_instructions = core_instructions[:]
        ctx = DAGGenContext(self.requirement, core_instructions)
        def MakeInstruction(num_core_sources: int, num_sources: int):
            return self._MakeRandomInstruction(ctx, num_core_sources, num_sources)
        for i in range(self.requirement.max_instructions):
            ctx.GenerateOneInstruction(MakeInstruction)
        return list(reversed(ctx.result_instructions()))

    def _MakeRandomInstruction(
            self,
            ctx: DAGGenContext,
            num_core_source_tensors: int,
            num_source_tensors: int
        ):
        dag_gen_class = self.dag_gen_class_generator.GetRandomDAGGenClass(
            num_core_source_tensors,
            num_source_tensors,
            self.requirement
        )
        return dag_gen_class.RandomGenerate(
            ctx.requirement,
            num_core_source_tensors,
            num_source_tensors
        )