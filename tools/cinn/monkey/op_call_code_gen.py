from dataclasses import dataclass

@dataclass
class CodeGenRequirement:
    module_name: str

@dataclass
class CodeGenContext:
    requirement: CodeGenRequirement
    instruction: Instruction
    code_gen_spec: CodeGenSpec


class Add:
    @classmethod
    def Paddle(cls, ctx: CodeGenContext):
        lhs_input_tensor = ctx.code_gen_spec.lhs_input_tensor_name
        rhs_input_tensor = ctx.code_gen_spec.rhs_input_tensor_name
        output_tensor = ctx.code_gen_spec.output_tensor_name
        return f"{output_tensor} = {lhs_input_tensor} + {rhs_input_tensor}"

class Negative:
    @classmethod
    def Paddle(cls, ctx: CodeGenContext):
        input_tensor_name = ctx.code_gen_spec.input_tensor_name
        output_tensor_name = ctx.code_gen_spec.output_tensor_name
        return f"{output_tensor_name} = -{input_tensor_name}"


class ReduceSum:
    @classmethod
    def Paddle(cls, ctx: CodeGenContext):
        axes = cls.GetReduceAxes(ctx)
        output_tensor = ctx.code_gen_spec.output_tensor_name
        input_tensor = ctx.code_gen_spec.input_tensor_name
        return f"{output_tensor} = {input_tensor}.sum(axis={axes}, keepdim=True)"

    @classmethod
    def GetReduceAxes(cls, ctx: CodeGenContext):
        input_dims_eq1 = ctx.code_gen_spec.input_dims_eq1
        output_dims_eq1 = ctx.code_gen_spec.output_dims_eq1
        assert input_dims_eq1 == [
            x and y
            for x,y in zip(input_dims_eq1, output_dims_eq1)
        ]
        assert output_dims_eq1 == [
            x or y
            for x,y in zip(input_dims_eq1, output_dims_eq1)
        ]
        is_reduced = [
            (not x) and y
            for x, y in zip(input_dims_eq1, output_dims_eq1)
        ]
        return [
            axis
            for axis in ([i] if is_reduced else [])
            for i, is_reduced in enumerate(is_reduced)
        ]

class Expand:
    @classmethod
    def Paddle(cls, ctx: CodeGenContext):
        output_tensor = ctx.code_gen_spec.output_tensor_name
        m = ctx.requirement.module_name
        input_tensor = ctx.code_gen_spec.input_tensor_name
        output_shape = ctx.code_gen_spec.output_shape
        return f"{output_tensor} = {m}.expand({input_tensor}, shape={output_shape})"


class Ones:
    @classmethod
    def Paddle(cls, ctx: CodeGenContext):
        m = ctx.requirement.module_name
        output_tensor_name = ctx.code_gen_spec.output_tensor_name
        output_shape = ctx.code_gen_spec.output_shape
        return f"{output_tensor_name} = {m}.ones({output_shape})"


class Zeros:
    @classmethod
    def Paddle(cls, ctx: CodeGenContext):
        m = ctx.requirement.module_name
        output_tensor_name = ctx.code_gen_spec.output_tensor_name
        output_shape = ctx.code_gen_spec.output_shape
        return f"{output_tensor_name} = {m}.zeros({output_shape})"