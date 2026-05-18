# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import hashlib
import importlib.util
import keyword
import re
import types
from functools import cache
from importlib.machinery import SourceFileLoader
from pathlib import Path

from paddle.autograd.backward_utils import ValueDict
from paddle.pir import Value, is_fake_value

FAST_KERNEL_CACHE_DIR = Path("~/.cache/paddle/sot/fast_kernel").expanduser()

INT_ARRAY_INPUT_NAMES = {
    "axis",
    "axes",
    "dilations",
    "ends",
    "kernel_size",
    "offsets",
    "out_size",
    "output_shape",
    "output_size",
    "paddings",
    "repeat_times",
    "sections",
    "shape",
    "starts",
    "steps",
    "strides",
}


class FastKernelRuntimeError(RuntimeError):
    pass


class FastKernelRuntimeUnsupported(FastKernelRuntimeError):
    pass


@cache
def kernel_arg_name_map() -> dict[str, list[str]]:
    from paddle.base import core

    kernel_ops = getattr(core.eager, "kernel_ops", None)
    if kernel_ops is None or not hasattr(
        kernel_ops, "get_kernel_ops_args_info"
    ):
        raise FastKernelRuntimeError(
            "core.eager.kernel_ops.get_kernel_ops_args_info is not available. "
            "Fast kernel runtime requires generated direct kernel pybind "
            "metadata and does not fall back to YAML or run_program."
        )
    return kernel_ops.get_kernel_ops_args_info()


class FunctionModule:
    def __init__(
        self, module: types.ModuleType, entry_point: str, path: Path
    ) -> None:
        self.module = module
        self.entry_point = entry_point
        self.path = path

    def get_function(self) -> types.FunctionType:
        func = getattr(self.module, self.entry_point, None)
        if func is None or not isinstance(func, types.FunctionType):
            raise FastKernelRuntimeError(
                f"Function '{self.entry_point}' not found in {self.path}."
            )
        return func


def load_function_module(module_path: Path, entry_point: str) -> FunctionModule:
    module_name = module_path.stem
    loader = SourceFileLoader(module_name, str(module_path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise FastKernelRuntimeError(
            f"Could not create module spec from {module_path}."
        )
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return FunctionModule(module, entry_point, module_path)


def load_function_from_source(
    source_code: str, entry_point: str
) -> types.FunctionType:
    source_hash = hashlib.sha256(source_code.encode("utf-8")).hexdigest()
    FAST_KERNEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    module_path = FAST_KERNEL_CACHE_DIR / f"function_{source_hash}.py"
    if not module_path.exists():
        module_path.write_text(source_code, encoding="utf-8")
    return load_function_module(module_path, entry_point).get_function()


class UniqueNameGenerator:
    def __init__(self):
        self._existing_names: dict[str, int] = {}

    def generate(self, base_name: str) -> str:
        base_name = re.sub(r"\W", "_", base_name)
        if not base_name or base_name[0].isdigit():
            base_name = f"v_{base_name}"
        if keyword.iskeyword(base_name):
            base_name = f"{base_name}_"
        if base_name not in self._existing_names:
            self._existing_names[base_name] = 0
            return base_name
        self._existing_names[base_name] += 1
        return f"{base_name}_{self._existing_names[base_name]}"


class FastKernelRuntime:
    def __init__(
        self,
        compiled_function: types.FunctionType,
        kernel_ops: types.ModuleType,
        constants: list[object],
        source_code: str,
        lowered_program,
    ) -> None:
        self.compiled_function = compiled_function
        self.kernel_ops = kernel_ops
        self.constants = constants
        self.source_code = source_code
        self.lowered_program = lowered_program

    def __call__(self, inputs, parameters):
        return self.compiled_function(
            inputs, parameters, self.kernel_ops, self.constants
        )


class FastKernelRuntimeCompiler:
    ENTRY_POINT = "compiled_program"

    def __init__(self, runnable_program, place):
        self.runnable_program = runnable_program
        self.program_attr = runnable_program.program_attr
        self.place = place
        self.program = self.lower_to_kernel_program(
            runnable_program.forward_program, place
        )
        self.value_names = ValueDict()
        self.name_generator = UniqueNameGenerator()
        self.constants: list[object] = []
        self.used_kernel_names: set[str] = set()
        self.constant_bindings: list[tuple[str, int]] = []
        self.input_locals: dict[str, str] = {}
        self.param_locals: dict[str, str] = {}
        self.phi_kernel_op_info_cache: dict[int, dict[str, object]] = {}
        self.lines: list[str] = []
        self.indent_level = 0

    def lower_to_kernel_program(self, program, place):
        import paddle

        lower_fn = getattr(
            paddle.base.libpaddle.pir, "apply_pd_op_to_kernel_pass", None
        )
        if lower_fn is None:
            raise FastKernelRuntimeUnsupported(
                "apply_pd_op_to_kernel_pass is not available. Fast kernel "
                "runtime must codegen from pd_kernel dialect and does not "
                "fall back to run_program."
            )
        return lower_fn(program, place)

    def append_line(self, line: str = "") -> None:
        self.lines.append(f"{'    ' * self.indent_level}{line}")

    def lookup_value(self, value: Value) -> str:
        if is_fake_value(value):
            raise FastKernelRuntimeUnsupported(
                "Fast kernel runtime does not support fake output values yet."
            )
        if value not in self.value_names:
            raise FastKernelRuntimeUnsupported(
                f"Value {value} is used before it is produced."
            )
        return self.value_names[value]

    def bind_value(self, value: Value, preferred_name: str) -> str:
        name = self.name_generator.generate(preferred_name)
        self.value_names[value] = name
        return name

    def add_constant(self, value: object) -> str:
        const_name = self.name_generator.generate("const")
        self.constants.append(value)
        self.constant_bindings.append((const_name, len(self.constants) - 1))
        return const_name

    def emit_kernel_locals(self) -> None:
        for kernel_name in sorted(self.used_kernel_names):
            local_name = self.kernel_local_name(kernel_name)
            self.append_line(f"{local_name} = _kernel_ops.{kernel_name}")
        for const_name, const_idx in self.constant_bindings:
            self.append_line(f"{const_name} = _constants[{const_idx}]")
        if self.used_kernel_names or self.constant_bindings:
            self.append_line()

    def kernel_local_name(self, kernel_name: str) -> str:
        return f"_kernel_{kernel_name}"

    def emit_input_unpack(self) -> None:
        input_names = [
            self.name_generator.generate(name)
            for name in self.program_attr["fx_names"]
        ]
        param_names = [
            self.name_generator.generate(name)
            for name in self.program_attr["fp_names"]
        ]
        self.input_locals = dict(
            zip(self.program_attr["fx_names"], input_names)
        )
        self.param_locals = dict(
            zip(self.program_attr["fp_names"], param_names)
        )
        if len(input_names) == 1:
            self.append_line(f"{input_names[0]}, = inputs")
        elif len(input_names) > 1:
            self.append_line(f"{', '.join(input_names)} = inputs")
        if len(param_names) == 1:
            self.append_line(f"{param_names[0]}, = parameters")
        elif len(param_names) > 1:
            self.append_line(f"{', '.join(param_names)} = parameters")
        self.append_line()
        self.emit_block_kwargs()

    def emit_block_kwargs(self) -> None:
        for name, value in self.program.global_block().kwargs().items():
            if name in self.input_locals:
                self.value_names[value] = self.input_locals[name]
            elif name in self.param_locals:
                self.value_names[value] = self.param_locals[name]

    def emit_data_op(self, op) -> None:
        name = op.str_attr("name")
        if name not in self.program_attr["fx_names"]:
            raise FastKernelRuntimeUnsupported(
                f"pd_op.data '{name}' is not a SOT fast runtime input."
            )
        self.value_names[op.results()[0]] = self.input_locals[name]

    def emit_parameter_op(self, op) -> None:
        name = op.str_attr("parameter_name")
        if name not in self.program_attr["fp_names"]:
            raise FastKernelRuntimeUnsupported(
                f"builtin.parameter '{name}' is not a SOT fast runtime parameter."
            )
        self.value_names[op.results()[0]] = self.param_locals[name]

    def emit_shadow_feed_op(self, op) -> None:
        if op.num_operands() != 1 or len(op.results()) != 1:
            raise FastKernelRuntimeUnsupported(
                "pd_op.shadow_feed with multiple operands or results is not "
                "supported by SOT fast kernel runtime."
            )
        self.value_names[op.results()[0]] = self.lookup_value(
            op.operand_source(0)
        )

    def kernel_origin_op_name(self, op) -> str:
        if not op.name().startswith("pd_kernel."):
            return op.name()
        return self.get_phi_kernel_op_info(op)["op_name"]

    def get_phi_kernel_op_info(self, op) -> dict[str, object]:
        cache_key = id(op)
        if cache_key in self.phi_kernel_op_info_cache:
            return self.phi_kernel_op_info_cache[cache_key]
        info_fn = getattr(self.kernel_ops, "get_phi_kernel_op_info", None)
        if info_fn is None:
            raise FastKernelRuntimeUnsupported(
                "core.eager.kernel_ops.get_phi_kernel_op_info is not "
                "available. Fast kernel runtime must codegen from lowered "
                "pd_kernel metadata and does not fall back to run_program."
            )
        info = info_fn(op)
        self.phi_kernel_op_info_cache[cache_key] = info
        return info

    def get_op_input_names(self, op) -> list[str]:
        if op.name() == "pd_kernel.phi_kernel":
            return list(self.get_phi_kernel_op_info(op)["input_names"])
        return list(op.get_input_names())

    def emit_combine_op(self, op) -> None:
        values = [
            self.lookup_value(op.operand_source(i))
            for i in range(op.num_operands())
        ]
        if len(op.results()) != 1:
            raise FastKernelRuntimeUnsupported(
                "builtin.combine with multiple results is not supported."
            )
        result_name = self.bind_value(op.results()[0], "combine")
        self.append_line(f"{result_name} = [{', '.join(values)}]")

    def emit_split_op(self, op) -> None:
        if op.num_operands() != 1:
            raise FastKernelRuntimeUnsupported(
                "builtin.split with multiple operands is not supported."
            )
        source_name = self.lookup_value(op.operand_source(0))
        for idx, result in enumerate(op.results()):
            result_name = self.bind_value(result, "split")
            self.append_line(f"{result_name} = {source_name}[{idx}]")

    def lookup_operand(self, op, idx: int) -> str:
        source = op.operand_source(idx)
        if source is None or is_fake_value(source):
            return "None"
        return self.lookup_value(source)

    def is_foldable_full_int_array(self, op) -> bool:
        if (
            self.kernel_origin_op_name(op) != "pd_op.full_int_array"
            or len(op.results()) != 1
        ):
            return False
        value = op.results()[0]
        if value.use_empty():
            return True
        for used_op in value.all_used_ops():
            if not self.kernel_origin_op_name(used_op).startswith("pd_op."):
                return False
            try:
                input_names = self.get_op_input_names(used_op)
            except ValueError:
                return False
            matched_input = False
            for idx, input_name in enumerate(input_names):
                source = used_op.operand_source(idx)
                if source is not None and source.is_same(value):
                    matched_input = True
                    if input_name not in INT_ARRAY_INPUT_NAMES:
                        return False
            if not matched_input:
                return False
        return True

    def emit_full_int_array_op(self, op) -> None:
        if not self.is_foldable_full_int_array(op):
            if op.name() == "pd_kernel.phi_kernel":
                self.emit_phi_kernel_direct_api(op)
            else:
                self.emit_kernel_op(op)
            return
        if op.results()[0].use_empty():
            return
        if op.name() == "pd_kernel.phi_kernel":
            value = self.get_phi_kernel_op_info(op)["attrs"]["value"]
        else:
            value = op.attrs()["value"]
        const_name = self.add_constant(list(value))
        self.value_names[op.results()[0]] = const_name

    def emit_direct_kernel_api(
        self,
        op,
        kernel_name: str,
        input_names: list[str],
        attr_names: list[str],
        attrs: dict[str, object],
    ) -> None:
        if not hasattr(self.kernel_ops, kernel_name):
            raise FastKernelRuntimeUnsupported(
                f"Direct kernel pybind API '{kernel_name}' is not generated. "
                "Fast kernel runtime does not fall back to run_program."
            )
        self.used_kernel_names.add(kernel_name)

        input_args = []
        for idx, input_name in enumerate(input_names):
            input_args.append((input_name, self.lookup_operand(op, idx)))

        attr_args = [
            (attr_name, self.add_constant(attrs[attr_name]))
            for attr_name in attr_names
            if attr_name in attrs
        ]
        available_args = dict(input_args + attr_args)
        arg_names = kernel_arg_name_map().get(
            kernel_name,
            [name for name, _ in input_args] + [name for name, _ in attr_args],
        )
        args = []
        for arg_name in arg_names:
            if arg_name not in available_args:
                raise FastKernelRuntimeUnsupported(
                    f"Argument '{arg_name}' of '{kernel_name}' is not "
                    "available in PIR operation inputs or attrs."
                )
            args.append(available_args[arg_name])

        call_expr = f"{self.kernel_local_name(kernel_name)}({', '.join(args)})"
        result_names = [
            self.bind_value(result, kernel_name) for result in op.results()
        ]
        if len(result_names) == 0:
            self.append_line(call_expr)
        elif len(result_names) == 1:
            self.append_line(f"{result_names[0]} = {call_expr}")
        else:
            self.append_line(f"{', '.join(result_names)} = {call_expr}")

    def emit_kernel_op(self, op) -> None:
        op_name = op.name()
        if not op_name.startswith("pd_op."):
            raise FastKernelRuntimeUnsupported(
                f"Operation {op_name} is not supported by fast kernel runtime."
            )
        self.emit_direct_kernel_api(
            op,
            op_name.removeprefix("pd_op."),
            list(op.get_input_names()),
            list(op.get_attr_names()),
            op.attrs(),
        )

    def emit_phi_kernel_direct_api(self, op) -> None:
        info = self.get_phi_kernel_op_info(op)
        origin_op_name = info["op_name"]
        if not origin_op_name.startswith("pd_op."):
            raise FastKernelRuntimeUnsupported(
                f"Lowered phi kernel origin op '{origin_op_name}' is not "
                "supported by fast kernel runtime."
            )
        kernel_name = origin_op_name.removeprefix("pd_op.")
        lowered_kernel_name = info["kernel_name"]
        if not hasattr(self.kernel_ops, kernel_name) and hasattr(
            self.kernel_ops, lowered_kernel_name
        ):
            kernel_name = lowered_kernel_name
        self.emit_direct_kernel_api(
            op,
            kernel_name,
            list(info["input_names"]),
            list(info["attr_names"]),
            info["attrs"],
        )

    def emit_phi_kernel_op(self, op) -> None:
        origin_op_name = self.kernel_origin_op_name(op)
        if origin_op_name == "pd_op.data":
            self.emit_data_op(op)
            return
        if origin_op_name == "pd_op.shadow_feed":
            self.emit_shadow_feed_op(op)
            return
        if origin_op_name == "builtin.parameter":
            self.emit_parameter_op(op)
            return
        if origin_op_name == "pd_op.full_int_array":
            self.emit_full_int_array_op(op)
            return
        self.emit_phi_kernel_direct_api(op)

    def emit_cinn_jit_kernel_op(self, op) -> None:
        kernel_name = "run_cinn_jit_kernel"
        if not hasattr(self.kernel_ops, kernel_name):
            raise FastKernelRuntimeUnsupported(
                "core.eager.kernel_ops.run_cinn_jit_kernel is not available. "
                "Fast kernel runtime does not fall back to run_program."
            )
        self.used_kernel_names.add(kernel_name)
        op_name = self.add_constant(op)
        inputs = [
            self.lookup_value(op.operand_source(i))
            for i in range(op.num_operands())
        ]
        call_expr = (
            f"{self.kernel_local_name(kernel_name)}"
            f"({op_name}, [{', '.join(inputs)}])"
        )
        result_names = [
            self.bind_value(result, "cinn_jit_kernel")
            for result in op.results()
        ]
        if len(result_names) == 0:
            self.append_line(call_expr)
        elif len(result_names) == 1:
            self.append_line(f"{result_names[0]} = {call_expr}")
        else:
            self.append_line(f"{', '.join(result_names)} = {call_expr}")

    def emit_return(self) -> None:
        return_values = []
        output_names = set(self.program_attr["fo_names"])
        for op in self.program.global_block().ops:
            if (
                op.name() == "builtin.shadow_output"
                and op.str_attr("output_name") in output_names
            ):
                return_values.append(op.operand_source(0))
        if not return_values:
            raise FastKernelRuntimeUnsupported(
                "Lowered fast kernel program has no forward output "
                "builtin.shadow_output. Fast kernel runtime does not fall "
                "back to run_program."
            )
        output_exprs = [self.lookup_value(value) for value in return_values]
        if not output_exprs:
            self.append_line("return ()")
        elif len(output_exprs) == 1:
            self.append_line(f"return ({output_exprs[0]},)")
        else:
            self.append_line(f"return ({', '.join(output_exprs)})")

    def emit_body_ops(self) -> None:
        for op in self.program.global_block().ops:
            op_name = op.name()
            if op_name == "pd_op.data":
                self.emit_data_op(op)
            elif op_name == "builtin.parameter":
                self.emit_parameter_op(op)
            elif op_name == "builtin.combine":
                self.emit_combine_op(op)
            elif op_name == "builtin.split":
                self.emit_split_op(op)
            elif op_name == "builtin.shadow_output":
                continue
            elif op_name == "pd_op.full_int_array":
                self.emit_full_int_array_op(op)
            elif op_name == "pd_op.shadow_feed":
                self.emit_shadow_feed_op(op)
            elif op_name == "pd_kernel.phi_kernel":
                self.emit_phi_kernel_op(op)
            elif op_name == "pd_kernel.legacy_kernel":
                raise FastKernelRuntimeUnsupported(
                    "pd_kernel.legacy_kernel is not supported by SOT fast "
                    "kernel runtime yet. It does not fall back to run_program."
                )
            elif op_name == "cinn_runtime.jit_kernel":
                self.emit_cinn_jit_kernel_op(op)
            else:
                self.emit_kernel_op(op)

    def compile(self) -> FastKernelRuntime:
        from paddle.base import core

        if not hasattr(core.eager, "kernel_ops"):
            raise FastKernelRuntimeUnsupported(
                "core.eager.kernel_ops is not available. Rebuild libpaddle "
                "with direct kernel pybind generation enabled."
            )
        self.kernel_ops = core.eager.kernel_ops
        self.append_line(
            f"def {self.ENTRY_POINT}("
            "inputs, parameters, _kernel_ops, _constants):"
        )
        self.indent_level += 1
        self.emit_input_unpack()
        self.emit_body_ops()
        self.emit_return()

        body_lines = self.lines[1:]
        self.lines = [self.lines[0]]
        self.emit_kernel_locals()
        self.lines.extend(body_lines)
        self.indent_level -= 1

        source_code = "\n".join(self.lines) + "\n"
        compiled_function = load_function_from_source(
            source_code, self.ENTRY_POINT
        )
        return FastKernelRuntime(
            compiled_function,
            self.kernel_ops,
            self.constants,
            source_code,
            self.program,
        )


def compile_fast_kernel_runtime(runnable_program, place) -> FastKernelRuntime:
    return FastKernelRuntimeCompiler(runnable_program, place).compile()
