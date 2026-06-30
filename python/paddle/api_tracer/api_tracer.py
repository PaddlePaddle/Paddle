# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import inspect
import math
import threading

import numpy as np
import yaml

import paddle

_trace_guard = threading.local()
_originals = {}  # {api_path: (parent_obj, method_name, original_fn)}
_hooked_apis = []


class HookAPIMap:
    pass


class ConfigDump:
    def __init__(self):
        pass

    def open_file(self, path):
        self.file = open(path, "a+")

    def dump_config(self, api, input_args, input_kwargs, outputs):
        result = api + "("
        for value in input_args:
            tmp = self.dump_item_str(api, value)
            if tmp == "":
                return
            result = result + tmp + ", "
        for key, value in input_kwargs.items():
            tmp = self.dump_item_str(api, value)
            if tmp == "":
                return
            result = result + key + "=" + tmp + ", "

        result = result + ")"
        # self.file.write(") -> ")
        # if isinstance(outputs, (list, tuple)):
        #     for output in outputs:
        #         self.file.write(self.dump_item_str(api, output) + ", ")
        # else:
        #     self.file.write(self.dump_item_str(api, outputs) + ", ")

        self.file.write(result)
        self.file.write("\n")
        self.file.flush()

    def dump_item_str(self, api, item):
        type_mapping = {
            np.int16: int,
            np.int32: int,
            np.int64: int,
            np.float16: float,
            np.float32: float,
            np.float64: float,
            np.integer: int,
            np.floating: float,
            np.bool_: bool,
            np.complexfloating: complex,
            np.str_: str,
            np.bytes_: bytes,
            # np.unicode_: str,
        }
        for numpy_type, builtin_type in type_mapping.items():
            if isinstance(item, numpy_type):
                item = builtin_type(item)
                break

        if isinstance(item, paddle.Tensor):
            result = (
                "Tensor(" + str(item.shape) + ',"' + str(item.dtype)[7:] + '"'
            )
            if item.place.is_cpu_place():
                result = result + ",place=" + str(item.place)
            if not item.is_contiguous():
                result = (
                    result + ",is_contiguous=False,strides=" + str(item.strides)
                )
            return result + ")"
        elif isinstance(item, paddle.base.core.DataType):
            return "Dtype(" + str(item)[7:] + ")"
        elif isinstance(item, paddle.base.core.VarDesc.VarType):
            return "VarType(" + str(item)[7:] + ")"
        elif isinstance(item, paddle.base.libpaddle.Place):
            return str(item)
        elif isinstance(item, list):
            result = "list["
            for sub_item in item:
                tmp = self.dump_item_str(api, sub_item)
                if tmp == "":
                    return ""
                result = result + tmp + ","
            result = result + "]"
            return result
        elif isinstance(item, tuple):
            result = "tuple("
            for sub_item in item:
                tmp = self.dump_item_str(api, sub_item)
                if tmp == "":
                    return ""
                result = result + tmp + ","
            result = result + ")"
            return result
        elif isinstance(item, slice):
            start_str = (
                str(int(item.start.numpy()))
                if isinstance(item.start, paddle.Tensor)
                else str(item.start)
            )
            stop_str = (
                str(int(item.stop.numpy()))
                if isinstance(item.stop, paddle.Tensor)
                else str(item.stop)
            )
            step_str = (
                str(int(item.step.numpy()))
                if isinstance(item.step, paddle.Tensor)
                else str(item.step)
            )
            return "slice(" + start_str + "," + stop_str + "," + step_str + ")"
        elif isinstance(item, complex):
            return (
                "complex("
                + self.dump_item_str(api, item.real)
                + ","
                + self.dump_item_str(api, item.imag)
                + ")"
            )
        elif item is None:
            return "None"
        elif isinstance(
            item, (paddle.base.Variable, paddle.base.libpaddle.pir.Value)
        ):
            return ""
        elif item == math.inf:
            return "math.inf"
        elif item == -math.inf:
            return "-math.inf"
        elif item == math.nan:
            return "math.nan"
        elif item == -math.nan:
            return "-math.nan"
        elif isinstance(item, (bool, int, float)):
            return str(item)
        elif isinstance(item, str):
            return '"' + item + '"'
        elif isinstance(item, type):
            return (
                "type("
                + str(item)[str(item).index("'") + 1 : str(item).rindex("'")]
                + ")"
            )
        elif isinstance(item, np.ndarray):
            return str(item)[1:-1]
        elif isinstance(item, np.dtype):
            return "Dtype(" + str(item) + ")"
        elif item == Ellipsis:
            return "Ellipsis"
        else:
            print(
                "[api_tracer error] : dump_item_str ",
                api,
                ", item = ",
                item,
                ", type(item) = ",
                type(item),
            )
            return ""


config_dump = ConfigDump()


class APITemplate:
    def __init__(self, api_name):
        self.api_name = api_name

    def __call__(self, *args, **kwargs):
        if getattr(_trace_guard, 'in_hook', False):
            return getattr(HookAPIMap, self.api_name)(*args, **kwargs)
        _trace_guard.in_hook = True
        try:
            output = getattr(HookAPIMap, self.api_name)(*args, **kwargs)
            try:
                config_dump.dump_config(self.api_name, args, kwargs, output)
            except Exception as err:
                print(
                    "[api_tracer error] : config_dump.dump_config ",
                    self.api_name,
                    str(err),
                )
            return output
        finally:
            _trace_guard.in_hook = False


def wrapped_api(api_name):
    def api_template(*args, **kwargs):
        return APITemplate(api_name)(*args, **kwargs)

    return api_template


def expand_wildcard(api_pattern):
    if not api_pattern.endswith('.*'):
        return [api_pattern]
    module_path = api_pattern[:-2]
    try:
        module = eval(module_path)
    except Exception as e:
        print(f"[api_tracer error] : expand_wildcard {api_pattern}, {e}")
        return []
    apis = []
    for name in dir(module):
        if module_path.endswith('_C_ops'):
            if name.startswith('__'):
                continue
        elif name.startswith('_'):
            continue
        try:
            attr = getattr(module, name)
            if (
                callable(attr)
                and not inspect.isclass(attr)
                and not inspect.ismodule(attr)
            ):
                apis.append(f"{module_path}.{name}")
        except Exception:
            pass
    return apis


def start_api_tracer(api_path, save_config_path):
    print(paddle.__version__)
    with open(api_path, "r") as f:
        raw_apis = yaml.safe_load(f).get("apis") or []

    sample_apis = []
    for api in raw_apis:
        sample_apis.extend(expand_wildcard(api))
    sample_apis = list(dict.fromkeys(sample_apis))
    print(f"[api_tracer] Expanded to {len(sample_apis)} APIs")

    for api in sample_apis:
        parent_package, method_name = api.rsplit(".", maxsplit=1)
        try:
            parent = eval(parent_package)
            original = getattr(parent, method_name)
            _originals[api] = (parent, method_name, original)
            setattr(HookAPIMap, api, original)
            setattr(parent, method_name, wrapped_api(api))
            _hooked_apis.append(api)
        except Exception as err:
            print(
                "[api_tracer error] : start_api_tracer ",
                api,
                str(err),
            )

    config_dump.open_file(save_config_path)


def stop_api_tracer():
    for api in _hooked_apis:
        entry = _originals.pop(api, None)
        if entry:
            parent, method_name, original = entry
            try:
                setattr(parent, method_name, original)
            except Exception as err:
                print(f"[api_tracer error] : stop_api_tracer {api} {err}")
    _hooked_apis.clear()
    if (
        hasattr(config_dump, 'file')
        and config_dump.file
        and not config_dump.file.closed
    ):
        config_dump.file.flush()
        config_dump.file.close()
    print("[api_tracer] Stopped, all hooks removed.")
