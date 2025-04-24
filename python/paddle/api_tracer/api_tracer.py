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

import math
import numpy as np
import yaml
import paddle
from types import ModuleType
 
 
class HookAPIMap:
    pass
 
 
class ConfigDump:
    def __init__(self):
        self.file = None
        self._type_mapping = {
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
        }
 
    def open_file(self, path):
        self.file = open(path, "a+")
 
    def close_file(self):
        if self.file:
            self.file.close()
 
    def dump_config(self, api, input_args, input_kwargs, outputs):
        try:
            args_str = self._serialize_args(input_args)
            kwargs_str = self._serialize_kwargs(input_kwargs)
            output_str = self._serialize_output(outputs)
            
            config_line = f"{api}({args_str}{kwargs_str}) -> {output_str}\n"
            self.file.write(config_line)
            self.file.flush()
        except Exception as e:
            print(f"[api_tracer error] : config_dump.dump_config {api}, {str(e)}")
 
    def _serialize_args(self, args):
        return ", ".join(self._serialize_item(arg) for arg in args)
 
    def _serialize_kwargs(self, kwargs):
        return ", ".join(f"{k}={self._serialize_item(v)}" for k, v in kwargs.items())
 
    def _serialize_output(self, outputs):
        if isinstance(outputs, (list, tuple)):
            return "[" + ", ".join(self._serialize_item(out) for out in outputs) + "]"
        return self._serialize_item(outputs)
 
    def _serialize_item(self, item):
        for np_type, py_type in self._type_mapping.items():
            if isinstance(item, np_type):
                return self._serialize_item(py_type(item))
 
        if isinstance(item, paddle.Tensor):
            shape = str(item.shape) if item.shape else "unknown"
            dtype = str(item.dtype)[7:] if item.dtype else "unknown"
            return f"Tensor({shape}, '{dtype}')"
        elif isinstance(item, paddle.core.VarDesc.VarType):
            return f"VarType({item.name})"
        elif isinstance(item, paddle.core.DataType):
            return f"Dtype({item.name})"
 
        if isinstance(item, slice):
            start = self._safe_str(item.start)
            stop = self._safe_str(item.stop)
            step = self._safe_str(item.step)
            return f"slice({start}, {stop}, {step})"
        elif isinstance(item, complex):
            return f"complex({self._serialize_item(item.real)}, {self._serialize_item(item.imag)})"
        elif item is None:
            return "None"
        elif isinstance(item, (math.inf, -math.inf, math.nan, -math.nan)):
            return str(item)
        elif isinstance(item, (bool, int, float)):
            return str(item)
        elif isinstance(item, str):
            return f'"{item}"'
        elif isinstance(item, type):
            return f"type({item.__name__})"
        elif isinstance(item, np.ndarray):
            return "ndarray(" + str(item.shape) + ")"
        elif isinstance(item, np.dtype):
            return f"Dtype({item.name})"
        elif item is Ellipsis:
            return "Ellipsis"
        elif isinstance(item, (list, tuple)):
            return self._serialize_sequence(item)
        else:
            print(f"[api_tracer error] : Unsupported type {type(item)} for item {item}")
            return "UNKNOWN_TYPE"
 
    def _serialize_sequence(self, seq):
        return (
            "list[" + ", ".join(self._serialize_item(i) for i in seq) + "]"
            if isinstance(seq, list)
            else f"tuple({', '.join(self._serialize_item(i) for i in seq)})"
        )
 
    def _safe_str(self, value):
        return str(value) if value is not None else "None"
 
 
config_dump = ConfigDump()
 
 
class APITemplate:
    def __init__(self, api_name):
        self.api_name = api_name
 
    def __call__(self, *args, **kwargs):
        original_func = getattr(HookAPIMap, self.api_name)
        output = original_func(*args, **kwargs)
        try:
            config_dump.dump_config(self.api_name, args, kwargs, output)
        except Exception as err:
            print(f"[api_tracer error] : config_dump.dump_config {self.api_name}, {str(err)}")
        return output
 
 
def wrapped_api(api_name):
    def api_template(*args, **kwargs):
        return APITemplate(api_name)(*args, **kwargs)
    return api_template
 
 
def start_api_tracer(api_path, save_config_path):
    print(paddle.__version__)
    
    with open(api_path, "r") as f:
        apis = yaml.safe_load(f)
        sample_apis = apis.get("apis", [])
 
    for api in sample_apis:
        try:
            module_name, method_name = api.rsplit(".", maxsplit=1)
            module = __import__(module_name, fromlist=[method_name])
            original_method = getattr(module, method_name)
            
            setattr(HookAPIMap, api, original_method)
            setattr(module, method_name, wrapped_api(api))
        except Exception as err:
            print(f"[api_tracer error] : start_api_tracer {api}, {str(err)}")
 
    config_dump.open_file(save_config_path)
    return config_dump  
