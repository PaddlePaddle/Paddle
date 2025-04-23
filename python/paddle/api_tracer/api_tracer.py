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
from typing import Any, Dict, List, Tuple, Union


class HookAPIMap:
    """Storage for original API implementations"""
    pass


class ConfigDump:
    def __init__(self):
        self.file = None
        self.type_mapping = {
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

    def open_file(self, path: str) -> None:
        """Open file for writing API traces"""
        self.file = open(path, "a+")

    def close_file(self) -> None:
        """Close the output file"""
        if self.file:
            self.file.close()

    def dump_config(self, api: str, input_args: tuple, input_kwargs: dict, outputs: Any) -> None:
        """Dump API call configuration to file"""
        if not self.file:
            return

        try:
            args_str = self._format_args(api, input_args)
            kwargs_str = self._format_kwargs(api, input_kwargs)
            
            if args_str is None or kwargs_str is None:
                return

            result = f"{api}({args_str}{kwargs_str})"
            self.file.write(result + "\n")
            self.file.flush()
        except Exception as e:
            print(f"[api_tracer error] dump_config failed for {api}: {str(e)}")

    def _format_args(self, api: str, args: tuple) -> str:
        """Format positional arguments"""
        parts = []
        for value in args:
            tmp = self.dump_item_str(api, value)
            if tmp == "":
                return None
            parts.append(tmp)
        return ", ".join(parts) + (", " if parts and parts[-1] else "")

    def _format_kwargs(self, api: str, kwargs: dict) -> str:
        """Format keyword arguments"""
        parts = []
        for key, value in kwargs.items():
            tmp = self.dump_item_str(api, value)
            if tmp == "":
                return None
            parts.append(f"{key}={tmp}")
        return ", ".join(parts)

    def dump_item_str(self, api: str, item: Any) -> str:
        """Convert various types to their string representation"""
        try:
            # Convert numpy types to Python built-in types
            for numpy_type, builtin_type in self.type_mapping.items():
                if isinstance(item, numpy_type):
                    item = builtin_type(item)
                    break

            if isinstance(item, paddle.Tensor):
                return f'Tensor({item.shape},"{str(item.dtype)[7:]}")'
            elif isinstance(item, (paddle.base.core.DataType, paddle.base.core.VarDesc.VarType)):
                return f"{type(item).__name__}({str(item)[7:]})"
            elif isinstance(item, (list, tuple)):
                container_type = "list" if isinstance(item, list) else "tuple"
                items_str = ",".join(self.dump_item_str(api, sub_item) for sub_item in item)
                return f"{container_type}[{items_str}]" if isinstance(item, list) else f"{container_type}({items_str})"
            elif isinstance(item, slice):
                start = self._get_slice_component(item.start)
                stop = self._get_slice_component(item.stop)
                step = self._get_slice_component(item.step)
                return f"slice({start},{stop},{step})"
            elif isinstance(item, complex):
                return f"complex({self.dump_item_str(api, item.real)},{self.dump_item_str(api, item.imag)})"
            elif item is None:
                return "None"
            elif isinstance(item, (paddle.base.Variable, paddle.base.libpaddle.pir.Value)):
                return ""
            elif item in (math.inf, -math.inf, math.nan, -math.nan):
                return str(item)
            elif isinstance(item, (bool, int, float)):
                return str(item)
            elif isinstance(item, str):
                return f'"{item}"'
            elif isinstance(item, type):
                type_str = str(item)
                return f"type({type_str[type_str.index("'")+1:type_str.rindex("'")]})"
            elif isinstance(item, np.ndarray):
                return str(item)[1:-1]
            elif isinstance(item, np.dtype):
                return f"Dtype({item})"
            elif item is Ellipsis:
                return "Ellipsis"
            else:
                print(f"[api_tracer warning] Unhandled type in {api}: {type(item)}")
                return ""
        except Exception as e:
            print(f"[api_tracer error] dump_item_str failed for {api}: {str(e)}")
            return ""

    def _get_slice_component(self, component: Any) -> str:
        """Helper method to handle slice components"""
        if isinstance(component, paddle.Tensor):
            return str(int(component.numpy()))
        return str(component) if component is not None else "None"


config_dump = ConfigDump()


class APITemplate:
    def __init__(self, api_name: str):
        self.api_name = api_name

    def __call__(self, *args, **kwargs) -> Any:
        """Wrapper for API calls that logs the configuration"""
        try:
            original_func = getattr(HookAPIMap, self.api_name)
            output = original_func(*args, **kwargs)
            config_dump.dump_config(self.api_name, args, kwargs, output)
            return output
        except Exception as e:
            print(f"[api_tracer error] API call failed for {self.api_name}: {str(e)}")
            raise


def wrapped_api(api_name: str):
    """Factory function to create API wrappers"""
    def api_template(*args, **kwargs):
        return APITemplate(api_name)(*args, **kwargs)
    return api_template


def start_api_tracer(api_path: str, save_config_path: str) -> None:
    """Initialize API tracing by wrapping specified APIs"""
    try:
        print(f"Paddle version: {paddle.__version__}")
        
        with open(api_path, "r") as f:
            apis = yaml.safe_load(f)
            sample_apis = apis.get("apis", [])
        
        for api in sample_apis:
            try:
                parent_package, method_name = api.rsplit(".", maxsplit=1)
                module = eval(parent_package)
                original_func = getattr(module, method_name)
                
                setattr(HookAPIMap, api, original_func)
                setattr(module, method_name, wrapped_api(api))
            except Exception as e:
                print(f"[api_tracer warning] Failed to wrap {api}: {str(e)}")
        
        config_dump.open_file(save_config_path)
    except Exception as e:
        print(f"[api_tracer error] Initialization failed: {str(e)}")
        raise


def stop_api_tracer() -> None:
    """Clean up and stop API tracing"""
    config_dump.close_file()