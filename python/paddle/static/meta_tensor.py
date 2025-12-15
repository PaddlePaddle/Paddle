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


from ..base.libpaddle import NativeMetaTensor

# class MetaTensor:
#     def __init__(self, *, shape=None, dtype=None):
#         self.native_meta_tensor = NativeMetaTensor()
#         if shape is not None:
#             self.native_meta_tensor.set_shape(shape)
#         if dtype is not None:
#             self.native_meta_tensor.set_dtype(dtype)

#     def set_shape(self, shape):
#         self.native_meta_tensor.set_shape(shape)

#     @property
#     def shape(self):
#         return self.native_meta_tensor.shape

#     def set_dtype(self, dtype):
#         self.native_meta_tensor.set_dtype(dtype)

#     @property
#     def dtype(self):
#         return self.native_meta_tensor.dtype

#     def __eq__(self, other):
#         return (
#             self.native_meta_tensor.dtype == other.native_meta_tensor.dtype
#             and self.native_meta_tensor.shape == other.native_meta_tensor.shape
#         )


MetaTensor = NativeMetaTensor

# def map_type(fn, type_, structure):
#     map_fn = lambda v: fn(v) if isinstance(v, type_) else v
#     return map_structure(map_fn, structure)

# def wrap_infer_meta(fn):
#     @wraps(fn)
#     def infer_meta(*args, **kwargs):
#         args, kwargs = map_type((args, kwargs))


# def MetaTensorWrapper(fn):
#     @wraps(fn)
#     def wrapper(*args, **kwargs):
#         # IrTensor -> MetaTensor
#         new_args = list(args)
#         for i, arg in enumerate(args):
#             if isinstance(arg, IrTensor):
#                 new_args[i] = MetaTensor(ir_tensor=arg)
#         for key, value in kwargs.items():
#             if isinstance(value, IrTensor):
#                 kwargs[key] = MetaTensor(ir_tensor=value)
#         outputs = fn(*new_args, **kwargs)
#         if isinstance(outputs, (list, tuple)):
#             return [output.ir_tensor for output in outputs]
#         return outputs.ir_tensor

#     return wrapper
