# Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

from cinn.core_api.runtime import (  # noqa: F401
    VoidPointer,
    cinn_arm_device,
    cinn_bool_t,
    cinn_buffer_copy,
    cinn_buffer_copy_to_device,
    cinn_buffer_copy_to_host,
    cinn_buffer_free,
    cinn_buffer_get_data_const_handle,
    cinn_buffer_get_data_handle,
    cinn_buffer_kind_t,
    cinn_buffer_load_float32,
    cinn_buffer_load_float64,
    cinn_buffer_malloc,
    cinn_buffer_on_device,
    cinn_buffer_on_host,
    cinn_buffer_t,
    cinn_device_interface_t,
    cinn_device_kind_t,
    cinn_device_release,
    cinn_device_sync,
    cinn_float32_t,
    cinn_float64_t,
    cinn_int8_t,
    cinn_int32_t,
    cinn_int64_t,
    cinn_opencl_device,
    cinn_pod_value_t,
    cinn_pod_value_to_buffer_p,
    cinn_pod_value_to_double,
    cinn_pod_value_to_float,
    cinn_pod_value_to_int8,
    cinn_pod_value_to_int32,
    cinn_pod_value_to_int64,
    cinn_pod_value_to_void_p,
    cinn_type_code_t,
    cinn_type_float,
    cinn_type_handle,
    cinn_type_int,
    cinn_type_t,
    cinn_type_uint,
    cinn_type_unk,
    cinn_uint32_t,
    cinn_uint64_t,
    cinn_unk_device,
    cinn_unk_t,
    cinn_value_t,
    cinn_x86_device,
    cinn_x86_device_interface,
    clear_seed,
    nullptr,
    seed,
    set_cinn_cudnn_deterministic,
)
