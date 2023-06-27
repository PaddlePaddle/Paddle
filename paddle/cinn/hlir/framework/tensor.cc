// Copyright (c) 2021 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/cinn/hlir/framework/tensor.h"

#include "paddle/cinn/runtime/cinn_runtime.h"

namespace cinn {
namespace hlir {
namespace framework {

void _Tensor_::set_type(Type type) {
  type_ = type;
  if (type.is_bool()) {
    buffer_->data()->type = cinn_bool_t();
  } else if (type.is_int(8)) {
    buffer_->data()->type = cinn_int8_t();
  } else if (type.is_int(16)) {
    buffer_->data()->type = cinn_int16_t();
  } else if (type.is_int(32)) {
    buffer_->data()->type = cinn_int32_t();
  } else if (type.is_int(64)) {
    buffer_->data()->type = cinn_int64_t();
  } else if (type.is_uint(8)) {
    buffer_->data()->type = cinn_uint8_t();
  } else if (type.is_uint(16)) {
    buffer_->data()->type = cinn_uint16_t();
  } else if (type.is_uint(32)) {
    buffer_->data()->type = cinn_uint32_t();
  } else if (type.is_uint(64)) {
    buffer_->data()->type = cinn_uint64_t();
  } else if (type.is_float(32)) {
    buffer_->data()->type = cinn_float32_t();
  } else if (type.is_float(64)) {
    buffer_->data()->type = cinn_float64_t();
  } else if (type.is_bfloat16()) {
    buffer_->data()->type = cinn_bfloat16_t();
  } else if (type.is_float16()) {
    buffer_->data()->type = cinn_float16_t();
  } else {
    buffer_->data()->type = cinn_unk_t();
  }
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
