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

#include "paddle/cinn/common/test_helper.h"

namespace cinn {
namespace common {

cinn_buffer_t* BufferBuilder::Build() {
  cinn_type_t cinn_type;
  if (type_ == type_of<float>()) {
    cinn_type = cinn_float32_t();
  } else if (type_ == type_of<double>()) {
    cinn_type = cinn_float64_t();
  } else if (type_ == type_of<int8_t>()) {
    cinn_type = cinn_int8_t();
  } else if (type_ == type_of<int32_t>()) {
    cinn_type = cinn_int32_t();
  } else if (type_ == type_of<int64_t>()) {
    cinn_type = cinn_int64_t();
  } else if (type_ == type_of<bool>()) {
    cinn_type = cinn_bool_t();
  } else {
    CINN_NOT_IMPLEMENTED
  }

  auto* buffer = cinn_buffer_t::new_(
      cinn_device_kind_t::cinn_x86_device, cinn_type, shape_, align_);

  cinn_buffer_malloc(nullptr, buffer);

  switch (init_type_) {
    case InitType::kZero:
      memset(buffer->memory, 0, buffer->memory_size);
      break;

    case InitType::kRandom:
      if (type_ == type_of<float>()) {
        RandomFloat<float>(buffer->memory, buffer->num_elements());
      } else if (type_ == type_of<double>()) {
        RandomFloat<double>(buffer->memory, buffer->num_elements());
      } else if (type_ == type_of<bool>()) {
        RandomInt<int8_t>(buffer->memory, buffer->num_elements());
      } else if (type_ == type_of<int8_t>()) {
        RandomInt<int8_t>(buffer->memory, buffer->num_elements());
      } else if (type_ == type_of<int32_t>()) {
        RandomInt<int32_t>(buffer->memory, buffer->num_elements());
      } else if (type_ == type_of<int64_t>()) {
        RandomInt<int64_t>(buffer->memory, buffer->num_elements());
      }
      break;

    case InitType::kSetValue:
      if (type_ == type_of<int>()) {
        SetVal<int>(buffer->memory, buffer->num_elements(), init_val_);
      } else if (type_ == type_of<int8_t>()) {
        SetVal<int8_t>(buffer->memory, buffer->num_elements(), init_val_);
      } else if (type_ == type_of<float>()) {
        SetVal<float>(buffer->memory, buffer->num_elements(), init_val_);
      } else {
        CINN_NOT_IMPLEMENTED
      }
      break;
  }

  return buffer;
}

}  // namespace common
}  // namespace cinn
