// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/common/macros.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/ddim.h"

namespace phi {
class PADDLE_API NativeMetaTensor {
 public:
  NativeMetaTensor() = default;
  NativeMetaTensor(phi::DataType dtype, phi::DDim dims)
      : dims_(dims), dtype_(dtype) {}
  DDim dims() const;
  DataType dtype() const;
  void set_dims(const DDim& dims);
  void set_dtype(DataType dtype);

 private:
  phi::DDim dims_;
  phi::DataType dtype_{phi::DataType::FLOAT32};
};
}  // namespace phi
