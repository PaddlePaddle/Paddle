// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_type.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/utils/utils.h"
#include "paddle/ir/core/value.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/extended_tensor.h"
#include "paddle/phi/core/utils/data_type.h"

namespace paddle {
namespace primitive {

class LazyTensor : public phi::ExtendedTensor,
                   public phi::TypeInfoTraits<phi::TensorBase, LazyTensor> {
 public:
  explicit LazyTensor(ir::Value value)
      : value_(value),
        dims_(value.type().dyn_cast<dialect::DenseTensorType>().dims()) {}

  static const char* name() { return "LazyTensor"; }

  const phi::DDim& dims() const override { return dims_; }

  int64_t numel() const override { return product(dims()); }

  DataType dtype() const override {
    return paddle::dialect::TransToPhiDataType(
        value_.type().dyn_cast<paddle::dialect::DenseTensorType>().dtype());
  }

  ir::Value getValue() const { return value_; }

  const phi::Place& place() const override { return place_; }

  bool initialized() const override { return value_.impl() != nullptr; }

 private:
  ir::Value value_;
  mutable phi::DDim dims_;
  phi::Place place_;
};

}  // namespace primitive
}  // namespace paddle
