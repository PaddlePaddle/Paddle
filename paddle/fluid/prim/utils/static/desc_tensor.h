// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/common/ddim.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/phi/core/extended_tensor.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/utils/any.h"

namespace paddle {
namespace prim {

class DescTensor : public phi::ExtendedTensor,
                   public phi::TypeInfoTraits<phi::TensorBase, DescTensor> {
 public:
  explicit DescTensor(framework::VarDesc* desc)
      : desc_ptr_(desc), dims_(common::make_ddim(desc->GetShape())) {}
  static const char* name() { return "DescTensor"; }

  std::string Name() const { return desc_ptr_->Name(); }

  std::vector<int64_t> shape() const { return desc_ptr_->GetShape(); }

  const phi::DDim& dims() const override {
    dims_ = common::make_ddim(desc_ptr_->GetShape());
    return dims_;
  }

  int64_t numel() const override { return product(dims()); }

  DataType dtype() const override {
    return phi::TransToPhiDataType(desc_ptr_->GetDataType());
  }

  framework::VarDesc* get_ptr() { return desc_ptr_; }

  const phi::Place& place() const override { return place_; }

  bool initialized() const override { return desc_ptr_ != nullptr; }

  // TODO(jiabin): override more operators here.

 private:
  // VarDesc's lifetime is held by block and it's program, so we just conceal
  // its funcs instead of its life.
  framework::VarDesc* desc_ptr_;
  // TODO(jiabin): This is really ugly, but we have to hold a dims here so that
  // we can inherit from ExtendedTensor Remove this when we make VarDesc's as
  // same as Tensor, or make Tensor's dims more lightly.
  mutable phi::DDim dims_;
  phi::Place place_;
};

}  // namespace prim
}  // namespace paddle
