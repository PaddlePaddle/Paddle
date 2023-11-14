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

#include "paddle/fluid/pir/dialect/operator/ir/ir_meta_tensor.h"
#include "paddle/fluid/pir/dialect/operator/ir/ir_tensor.h"

namespace paddle {
namespace dialect {
static inline void ValidCheck(const IrMetaTensor& meta_tensor) {
  PADDLE_ENFORCE_EQ(meta_tensor.initialized(),
                    true,
                    phi::errors::InvalidArgument(
                        "The current MetaTensor is not initialized."));
}

int64_t IrMetaTensor::numel() const override {
  ValidCheck(*this);
  return tensor_->numel();
}

DDim IrMetaTensor::dims() const override {
  ValidCheck(*this);
  return tensor_->dims();
}

phi::DataType IrMetaTensor::dtype() const override {
  ValidCheck(*this);
  return tensor_->dtype();
}

DataLayout IrMetaTensor::layout() const override {
  ValidCheck(*this);
  return tensor_->layout();
}

void IrMetaTensor::set_dims(const DDim& dims) override {
  static_cast<paddle::dialect::IrTensor*>(tensor_)->SetDims(dims);
}

void IrMetaTensor::set_dtype(phi::DataType dtype) override {
  static_cast<paddle::dialect::IrTensor*>(tensor_)->SetDtype(dtype);
}

void IrMetaTensor::set_layout(DataLayout layout) override {
  static_cast<paddle::dialect::IrTensor*>(tensor_)->SetLayout(layout);
}

void IrMetaTensor::share_lod(const MetaTensor& meta_tensor) override {
  static_cast<paddle::dialect::IrTensor*>(tensor_)->SetLod(meta_tensor.lod());
}

void IrMetaTensor::share_dims(const MetaTensor& meta_tensor) override {
  set_dims(meta_tensor.dims());
}

void IrMetaTensor::share_meta(const MetaTensor& meta_tensor) override {
  share_dims(meta_tensor);
  set_dtype(meta_tensor.dtype());
  set_layout(meta_tensor.layout());
  share_lod(meta_tensor);
}

bool IrMetaTensor::initialized() const override { return tensor_ != nullptr; }

bool IrMetaTensor::is_selected_rows() const override { return false; }

bool IrMetaTensor::is_tensor_array() const override { return false; }

bool IrMetaTensor::is_dense() const override { return false; }

}  // namespace dialect
}  // namespace paddle
