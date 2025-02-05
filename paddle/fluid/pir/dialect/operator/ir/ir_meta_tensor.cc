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
#include "paddle/fluid/pir/dialect/operator/ir/ir_selected_rows.h"
#include "paddle/fluid/pir/dialect/operator/ir/ir_sparse_tensor.h"
#include "paddle/fluid/pir/dialect/operator/ir/ir_tensor.h"

namespace paddle::dialect {
static inline void ValidCheck(const IrMetaTensor& meta_tensor) {
  PADDLE_ENFORCE_EQ(meta_tensor.initialized(),
                    true,
                    common::errors::InvalidArgument(
                        "The current MetaTensor is not initialized."));
}

int64_t IrMetaTensor::numel() const {
  ValidCheck(*this);
  return tensor_->numel();
}

phi::DDim IrMetaTensor::dims() const {
  ValidCheck(*this);
  return tensor_->dims();
}

phi::DataType IrMetaTensor::dtype() const {
  ValidCheck(*this);
  return tensor_->dtype();
}

phi::DataLayout IrMetaTensor::layout() const {
  ValidCheck(*this);
  return tensor_->layout();
}

const phi::LegacyLoD& IrMetaTensor::lod() const {
  ValidCheck(*this);
  return static_cast<paddle::dialect::IrTensor*>(tensor_)->lod();
}

void IrMetaTensor::set_dims(const phi::DDim& dims) {
  if (paddle::dialect::IrTensor::classof(tensor_)) {
    static_cast<paddle::dialect::IrTensor*>(tensor_)->SetDims(dims);
  } else if (paddle::dialect::IrSelectedRows::classof(tensor_)) {
    static_cast<paddle::dialect::IrSelectedRows*>(tensor_)->SetDims(dims);
  } else if (paddle::dialect::IrSparseCooTensor::classof(tensor_)) {
    static_cast<paddle::dialect::IrSparseCooTensor*>(tensor_)->SetDims(dims);
  } else if (paddle::dialect::IrSparseCsrTensor::classof(tensor_)) {
    static_cast<paddle::dialect::IrSparseCsrTensor*>(tensor_)->SetDims(dims);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "The current MetaTensor is not initialized."));
  }
}

void IrMetaTensor::set_dtype(phi::DataType dtype) {
  if (paddle::dialect::IrTensor::classof(tensor_)) {
    static_cast<paddle::dialect::IrTensor*>(tensor_)->SetDtype(dtype);
  } else if (paddle::dialect::IrSelectedRows::classof(tensor_)) {
    static_cast<paddle::dialect::IrSelectedRows*>(tensor_)->SetDtype(dtype);
  } else if (paddle::dialect::IrSparseCooTensor::classof(tensor_)) {
    static_cast<paddle::dialect::IrSparseCooTensor*>(tensor_)->SetDtype(dtype);
  } else if (paddle::dialect::IrSparseCsrTensor::classof(tensor_)) {
    static_cast<paddle::dialect::IrSparseCsrTensor*>(tensor_)->SetDtype(dtype);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "The current MetaTensor is not initialized."));
  }
}

void IrMetaTensor::set_layout(phi::DataLayout layout) {
  if (paddle::dialect::IrTensor::classof(tensor_)) {
    static_cast<paddle::dialect::IrTensor*>(tensor_)->SetLayout(layout);
  } else if (paddle::dialect::IrSelectedRows::classof(tensor_)) {
    static_cast<paddle::dialect::IrSelectedRows*>(tensor_)->SetLayout(layout);
  } else if (paddle::dialect::IrSparseCooTensor::classof(tensor_)) {
    static_cast<paddle::dialect::IrSparseCooTensor*>(tensor_)->SetLayout(
        layout);
  } else if (paddle::dialect::IrSparseCsrTensor::classof(tensor_)) {
    static_cast<paddle::dialect::IrSparseCsrTensor*>(tensor_)->SetLayout(
        layout);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "The current MetaTensor is not initialized."));
  }
}

void IrMetaTensor::share_lod(const MetaTensor& meta_tensor) {
  auto& ir_meta_tensor = static_cast<const IrMetaTensor&>(meta_tensor);
  static_cast<paddle::dialect::IrTensor*>(tensor_)->SetLod(
      ir_meta_tensor.lod());
}

void IrMetaTensor::share_dims(const MetaTensor& meta_tensor) {
  auto& ir_meta_tensor = static_cast<const IrMetaTensor&>(meta_tensor);
  set_dims(ir_meta_tensor.dims());
}

void IrMetaTensor::share_meta(const MetaTensor& meta_tensor) {
  auto& ir_meta_tensor = static_cast<const IrMetaTensor&>(meta_tensor);
  share_dims(ir_meta_tensor);
  set_dtype(ir_meta_tensor.dtype());
  set_layout(ir_meta_tensor.layout());
  share_lod(ir_meta_tensor);
}

bool IrMetaTensor::initialized() const { return tensor_ != nullptr; }

bool IrMetaTensor::is_selected_rows() const {
  return IrSelectedRows::classof(tensor_);
}

bool IrMetaTensor::is_tensor_array() const { return false; }

bool IrMetaTensor::is_dense() const { return false; }

}  // namespace paddle::dialect
