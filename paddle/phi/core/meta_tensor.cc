/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/meta_tensor.h"

#include "glog/logging.h"

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/selected_rows.h"
#include "paddle/phi/core/string_tensor.h"
#include "paddle/phi/core/string_tensor_utils.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

static inline void ValidCheck(const MetaTensor& meta_tensor) {
  PADDLE_ENFORCE_EQ(meta_tensor.initialized(),
                    true,
                    phi::errors::InvalidArgument(
                        "The current MetaTensor is not initialized."));
}

int64_t MetaTensor::numel() const {
  ValidCheck(*this);
  return tensor_->numel();
}

DDim MetaTensor::dims() const {
  ValidCheck(*this);
  if (phi::SelectedRows::classof(tensor_)) {
    return static_cast<SelectedRows*>(tensor_)->GetCompleteDims();
  } else {
    return tensor_->dims();
  }
}

DataType MetaTensor::dtype() const {
  ValidCheck(*this);
  return tensor_->dtype();
}

DataLayout MetaTensor::layout() const {
  ValidCheck(*this);
  return tensor_->layout();
}

void MetaTensor::set_dims(const DDim& dims) {
  ValidCheck(*this);
  if (phi::DenseTensor::classof(tensor_)) {
    DenseTensorUtils::GetMutableMeta(static_cast<DenseTensor*>(tensor_))->dims =
        dims;
  } else if (phi::StringTensor::classof(tensor_)) {
    StringTensorUtils::GetMutableMeta(static_cast<StringTensor*>(tensor_))
        ->dims = dims;
  } else if (phi::SelectedRows::classof(tensor_)) {
    static_cast<SelectedRows*>(tensor_)->set_height(dims[0]);
  } else if (phi::SparseCooTensor::classof(tensor_)) {
    DenseTensorUtils::GetMutableMeta(static_cast<SparseCooTensor*>(tensor_))
        ->dims = dims;
  } else if (phi::SparseCsrTensor::classof(tensor_)) {
    DenseTensorUtils::GetMutableMeta(static_cast<SparseCsrTensor*>(tensor_))
        ->dims = dims;
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Unsupported setting dims for `%s`.", tensor_->type_info().name()));
  }
}

void MetaTensor::set_dtype(DataType dtype) {
  ValidCheck(*this);
  if (phi::DenseTensor::classof(tensor_)) {
    DenseTensorUtils::GetMutableMeta(static_cast<DenseTensor*>(tensor_))
        ->dtype = dtype;
  } else if (phi::StringTensor::classof(tensor_)) {
    // No need to set dtype
  } else if (phi::SelectedRows::classof(tensor_)) {
    DenseTensorUtils::GetMutableMeta(
        static_cast<SelectedRows*>(tensor_)->mutable_value())
        ->dtype = dtype;
  } else if (phi::SparseCooTensor::classof(tensor_)) {
    DenseTensorUtils::GetMutableMeta(static_cast<SparseCooTensor*>(tensor_))
        ->dtype = dtype;
  } else if (phi::SparseCsrTensor::classof(tensor_)) {
    DenseTensorUtils::GetMutableMeta(static_cast<SparseCsrTensor*>(tensor_))
        ->dtype = dtype;
    // No need to set dtype
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Unsupported settting dtype for `%s`.", tensor_->type_info().name()));
  }
}

void MetaTensor::set_layout(DataLayout layout) {
  ValidCheck(*this);
  if (phi::DenseTensor::classof(tensor_)) {
    DenseTensorUtils::GetMutableMeta(static_cast<DenseTensor*>(tensor_))
        ->layout = layout;
  } else if (phi::StringTensor::classof(tensor_)) {
    // No need to set layout
  } else if (phi::SelectedRows::classof(tensor_)) {
    DenseTensorUtils::GetMutableMeta(
        static_cast<SelectedRows*>(tensor_)->mutable_value())
        ->layout = layout;
  } else if (phi::SparseCooTensor::classof(tensor_)) {
    DenseTensorUtils::GetMutableMeta(static_cast<SparseCooTensor*>(tensor_))
        ->layout = layout;
  } else if (phi::SparseCsrTensor::classof(tensor_)) {
    DenseTensorUtils::GetMutableMeta(static_cast<SparseCsrTensor*>(tensor_))
        ->layout = layout;
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Unsupported settting layout for `%s`.", tensor_->type_info().name()));
  }
}

void MetaTensor::share_lod(const MetaTensor& meta_tensor) {
  ValidCheck(*this);
  ValidCheck(meta_tensor);
  if (phi::SparseCooTensor::classof(tensor_) ||
      phi::SparseCsrTensor::classof(tensor_)) {
    return;
  }
  if (meta_tensor.lod().size() == 0) {
    // no need share
    return;
  }
  if (phi::DenseTensor::classof(tensor_)) {
    DenseTensorUtils::GetMutableMeta(static_cast<DenseTensor*>(tensor_))->lod =
        meta_tensor.lod();
  } else if (phi::SelectedRows::classof(tensor_)) {
    DenseTensorUtils::GetMutableMeta(
        static_cast<SelectedRows*>(tensor_)->mutable_value())
        ->lod = meta_tensor.lod();
  } else {
    PADDLE_THROW(
        phi::errors::Unimplemented("Unsupported sharing lod inplace for `%s`.",
                                   tensor_->type_info().name()));
  }
}

void MetaTensor::share_meta(const MetaTensor& meta_tensor) {
  ValidCheck(*this);
  if (phi::DenseTensor::classof(tensor_) ||
      phi::SelectedRows::classof(tensor_) ||
      phi::SparseCooTensor::classof(tensor_) ||
      phi::SparseCsrTensor::classof(tensor_)) {
    share_dims(meta_tensor);
    set_dtype(meta_tensor.dtype());
    set_layout(meta_tensor.layout());
    share_lod(meta_tensor);
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Unsupported sharing meta for `%s`.", tensor_->type_info().name()));
  }
}

TensorBase* MetaTensor::tensor() const { return tensor_; }

bool MetaTensor::is_dense() const { return DenseTensor::classof(tensor_); }
bool MetaTensor::is_selected_rows() const {
  return SelectedRows::classof(tensor_);
}

bool MetaTensor::is_tensor_array() const { return false; }

void MetaTensor::share_dims(const MetaTensor& meta_tensor) {
  ValidCheck(*this);
  bool is_dense_tensor = phi::DenseTensor::classof(tensor_);
  bool is_selected_rows = phi::SelectedRows::classof(tensor_);
  bool is_sparse_coo = phi::SparseCooTensor::classof(tensor_);
  bool is_sparse_csr = phi::SparseCsrTensor::classof(tensor_);
  if (is_dense_tensor || is_selected_rows || is_sparse_coo || is_sparse_csr) {
    if (is_selected_rows) {
      const auto in_tensor_base = meta_tensor.tensor();
      PADDLE_ENFORCE_EQ(
          phi::SelectedRows::classof(in_tensor_base),
          true,
          errors::InvalidArgument("The input MetaTensor is SelectedRows, but "
                                  "the output MetaTensor is not this type."));
      auto* selected_rows_out = static_cast<SelectedRows*>(tensor_);
      auto* selected_rows_in = static_cast<SelectedRows*>(in_tensor_base);
      selected_rows_out->set_rows(selected_rows_in->rows());
      selected_rows_out->set_height(selected_rows_in->height());
      DenseTensorUtils::GetMutableMeta(
          static_cast<SelectedRows*>(tensor_)->mutable_value())
          ->dims = selected_rows_in->mutable_value()->dims();
    } else {
      set_dims(meta_tensor.dims());
    }
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Unsupported sharing dims for `%s`.", tensor_->type_info().name()));
  }
}

bool MetaTensor::initialized() const { return tensor_ != nullptr; }

// Private Member Methods

const LoD& MetaTensor::lod() const {
  if (phi::DenseTensor::classof(tensor_)) {
    return static_cast<DenseTensor*>(tensor_)->lod();
  } else if (phi::SelectedRows::classof(tensor_)) {
    return static_cast<SelectedRows*>(tensor_)->value().lod();
  } else if (phi::SparseCooTensor::classof(tensor_)) {
    return static_cast<SparseCooTensor*>(tensor_)->non_zero_elements().lod();
  } else if (phi::SparseCsrTensor::classof(tensor_)) {
    return static_cast<SparseCsrTensor*>(tensor_)->non_zero_elements().lod();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented("Unsupported getting lod of `%s`.",
                                            tensor_->type_info().name()));
  }
}

}  // namespace phi
