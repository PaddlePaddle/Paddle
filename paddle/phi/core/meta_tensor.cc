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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/selected_rows.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

int64_t MetaTensor::numel() const { return tensor_->numel(); }

DDim MetaTensor::dims() const { return tensor_->dims(); }

DataType MetaTensor::dtype() const { return tensor_->dtype(); }

DataLayout MetaTensor::layout() const { return tensor_->layout(); }

void MetaTensor::set_dims(const DDim& dims) {
  if (phi::DenseTensor::classof(tensor_)) {
    DenseTensorUtils::GetMutableMeta(static_cast<DenseTensor*>(tensor_))->dims =
        dims;
  } else if (phi::SelectedRows::classof(tensor_)) {
    DenseTensorUtils::GetMutableMeta(
        static_cast<SelectedRows*>(tensor_)->mutable_value())
        ->dims = dims;
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Unsupported setting dims for `%s`.", tensor_->type_info().name()));
  }
}

void MetaTensor::set_dtype(DataType dtype) {
  if (phi::DenseTensor::classof(tensor_)) {
    DenseTensorUtils::GetMutableMeta(static_cast<DenseTensor*>(tensor_))
        ->dtype = dtype;
  } else if (phi::SelectedRows::classof(tensor_)) {
    DenseTensorUtils::GetMutableMeta(
        static_cast<SelectedRows*>(tensor_)->mutable_value())
        ->dtype = dtype;
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Unsupported settting dtype for `%s`.", tensor_->type_info().name()));
  }
}

void MetaTensor::set_layout(DataLayout layout) {
  if (phi::DenseTensor::classof(tensor_)) {
    DenseTensorUtils::GetMutableMeta(static_cast<DenseTensor*>(tensor_))
        ->layout = layout;
  } else if (phi::SelectedRows::classof(tensor_)) {
    DenseTensorUtils::GetMutableMeta(
        static_cast<SelectedRows*>(tensor_)->mutable_value())
        ->layout = layout;
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Unsupported settting layout for `%s`.", tensor_->type_info().name()));
  }
}

void MetaTensor::share_lod(const MetaTensor& meta_tensor) {
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

const LoD& MetaTensor::lod() const {
  if (phi::DenseTensor::classof(tensor_)) {
    return static_cast<DenseTensor*>(tensor_)->lod();
  } else if (phi::SelectedRows::classof(tensor_)) {
    return static_cast<SelectedRows*>(tensor_)->value().lod();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented("Unsupported getting lod of `%s`.",
                                            tensor_->type_info().name()));
  }
}

void MetaTensor::share_meta(const MetaTensor& meta_tensor) {
  if (phi::DenseTensor::classof(tensor_) ||
      phi::SelectedRows::classof(tensor_)) {
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

void MetaTensor::share_dims(const MetaTensor& meta_tensor) {
  bool is_dense_tensor = phi::DenseTensor::classof(tensor_);
  bool is_selected_rows = phi::SelectedRows::classof(tensor_);
  if (is_dense_tensor || is_selected_rows) {
    set_dims(meta_tensor.dims());
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
    }
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Unsupported sharing dims for `%s`.", tensor_->type_info().name()));
  }
}

}  // namespace phi
