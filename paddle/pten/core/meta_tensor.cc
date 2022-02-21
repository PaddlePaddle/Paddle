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

#include "paddle/pten/core/meta_tensor.h"

#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/enforce.h"
#include "paddle/pten/core/selected_rows.h"
#include "paddle/pten/core/tensor_utils.h"

namespace pten {

int64_t MetaTensor::numel() const { return tensor_->numel(); }

DDim MetaTensor::dims() const { return tensor_->dims(); }

DataType MetaTensor::dtype() const { return tensor_->dtype(); }

DataLayout MetaTensor::layout() const { return tensor_->layout(); }

void MetaTensor::set_dims(const DDim& dims) {
  if (pten::DenseTensor::classof(tensor_)) {
    DenseTensorUtils::GetMutableMeta(static_cast<DenseTensor*>(tensor_))->dims =
        dims;
  } else if (pten::SelectedRows::classof(tensor_)) {
    DenseTensorUtils::GetMutableMeta(
        static_cast<SelectedRows*>(tensor_)->mutable_value())
        ->dims = dims;
  } else {
    PADDLE_THROW(pten::errors::Unimplemented(
        "Unsupported setting dims for `%s`.", tensor_->type_info().name()));
  }
}

void MetaTensor::set_dtype(DataType dtype) {
  if (pten::DenseTensor::classof(tensor_)) {
    DenseTensorUtils::GetMutableMeta(static_cast<DenseTensor*>(tensor_))
        ->dtype = dtype;
  } else if (pten::SelectedRows::classof(tensor_)) {
    DenseTensorUtils::GetMutableMeta(
        static_cast<SelectedRows*>(tensor_)->mutable_value())
        ->dtype = dtype;
  } else {
    PADDLE_THROW(pten::errors::Unimplemented(
        "Unsupported settting dtype for `%s`.", tensor_->type_info().name()));
  }
}

void MetaTensor::set_layout(DataLayout layout) {
  if (pten::DenseTensor::classof(tensor_)) {
    DenseTensorUtils::GetMutableMeta(static_cast<DenseTensor*>(tensor_))
        ->layout = layout;
  } else if (pten::SelectedRows::classof(tensor_)) {
    DenseTensorUtils::GetMutableMeta(
        static_cast<SelectedRows*>(tensor_)->mutable_value())
        ->layout = layout;
  } else {
    PADDLE_THROW(pten::errors::Unimplemented(
        "Unsupported settting layout for `%s`.", tensor_->type_info().name()));
  }
}

void MetaTensor::share_lod(const MetaTensor& meta_tensor) {
  if (pten::DenseTensor::classof(tensor_)) {
    DenseTensorUtils::GetMutableMeta(static_cast<DenseTensor*>(tensor_))->lod =
        meta_tensor.lod();
  } else if (pten::SelectedRows::classof(tensor_)) {
    DenseTensorUtils::GetMutableMeta(
        static_cast<SelectedRows*>(tensor_)->mutable_value())
        ->lod = meta_tensor.lod();
  } else {
    PADDLE_THROW(
        pten::errors::Unimplemented("Unsupported sharing lod inplace for `%s`.",
                                    tensor_->type_info().name()));
  }
}

const LoD& MetaTensor::lod() const {
  if (pten::DenseTensor::classof(tensor_)) {
    return static_cast<DenseTensor*>(tensor_)->lod();
  } else if (pten::SelectedRows::classof(tensor_)) {
    return static_cast<SelectedRows*>(tensor_)->value().lod();
  } else {
    PADDLE_THROW(pten::errors::Unimplemented("Unsupported getting lod of `%s`.",
                                             tensor_->type_info().name()));
  }
}

void MetaTensor::share_meta(const MetaTensor& meta_tensor) {
  if (pten::DenseTensor::classof(tensor_)) {
    set_dims(meta_tensor.dims());
    set_dtype(meta_tensor.dtype());
    set_layout(meta_tensor.layout());
    share_lod(meta_tensor);
  } else if (pten::SelectedRows::classof(tensor_)) {
    set_dims(meta_tensor.dims());
    set_dtype(meta_tensor.dtype());
    set_layout(meta_tensor.layout());
    share_lod(meta_tensor);
  } else {
    PADDLE_THROW(pten::errors::Unimplemented(
        "Unsupported sharing meta for `%s`.", tensor_->type_info().name()));
  }
}

}  // namespace pten
