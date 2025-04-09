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

#pragma once

#include "paddle/common/ddim.h"
#include "paddle/common/layout.h"
#include "paddle/common/macros.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/tensor_base.h"
#include "paddle/phi/core/tensor_meta.h"

namespace phi {

struct TEST_API MetaConfig {
  bool is_runtime{true};
  bool is_run_mkldnn_kernel{false};
  MetaConfig() = default;

  // supporting implicit construction is easier to use
  MetaConfig(bool is_runtime, bool is_run_mkldnn_kernel)
      : is_runtime(is_runtime),
        is_run_mkldnn_kernel(is_run_mkldnn_kernel) {}  // NOLINT
};

class TEST_API MetaTensor {
 public:
  typedef void (*unspecified_bool_type)();

  MetaTensor() : tensor_(nullptr) {}

  // supporting implicit construction is easier to use
  MetaTensor(TensorBase* tensor, bool strided_kernel_used = false)  // NOLINT
      : tensor_(tensor), strided_kernel_used_(strided_kernel_used) {}
  MetaTensor(const TensorBase& tensor,
             bool strided_kernel_used = false)
      : tensor_(const_cast<TensorBase*>(&tensor)),  // NOLINT
        strided_kernel_used_(strided_kernel_used) {}
  MetaTensor(const TensorBase* tensor,
             bool strided_kernel_used = false)     // NOLINT
      : tensor_(const_cast<TensorBase*>(tensor)),  // NOLINT
        strided_kernel_used_(strided_kernel_used) {}
  MetaTensor(TensorBase& tensor, bool strided_kernel_used = false)  // NOLINT
      : tensor_(&tensor),                                           // NOLINT
        strided_kernel_used_(strided_kernel_used) {}

  MetaTensor(MetaTensor&&) = default;
  MetaTensor& operator=(MetaTensor&&) = default;
  MetaTensor(const MetaTensor&) = default;
  MetaTensor& operator=(const MetaTensor&) = default;

  virtual ~MetaTensor() = default;

  virtual int64_t numel() const;
  virtual DDim dims() const;
  size_t size() const;  // Returns the number of tensors in TensorArray.
  DDim dims(int64_t index) const;
  virtual DataType dtype() const;
  virtual DataLayout layout() const;
  virtual DDim strides() const;
  virtual void set_dims(const DDim& dims);
  virtual void set_dtype(DataType dtype);
  virtual void set_layout(DataLayout layout);
  virtual void set_strides(const DDim& strides);

  virtual void share_lod(const MetaTensor& meta_tensor);
  void share_lod(const LegacyLoD& legacy_lod);
  void share_lod(const MetaTensor& meta_tensor, int64_t index);
  virtual void share_meta(const MetaTensor& meta_tensor);
  virtual void share_dims(const MetaTensor& meta_tensor);
  virtual void share_strides(const MetaTensor& meta_tensor);

  virtual bool initialized() const;

  virtual bool is_selected_rows() const;
  virtual bool is_dense() const;
  virtual bool is_dist() const;

  // TODO(YuanRisheng) This API is for compatible with Fluid
  //  and it will be deleted in the future.
  virtual bool is_tensor_array() const;

  virtual bool is_same_tensor(const MetaTensor& meta_tensor) const;

  virtual operator unspecified_bool_type() const {
    return tensor_ == nullptr ? 0 : unspecified_bool_true;
  }

  virtual bool operator!() const { return tensor_ == nullptr; }

 protected:
  static void unspecified_bool_true() {}

 protected:
  // Because the lod in compiletime and runtime is different,
  // so `LoD` cannot in public methods
  const LegacyLoD& lod() const;
  const LegacyLoD& lod(int64_t index) const;
  TensorBase* tensor() const;

  TensorBase* tensor_ = nullptr;
  bool strided_kernel_used_ = false;
};

}  // namespace phi
