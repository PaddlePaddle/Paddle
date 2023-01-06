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

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/macros.h"
#include "paddle/phi/core/tensor_base.h"
#include "paddle/phi/core/tensor_meta.h"

namespace phi {

// TODO(chenweihang): add other flags if needed
struct MetaConfig {
  bool is_runtime{true};
  bool is_run_mkldnn_kernel{false};
  MetaConfig() = default;

  // supporting implicit construction is easier to use
  MetaConfig(bool is_runtime, bool is_run_mkldnn_kernel)
      : is_runtime(is_runtime),
        is_run_mkldnn_kernel(is_run_mkldnn_kernel) {}  // NOLINT
};

class MetaTensor {
 public:
  typedef void (*unspecified_bool_type)();

  MetaTensor() : tensor_(nullptr) {}

  // supporting implicit construction is easier to use
  MetaTensor(TensorBase* tensor) : tensor_(tensor) {}  // NOLINT
  MetaTensor(const TensorBase& tensor)                 // NOLINT
      : tensor_(const_cast<TensorBase*>(&tensor)) {}
  MetaTensor(TensorBase& tensor) : tensor_(&tensor) {}  // NOLINT

  MetaTensor(MetaTensor&&) = default;
  MetaTensor& operator=(MetaTensor&&) = default;
  MetaTensor(const MetaTensor&) = default;
  MetaTensor& operator=(const MetaTensor&) = default;

  virtual ~MetaTensor() = default;

  virtual int64_t numel() const;
  virtual DDim dims() const;
  virtual DataType dtype() const;
  virtual DataLayout layout() const;
  virtual void set_dims(const DDim& dims);
  virtual void set_dtype(DataType dtype);
  virtual void set_layout(DataLayout layout);

  virtual void share_lod(const MetaTensor& meta_tensor);
  virtual void share_meta(const MetaTensor& meta_tensor);
  virtual void share_dims(const MetaTensor& meta_tensor);

  virtual bool initialized() const;

  virtual bool is_selected_rows() const;
  virtual bool is_dense() const;
  // TODO(YuanRisheng) This API is for compatible with Fluid
  //  and it will be deleted in the future.
  virtual bool is_tensor_array() const;

  virtual operator unspecified_bool_type() const {
    return tensor_ == nullptr ? 0 : unspecified_bool_true;
  }

  virtual bool operator!() const { return tensor_ == nullptr; }

 protected:
  static void unspecified_bool_true() {}

 private:
  // Because the lod in compiletime and runtime is different,
  // so `LoD` cannot in public methods
  const LoD& lod() const;
  TensorBase* tensor() const;

  TensorBase* tensor_ = nullptr;
};

}  // namespace phi
