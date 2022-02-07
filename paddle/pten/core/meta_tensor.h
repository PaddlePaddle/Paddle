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

#include "paddle/pten/common/data_type.h"
#include "paddle/pten/common/layout.h"
#include "paddle/pten/core/ddim.h"
#include "paddle/pten/core/macros.h"
#include "paddle/pten/core/tensor_base.h"
#include "paddle/pten/core/tensor_meta.h"

namespace pten {

// TODO(chenweihang): add other flags if needed
struct MetaConfig {
  bool is_runtime{true};

  MetaConfig() = default;

  // supporting implicit construction is easier to use
  MetaConfig(bool is_runtime) : is_runtime(is_runtime) {}  // NOLINT
};

class MetaTensor {
 public:
  MetaTensor() = default;

  // supporting implicit construction is easier to use
  MetaTensor(TensorBase* tensor) : tensor_(tensor) {}  // NOLINT
  MetaTensor(const TensorBase& tensor)                 // NOLINT
      : tensor_(const_cast<TensorBase*>(&tensor)) {}
  MetaTensor(TensorBase& tensor) : tensor_(&tensor) {}  // NOLINT

  MetaTensor(const MetaTensor&) = default;
  MetaTensor(MetaTensor&&) = default;
  MetaTensor& operator=(const MetaTensor&) = delete;
  MetaTensor& operator=(MetaTensor&&) = delete;

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

 private:
  // Because the lod in compiletime and runtime is different,
  // so `LoD` cannot in public methods
  const LoD& lod() const;

  TensorBase* tensor_;
};

}  // namespace pten
