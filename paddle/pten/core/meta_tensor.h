/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/pten/core/macros.h"
#include "paddle/pten/core/tensor_base.h"
#include "paddle/pten/core/tensor_meta.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/framework/ddim.h"

namespace pten {

class MetaTensor {
 public:
  explicit MetaTensor(TensorBase* tensor) : tensor_(tensor) {}

  MetaTensor() = default;
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

 private:
  const LoD& lod() const;
  TensorBase* tensor_;
};

}  // namespace pten
