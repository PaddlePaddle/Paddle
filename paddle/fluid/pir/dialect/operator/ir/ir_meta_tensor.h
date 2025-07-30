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

#pragma once

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/meta_tensor.h"

namespace paddle {
namespace dialect {
class IrMetaTensor : public phi::MetaTensor {
 public:
  using MetaTensor::MetaTensor;

  int64_t numel() const override;

  phi::DDim dims() const override;

  phi::DataType dtype() const override;

  phi::DataLayout layout() const override;

  const phi::LegacyLoD& lod() const;

  void set_dims(const phi::DDim& dims) override;

  void set_dtype(phi::DataType dtype) override;

  void set_layout(phi::DataLayout layout) override;

  void share_lod(const MetaTensor& meta_tensor) override;

  void share_dims(const MetaTensor& meta_tensor) override;

  void share_meta(const MetaTensor& meta_tensor) override;

  bool initialized() const override;

  bool is_selected_rows() const override;
  bool is_tensor_array() const override;
  bool is_dense() const override;

  operator unspecified_bool_type() const override {
    return initialized() ? unspecified_bool_true : 0;
  }

  bool operator!() const override { return tensor_ == nullptr; }

  static void unspecified_bool_true() {}
};

}  // namespace dialect
}  // namespace paddle
