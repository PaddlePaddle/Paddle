// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <vector>
#include "paddle/fluid/lite/core/compatible_tensor.h"
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/op_lite.h"
#include "paddle/fluid/lite/core/scope.h"
#include "paddle/fluid/lite/operators/op_params.h"
#include "paddle/fluid/lite/utils/all.h"

namespace paddle {
namespace lite {
namespace operators {

/*
 * The data types used by the two adjacent layers in the model should
 * be the same. When the two operators accept different data types,
 * we may need to implicitly add a data type conversion operator.
 * Currently, this operator only supports mutual conversion of int8
 * and float32 types.
 */
class CalibOpLite : public OpLite {
 public:
  CalibOpLite() {}

  explicit CalibOpLite(const std::string &type) : OpLite(type) {}

  bool CheckShape() const override;

  bool InferShape() const override;

  bool AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope);

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override { return "calib"; }

 private:
  mutable CalibParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
