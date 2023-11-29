/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include <memory>
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {
const char *const kGelu = "gelu";
const char *const kGeluGrad = "gelu_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, GeluEquivalenceTrans) {
  builder::Op inputs = *(map_inputs["X"].at(0));
  auto *op = node->Op();
  auto approximate = PADDLE_GET_CONST(bool, op->GetAttr("approximate"));
  auto use_mkldnn = op->HasAttr("use_mkldnn")
                        ? PADDLE_GET_CONST(bool, op->GetAttr("use_mkldnn"))
                        : false;
  auto use_cudnn = op->HasAttr("use_cudnn")
                       ? PADDLE_GET_CONST(bool, op->GetAttr("use_cudnn"))
                       : false;
  auto input_shape = inputs.GetType().GetShape();
  if (use_mkldnn || use_cudnn) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "GCU not support mkldnn/cudnn for gelu"));
    return nullptr;
  }
  auto result = builder::Gelu(inputs, approximate);
  if (result.GetType().GetRank() == 0) {
    result = builder::Reshape(result, input_shape);
  }
  return std::make_shared<GcuOp>(result);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, GeluGradEquivalenceTrans) {
  builder::Op x = *(map_inputs["X"].at(0));
  builder::Op out = *(map_inputs["Out@GRAD"].at(0));
  return std::make_shared<GcuOp>(builder::GeluGrad(out, x));
}

EQUIVALENCE_TRANS_FUNC_REG(kGelu, INSENSITIVE, GeluEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kGeluGrad, INSENSITIVE, GeluGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
