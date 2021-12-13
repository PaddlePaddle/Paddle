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

#include <string>
#include <vector>

#include "paddle/pten/api/ext/op_meta_info.h"

namespace paddle {
namespace framework {

class OpMetaInfoHelper {
 public:
  static const std::string& GetOpName(const paddle::OpMetaInfo& info) {
    return info.name_;
  }
  static const std::vector<std::string>& GetInputs(
      const paddle::OpMetaInfo& info) {
    return info.inputs_;
  }
  static const std::vector<std::string>& GetOutputs(
      const paddle::OpMetaInfo& info) {
    return info.outputs_;
  }
  static const std::vector<std::string>& GetAttrs(
      const paddle::OpMetaInfo& info) {
    return info.attrs_;
  }
  static const KernelFunc& GetKernelFn(const paddle::OpMetaInfo& info) {
    return info.kernel_fn_;
  }
  static const InferShapeFunc& GetInferShapeFn(const paddle::OpMetaInfo& info) {
    return info.infer_shape_fn_;
  }
  static const InferDtypeFunc& GetInferDtypeFn(const paddle::OpMetaInfo& info) {
    return info.infer_dtype_fn_;
  }
};

}  // namespace framework
}  // namespace paddle
