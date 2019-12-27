// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <unordered_map>
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/type_defs.h"

namespace paddle {
namespace framework {

/*
  Inplace Inference for create In->Out pairs for inplaced operator.
  If we specify a pair of corresponding names. For example, X->Out.
  then Out will inplaced use X's memory. The base class will do
  legality validation for both variables.
*/

class InplaceOpInference {
 public:
  virtual ~InplaceOpInference() {}
  virtual std::unordered_map<std::string, std::string> operator()(
      bool use_cuda) const = 0;
};

#define DECLARE_INPLACE_OP_INFERER(class_name, ...)                         \
  class class_name final : public ::paddle::framework::InplaceOpInference { \
   public:                                                                  \
    std::unordered_map<std::string, std::string> operator()(                \
        bool use_cuda) const final {                                        \
      return {__VA_ARGS__};                                                 \
    }                                                                       \
  }

#define DECLARE_CUDA_ONLY_INPLACE_OP_INFERER(class_name, ...)               \
  class class_name final : public ::paddle::framework::InplaceOpInference { \
   public:                                                                  \
    std::unordered_map<std::string, std::string> operator()(                \
        bool use_cuda) const final {                                        \
      if (use_cuda) {                                                       \
        return {__VA_ARGS__};                                               \
      } else {                                                              \
        return {};                                                          \
      }                                                                     \
    }                                                                       \
  }

}  // namespace framework
}  // namespace paddle
