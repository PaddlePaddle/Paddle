// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/utils/model_utils.h"
#include <set>
#include "paddle/fluid/framework/var_type_inference.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/framework/framework.pb.h"

namespace paddle::inference {

using paddle::framework::proto::VarType;

// Get all model's weights and return the data_type, e.g., fp16/bf16 or fp32.
phi::DataType GetModelPrecision(const framework::ProgramDesc& program) {
  std::set<VarType::Type> model_types{
      VarType::FP32,
      VarType::FP16,
      VarType::BF16,
  };

  phi::DataType ret = phi::DataType::FLOAT32;
  size_t block_size = program.Size();

  for (size_t i = 0; i < block_size; ++i) {
    const auto& block = program.Block(i);
    for (auto* var : block.AllVars()) {
      if (!(var->GetType() == VarType::DENSE_TENSOR ||
            var->GetType() == VarType::DENSE_TENSOR_ARRAY))
        continue;

      if (!var->Persistable()) continue;
      auto t = var->GetDataType();
      if (!model_types.count(t)) continue;

      if (t == VarType::FP16) {
        if (ret != phi::DataType::FLOAT32 && ret != phi::DataType::FLOAT16) {
          PADDLE_THROW(common::errors::PreconditionNotMet(
              "The model's weights already has been set %s type, but also has "
              "%s type, which is an error, please check the model.",
              ret,
              phi::DataType::FLOAT16));
        }
        ret = phi::DataType::FLOAT16;
      } else if (t == VarType::BF16) {
        if (ret != phi::DataType::FLOAT32 && ret != phi::DataType::BFLOAT16) {
          PADDLE_THROW(common::errors::PreconditionNotMet(
              "The model's weights already has been set %s type, but also has "
              "%s type, which is an error, please check the model.",
              ret,
              phi::DataType::BFLOAT16));
        }
        ret = phi::DataType::BFLOAT16;
      }
    }
  }

  return ret;
}

}  // namespace paddle::inference
