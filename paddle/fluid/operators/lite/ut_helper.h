/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/platform/errors.h"

namespace paddle {
namespace inference {
namespace lite {

void AddTensorToBlockDesc(framework::proto::BlockDesc* block,
                          const std::string& name,
                          const std::vector<int64_t>& shape,
                          bool persistable = false) {
  using framework::proto::VarType;
  auto* var = block->add_vars();
  framework::VarDesc desc(name);
  desc.SetType(VarType::LOD_TENSOR);
  desc.SetDataType(VarType::FP32);
  desc.SetShape(shape);
  desc.SetPersistable(persistable);
  *var = *desc.Proto();
}

void AddFetchListToBlockDesc(framework::proto::BlockDesc* block,
                             const std::string& name) {
  using framework::proto::VarType;
  auto* var = block->add_vars();
  framework::VarDesc desc(name);
  desc.SetType(VarType::FETCH_LIST);
  *var = *desc.Proto();
}

void serialize_params(std::string* str, framework::Scope* scope,
                      const std::vector<std::string>& params) {
  std::ostringstream os;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  platform::CUDAPlace place;
  platform::CUDADeviceContext ctx(place);
#else
  platform::CPUDeviceContext ctx;
#endif
  for (const auto& param : params) {
    PADDLE_ENFORCE_NOT_NULL(
        scope->FindVar(param),
        platform::errors::NotFound("Block should already have a '%s' variable",
                                   param));
    auto* tensor = scope->FindVar(param)->GetMutable<framework::LoDTensor>();
    framework::SerializeToStream(os, *tensor, ctx);
  }
  *str = os.str();
}
/*
 * Get a random float value between [low, high]
 */
float random(float low, float high) {
  // static std::random_device rd;
  static std::mt19937 mt(100);
  std::uniform_real_distribution<double> dist(low, high);
  return dist(mt);
}
void RandomizeTensor(framework::LoDTensor* tensor,
                     const platform::Place& place) {
  auto dims = tensor->dims();
  size_t num_elements = analysis::AccuDims(dims, dims.size());
  PADDLE_ENFORCE_GT(num_elements, 0,
                    platform::errors::InvalidArgument(
                        "The input tensor dimension of the randomized tensor "
                        "function should be greater than zero."));
  platform::CPUPlace cpu_place;
  framework::LoDTensor temp_tensor;
  temp_tensor.Resize(dims);
  auto* temp_data = temp_tensor.mutable_data<float>(cpu_place);
  for (size_t i = 0; i < num_elements; i++) {
    *(temp_data + i) = random(0., 1.);
  }
  paddle::framework::TensorCopySync(temp_tensor, place, tensor);
}

void CreateTensor(framework::Scope* scope, const std::string& name,
                  const std::vector<int64_t>& shape, bool in_cuda = true) {
  auto* var = scope->Var(name);
  auto* tensor = var->GetMutable<framework::LoDTensor>();
  auto dims = phi::make_ddim(shape);
  tensor->Resize(dims);
  platform::Place place;
  if (in_cuda) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    place = platform::CUDAPlace(0);
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "You must define PADDLE_WITH_CUDA for using CUDAPlace."));
#endif
  } else {
    place = platform::CPUPlace();
  }
  RandomizeTensor(tensor, place);
}

}  // namespace lite
}  // namespace inference
}  // namespace paddle
