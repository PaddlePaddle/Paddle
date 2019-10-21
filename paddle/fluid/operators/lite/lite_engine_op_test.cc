/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#include <gtest/gtest.h>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "paddle/fluid/operators/lite/lite_engine_op.h"
#include "paddle/fluid/operators/lite/ut_helper.h"

#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"

USE_NO_KERNEL_OP(lite_engine)
namespace paddle {
namespace operators {

namespace {
void CreateTensor(framework::Scope* scope, const std::string& name,
                  const std::vector<int64_t>& shape) {
  auto* var = scope->Var(name);
  auto* tensor = var->GetMutable<framework::LoDTensor>();
  auto dims = framework::make_ddim(shape);
  tensor->Resize(dims);
#ifdef PADDLE_WITH_CUDA
  platform::CUDAPlace place;
#else
  platform::CPUPlace place;
#endif
  inference::lite::RandomizeTensor(tensor, place);
}

void AddTensorToBlockDesc(framework::proto::BlockDesc* block,
                          const std::string& name,
                          const std::vector<int64_t>& shape, bool persistable) {
  using framework::proto::VarType;
  auto* var = block->add_vars();
  framework::VarDesc desc(name);
  desc.SetType(VarType::LOD_TENSOR);
  desc.SetDataType(VarType::FP32);
  desc.SetShape(shape);
  desc.SetPersistable(persistable);
  *var = *desc.Proto();
}
}  // namespace

TEST(LiteEngineOp, manual) {
  framework::ProgramDesc program;
  auto* block_ = program.Proto()->mutable_blocks(0);

  LOG(INFO) << "create block desc";
  framework::BlockDesc block_desc(&program, block_);
  LOG(INFO) << "create elementwise_add op";
  auto* elt_add = block_desc.AppendOp();
  elt_add->SetType("elementwise_add");
  elt_add->SetInput("X", std::vector<std::string>({"x"}));
  elt_add->SetInput("Y", std::vector<std::string>({"y"}));
  elt_add->SetOutput("Out", std::vector<std::string>({"z"}));
  elt_add->SetAttr("axis", -1);
  LOG(INFO) << "create fetch op";
  auto* fetch = block_desc.AppendOp();
  fetch->SetType("fetch");
  fetch->SetInput("X", std::vector<std::string>({"z"}));
  fetch->SetOutput("Out", std::vector<std::string>({"out"}));
  fetch->SetAttr("col", 0);
  // Set inputs' variable shape in BlockDesc
  AddTensorToBlockDesc(block_, "x", std::vector<int64_t>({2, 4}), true);
  AddTensorToBlockDesc(block_, "y", std::vector<int64_t>({2, 4}), true);
  AddTensorToBlockDesc(block_, "z", std::vector<int64_t>({2, 4}), false);
  AddTensorToBlockDesc(block_, "out", std::vector<int64_t>({2, 4}), false);

  *block_->add_ops() = *elt_add->Proto();
  *block_->add_ops() = *fetch->Proto();

  framework::Scope scope;
#ifdef PADDLE_WITH_CUDA
  platform::CUDAPlace place;
  platform::CUDADeviceContext ctx(place);
#else
  platform::CPUPlace place;
  platform::CPUDeviceContext ctx(place);
#endif
  // Prepare variables.
  CreateTensor(&scope, "x", std::vector<int64_t>({2, 4}));
  CreateTensor(&scope, "y", std::vector<int64_t>({2, 4}));
  CreateTensor(&scope, "z", std::vector<int64_t>({2, 4}));
  CreateTensor(&scope, "out", std::vector<int64_t>({2, 4}));

  ASSERT_EQ(block_->ops_size(), 2);

  auto serialize_params = [](std::string* str, framework::Scope* scope,
                             const std::vector<std::string>& params) {
    std::ostringstream os;
#ifdef PADDLE_WITH_CUDA
    platform::CUDAPlace place;
    platform::CUDADeviceContext ctx(place);
#else
    platform::CPUDeviceContext ctx;
#endif
    for (const auto& param : params) {
      PADDLE_ENFORCE_NOT_NULL(scope->FindVar(param),
                              "Block should already have a '%s' variable",
                              param);
      auto* tensor = scope->FindVar(param)->GetMutable<framework::LoDTensor>();
      framework::SerializeToStream(os, *tensor, ctx);
    }
    *str = os.str();
  };
  std::vector<std::string> repetitive_params{"x", "y"};
  inference::lite::EngineConfig config;
  config.prefer_place = {
#ifdef PADDLE_WITH_CUDA
      TARGET(kCUDA), PRECISION(kFloat),
#else
      TARGET(kX86), PRECISION(kFloat)
#endif
  };
  config.valid_places = {
      paddle::lite::Place({TARGET(kHost), PRECISION(kAny)}),
      paddle::lite::Place({TARGET(kX86), PRECISION(kFloat)}),
#ifdef PADDLE_WITH_CUDA
      paddle::lite::Place({TARGET(kCUDA), PRECISION(kFloat)}),
#endif
  };
  serialize_params(&(config.param), &scope, repetitive_params);
  config.model = program.Proto()->SerializeAsString();

  LOG(INFO) << "create lite_engine desc";
  framework::OpDesc engine_op_desc(nullptr);
  engine_op_desc.SetType("lite_engine");
  engine_op_desc.SetInput("Xs", std::vector<std::string>({"x", "y"}));
  engine_op_desc.SetOutput("Ys", std::vector<std::string>({"out"}));
  std::string engine_key = "engine_0";
  engine_op_desc.SetAttr("engine_key", engine_key);
  engine_op_desc.SetBlockAttr("sub_block", &block_desc);

  inference::Singleton<inference::lite::EngineManager>::Global().Create(
      engine_key, config);

  LOG(INFO) << "create engine op";
  auto engine_op = framework::OpRegistry::CreateOp(engine_op_desc);
  LOG(INFO) << "engine_op " << engine_op.get();

  // Execute them.
  LOG(INFO) << "engine_op run";
  engine_op->Run(scope, place);
}
}  // namespace operators
}  // namespace paddle
