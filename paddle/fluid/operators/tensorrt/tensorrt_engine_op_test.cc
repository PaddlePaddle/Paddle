/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/tensorrt/tensorrt_engine_op.h"

#include <gtest/gtest.h>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/convert/ut_helper.h"
#include "paddle/phi/common/data_type.h"

USE_NO_KERNEL_OP(tensorrt_engine);
namespace paddle {
namespace operators {

namespace {
void CreateCUDATensor(framework::Scope* scope,
                      const std::string& name,
                      const std::vector<int64_t>& shape) {
  auto* var = scope->Var(name);
  auto* tensor = var->GetMutable<phi::DenseTensor>();
  auto dims = phi::make_ddim(shape);
  tensor->Resize(dims);
  platform::CUDAPlace place;
  phi::GPUContext ctx(place);
  ctx.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                       .GetAllocator(place, ctx.stream())
                       .get());
  ctx.PartialInitWithAllocator();
  inference::tensorrt::RandomizeTensor(tensor, place, ctx);
}

void AddTensorToBlockDesc(framework::proto::BlockDesc* block,
                          const std::string& name,
                          const std::vector<int64_t>& shape) {
  using framework::proto::VarType;
  auto* var = block->add_vars();
  framework::VarDesc desc(name);
  desc.SetType(VarType::LOD_TENSOR);
  desc.SetDataType(VarType::FP32);
  desc.SetShape(shape);
  *var = *desc.Proto();
}

}  // namespace

using inference::analysis::SetAttr;

void DynamicShapeTest(bool allow_build_at_runtime) {
  framework::ProgramDesc program;
  auto* block_ = program.Proto()->add_blocks();
  block_->set_idx(0);
  block_->set_parent_idx(-1);

  LOG(INFO) << "create block desc";
  framework::BlockDesc block_desc(&program, block_);
  LOG(INFO) << "create fc op";
  auto* fc0 = block_desc.AppendOp();
  fc0->SetType("fc");
  fc0->SetInput("X", std::vector<std::string>({"x"}));     // 4 x 1 x 1
  fc0->SetInput("Y", std::vector<std::string>({"y"}));     // 4 x 6
  fc0->SetOutput("Out", std::vector<std::string>({"z"}));  // 6 x 1 x 1

  LOG(INFO) << "create fc op";
  auto* fc1 = block_desc.AppendOp();
  fc1->SetType("fc");
  fc1->SetInput("X", std::vector<std::string>({"z"}));
  fc1->SetInput("Y", std::vector<std::string>({"y0"}));     // 6 x 8
  fc1->SetOutput("Out", std::vector<std::string>({"z0"}));  // 8 x 1 x 1

  // Set inputs' variable shape in BlockDesc
  // the batch size is 2, so the dims of 'x' is {2, 4, 1, 1}
  AddTensorToBlockDesc(block_, "x", std::vector<int64_t>({2, 4, 1, 1}));
  AddTensorToBlockDesc(block_, "y", std::vector<int64_t>({4, 6}));
  AddTensorToBlockDesc(block_, "y0", std::vector<int64_t>({6, 8}));
  AddTensorToBlockDesc(block_, "z", std::vector<int64_t>({2, 6}));

  // It is wired, need to copy manually.
  *block_->add_ops() = *fc0->Proto();
  *block_->add_ops() = *fc1->Proto();

  ASSERT_EQ(block_->ops_size(), 2);

  LOG(INFO) << "create tensorrt desc";
  framework::OpDesc engine_op_desc(nullptr);
  engine_op_desc.SetType("tensorrt_engine");
  engine_op_desc.SetInput("Xs", std::vector<std::string>({"x"}));
  engine_op_desc.SetOutput("Ys", std::vector<std::string>({"z0"}));
  engine_op_desc.SetAttr("origin_outputs_dtype", std::vector<int>{5});

  engine_op_desc.SetBlockAttr("sub_block", &block_desc);
  engine_op_desc.SetAttr("max_batch_size", static_cast<int>(2));
  engine_op_desc.SetAttr("workspace_size", static_cast<int64_t>(1 << 20));
  engine_op_desc.SetAttr("parameters", std::vector<std::string>({}));
  engine_op_desc.SetAttr("engine_key", std::string("a_engine"));
  engine_op_desc.SetAttr("calibration_engine_key",
                         std::string("a_calib_engine"));
  engine_op_desc.SetAttr("predictor_id", 1);
  engine_op_desc.SetAttr("calibration_data", std::string(""));
  engine_op_desc.SetAttr("enable_int8", static_cast<bool>(false));
  engine_op_desc.SetAttr("enable_fp16", static_cast<bool>(false));
  engine_op_desc.SetAttr("use_calib_mode", static_cast<bool>(false));
  engine_op_desc.SetAttr("output_name_mapping",
                         std::vector<std::string>({"z0"}));
  engine_op_desc.SetAttr("origin_output_rank", std::vector<int>({2}));
  engine_op_desc.SetAttr("subgraph", std::string(block_->SerializeAsString()));
  engine_op_desc.SetAttr("engine_serialized_data", std::string(""));
  int device_id = 0;
  engine_op_desc.SetAttr("gpu_id", device_id);
  engine_op_desc.SetAttr("shape_range_info_path", std::string(""));
  engine_op_desc.SetAttr("model_opt_cache_dir", std::string(""));
  engine_op_desc.SetAttr("allow_build_at_runtime", allow_build_at_runtime);
  engine_op_desc.SetAttr("use_static_engine", true);
  engine_op_desc.SetAttr("dynamic_shape_names", std::vector<std::string>{"x"});
  engine_op_desc.SetAttr("dynamic_shape_lens", std::vector<int>{4});
  engine_op_desc.SetAttr("min_input_shape", std::vector<int>{1, 4, 1, 1});
  engine_op_desc.SetAttr("max_input_shape", std::vector<int>{2, 4, 1, 1});
  engine_op_desc.SetAttr("opt_input_shape", std::vector<int>{2, 4, 1, 1});
  engine_op_desc.SetAttr("model_precision",
                         static_cast<int>(phi::DataType::FLOAT32));

  LOG(INFO) << "create engine op";
  auto engine_op = framework::OpRegistry::CreateOp(engine_op_desc);
  LOG(INFO) << "engine_op " << engine_op.get();

  framework::Scope scope;
  platform::CUDAPlace place;
  phi::GPUContext ctx(place);
  ctx.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                       .GetAllocator(place, ctx.stream())
                       .get());
  ctx.PartialInitWithAllocator();
  // Prepare variables.
  if (allow_build_at_runtime)
    CreateCUDATensor(&scope, "x", std::vector<int64_t>({3, 4, 1, 1}));
  else
    CreateCUDATensor(&scope, "x", std::vector<int64_t>({2, 4, 1, 1}));
  CreateCUDATensor(&scope, "y", std::vector<int64_t>({4, 6}));

  CreateCUDATensor(&scope, "y0", std::vector<int64_t>({6, 8}));
  CreateCUDATensor(&scope, "z0", std::vector<int64_t>({2, 8}));

  // Execute them.
  LOG(INFO) << "engine_op run";
  inference::tensorrt::OpTeller::Global().SetOpConverterType(
      "fc", inference::tensorrt::OpConverterType::Default);
  engine_op->Run(scope, place);
}

TEST(TensorRTEngineOp, manual) {
  DynamicShapeTest(false);
  DynamicShapeTest(true);
}

void Execute(int batch_size, int input_dim, int output_dim, int nlayers = 1) {
  framework::ProgramDesc program;
  framework::Scope scope;
  platform::CUDAPlace place;
  phi::GPUContext ctx(place);
  ctx.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                       .GetAllocator(place, ctx.stream())
                       .get());
  ctx.PartialInitWithAllocator();

  auto* block_ = program.Proto()->add_blocks();
  block_->set_idx(0);
  block_->set_parent_idx(-1);

  using shape_t = std::vector<int64_t>;

  LOG(INFO) << "create block desc";
  framework::BlockDesc block_desc(&program, block_);

  auto AddFCLayer = [&](const std::string& x_name,
                        const std::string& y_name,
                        const std::string& z_name,
                        bool x_created,
                        const shape_t& x_shape,
                        const shape_t& y_shape,
                        const shape_t& z_shape) {
    LOG(INFO) << "create fc op";
    auto* fc = block_desc.AppendOp();
    fc->SetType("mul");
    fc->SetInput("X", std::vector<std::string>({x_name}));
    fc->SetInput("Y", std::vector<std::string>({y_name}));
    fc->SetOutput("Out", std::vector<std::string>({z_name}));

    // Set inputs' variable shape in BlockDesc
    if (!x_created) {
      AddTensorToBlockDesc(
          block_, x_name, std::vector<int64_t>({batch_size, input_dim, 1, 1}));
    }
    AddTensorToBlockDesc(
        block_, y_name, std::vector<int64_t>({input_dim, output_dim}));
    AddTensorToBlockDesc(
        block_, z_name, std::vector<int64_t>({batch_size, output_dim}));

    // Prepare variables.
    if (!x_created) {
      CreateCUDATensor(&scope, x_name, std::vector<int64_t>(x_shape));
    }
    CreateCUDATensor(&scope, y_name, std::vector<int64_t>(y_shape));
    CreateCUDATensor(&scope, z_name, std::vector<int64_t>(z_shape));

    // It is wired, need to copy manually.
    *block_->add_ops() = *fc->Proto();
  };

  // Test with 4 layer FC
  AddFCLayer("x0",
             "y0",
             "z0",
             false,
             {batch_size, input_dim},
             {input_dim, output_dim},
             {batch_size, output_dim});
  AddFCLayer("z0",
             "y1",
             "z1",
             true,
             {},
             {output_dim, output_dim},
             {batch_size, output_dim});
  AddFCLayer("z1",
             "y2",
             "z2",
             true,
             {},
             {output_dim, output_dim},
             {batch_size, output_dim});
  AddFCLayer("z2",
             "y3",
             "z3",
             true,
             {},
             {output_dim, output_dim},
             {batch_size, output_dim});

  LOG(INFO) << "create tensorrt desc";
  framework::OpDesc engine_op_desc(nullptr);
  engine_op_desc.SetType("tensorrt_engine");
  engine_op_desc.SetInput("Xs", std::vector<std::string>({"x0"}));
  engine_op_desc.SetOutput("Ys", std::vector<std::string>({"z3"}));

  engine_op_desc.SetBlockAttr("sub_block", &block_desc);
  engine_op_desc.SetAttr("max_batch_size", static_cast<int>(batch_size));
  engine_op_desc.SetAttr("workspace_size", static_cast<int64_t>(1 << 20));
  engine_op_desc.SetAttr("parameters",
                         std::vector<std::string>({"y0", "y1", "y2", "y3"}));
  engine_op_desc.SetAttr("engine_key", std::string("b_engine"));
  engine_op_desc.SetAttr("calibration_engine_key",
                         std::string("b_calib_engine"));
  engine_op_desc.SetAttr("predictor_id", 1);
  engine_op_desc.SetAttr("calibration_data", std::string(""));
  engine_op_desc.SetAttr("enable_int8", static_cast<bool>(false));
  engine_op_desc.SetAttr("enable_fp16", static_cast<bool>(false));
  engine_op_desc.SetAttr("use_calib_mode", static_cast<bool>(false));
  engine_op_desc.SetAttr("output_name_mapping",
                         std::vector<std::string>({"z3"}));
  engine_op_desc.SetAttr("origin_output_rank", std::vector<int>({2}));
  engine_op_desc.SetAttr("subgraph", std::string(block_->SerializeAsString()));
  engine_op_desc.SetAttr("engine_serialized_data", std::string(""));
  int device_id = 0;
  engine_op_desc.SetAttr("gpu_id", device_id);
  engine_op_desc.SetAttr("shape_range_info_path", std::string(""));
  engine_op_desc.SetAttr("model_opt_cache_dir", std::string(""));
  engine_op_desc.SetAttr("allow_build_at_runtime", false);
  engine_op_desc.SetAttr("use_static_engine", false);

  auto engine_op = framework::OpRegistry::CreateOp(engine_op_desc);

  // Execute them.
  engine_op->Run(scope, place);
}

// Test with a larger FC layer.
// TEST(TensorRTEngineOp, fc) { Execute(40, 28, 28); }

}  // namespace operators
}  // namespace paddle

USE_TRT_CONVERTER(fc)
