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

#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/platform/device_code.h"
#include "paddle/fluid/platform/init.h"

namespace paddle {
namespace operators {

using CPUKernelFunc = std::function<void(size_t n, std::vector<void*> args)>;

template <typename T>
phi::DenseTensor* CreateTensor(framework::Scope* scope,
                               const platform::Place& place,
                               const std::string& name,
                               const std::vector<int64_t>& shape) {
  auto* var = scope->Var(name);
  auto* tensor = var->GetMutable<framework::LoDTensor>();
  if (shape.size() > 0) {
    tensor->mutable_data<T>(phi::make_ddim(shape), place);
  }
  return tensor;
}

template <typename T>
void SetupRandomCPUTensor(phi::DenseTensor* tensor,
                          const std::vector<int64_t>& shape) {
  static unsigned int seed = 100;
  std::mt19937 rng(seed++);
  std::uniform_real_distribution<double> uniform_dist(0, 1);

  T* ptr = tensor->mutable_data<T>(phi::make_ddim(shape), platform::CPUPlace());
  for (int64_t i = 0; i < tensor->numel(); ++i) {
    ptr[i] = static_cast<T>(uniform_dist(rng)) - static_cast<T>(0.5);
  }
}

framework::OpDesc* CreateFusionGroupOp(
    framework::ProgramDesc* program,
    const std::vector<std::string>& input_names,
    const std::vector<std::vector<int64_t>>& input_shapes,
    const std::vector<std::string>& output_names,
    int type,
    std::string func_name) {
  EXPECT_EQ(input_names.size(), input_shapes.size());

  std::vector<int> input_dtypes(input_names.size(),
                                framework::proto::VarType::FP32);
  std::vector<int> output_dtypes(output_names.size(),
                                 framework::proto::VarType::FP32);

  for (size_t i = 0; i < input_names.size(); ++i) {
    auto* var = program->MutableBlock(0)->Var(input_names[i]);
    var->SetType(framework::proto::VarType::LOD_TENSOR);
    var->SetDataType(framework::proto::VarType::FP32);
    var->SetShape(input_shapes[i]);
  }
  for (size_t j = 0; j < output_names.size(); ++j) {
    auto* var = program->MutableBlock(0)->Var(output_names[j]);
    var->SetType(framework::proto::VarType::LOD_TENSOR);
    var->SetDataType(framework::proto::VarType::FP32);
  }

  auto* op = program->MutableBlock(0)->AppendOp();
  op->SetType("fusion_group");
  op->SetInput("Inputs", input_names);
  op->SetOutput("Outs", output_names);
  op->SetAttr("inputs_dtype", input_dtypes);
  op->SetAttr("outs_dtype", output_dtypes);
  op->SetAttr("type", type);
  op->SetAttr("func_name", func_name);
  op->SetAttr(framework::OpProtoAndCheckerMaker::OpRoleAttrName(),
              static_cast<int>(framework::OpRole::kForward));
  return op;
}

void PrepareDeviceCode(platform::Place place,
                       std::string func_name,
                       std::string cuda_kernel_str) {
  paddle::platform::DeviceCodePool& pool =
      paddle::platform::DeviceCodePool::Init({place});

  std::unique_ptr<paddle::platform::DeviceCode> code(
      new paddle::platform::CUDADeviceCode(place, func_name, cuda_kernel_str));
  code->Compile();
  pool.Set(std::move(code));
}

void CheckOutputs(framework::Scope* scope,
                  const std::vector<std::string>& output_names,
                  std::vector<phi::DenseTensor>* cpu_tensors,
                  size_t num_inputs,
                  CPUKernelFunc cpu_kernel_func) {
  std::vector<phi::DenseTensor> cpu_outputs;
  cpu_outputs.resize(output_names.size());
  for (size_t j = 0; j < output_names.size(); ++j) {
    auto* var = scope->Var(output_names[j]);
    const auto& dev_tensor = var->Get<framework::LoDTensor>();
    paddle::framework::TensorCopySync(
        dev_tensor, platform::CPUPlace(), &(cpu_outputs[j]));

    cpu_tensors->at(num_inputs + j)
        .mutable_data<float>(dev_tensor.dims(), platform::CPUPlace());
  }

  size_t n = cpu_tensors->at(0).numel();
  std::vector<void*> args;
  for (size_t i = 0; i < cpu_tensors->size(); ++i) {
    args.push_back(cpu_tensors->at(i).data<float>());
  }
  cpu_kernel_func(n, args);

  for (size_t j = 0; j < output_names.size(); ++j) {
    auto* dev_ptr = cpu_outputs[j].data<float>();
    auto* cpu_ptr = cpu_tensors->at(num_inputs + j).data<float>();
    int64_t length = cpu_outputs[j].numel();
    LOG(INFO) << "Check the " << j << "th output...";
    for (int64_t i = 0; i < length; ++i) {
      EXPECT_NEAR(dev_ptr[i], cpu_ptr[i], 1.E-05);
    }
  }
}

void TestMain(const std::vector<std::string>& input_names,
              const std::vector<std::vector<int64_t>>& input_shapes,
              const std::vector<std::string>& output_names,
              int type,
              std::string func_name,
              std::string cuda_kernel_str,
              CPUKernelFunc cpu_kernel_func) {
  // Compile the device code
  paddle::framework::InitDevices({0});
  platform::CUDAPlace place = platform::CUDAPlace(0);
  PrepareDeviceCode(place, func_name, cuda_kernel_str);

  // Create a ProgramDesc that has a fusion_group_op.
  framework::ProgramDesc program;
  framework::OpDesc* op_desc = CreateFusionGroupOp(
      &program, input_names, input_shapes, output_names, type, func_name);
  auto fusion_group_op = framework::OpRegistry::CreateOp(*op_desc);

  framework::Scope scope;

  // Prepare input tensors.
  std::vector<phi::DenseTensor> cpu_tensors;
  cpu_tensors.resize(input_names.size() + output_names.size());
  for (size_t i = 0; i < input_names.size(); ++i) {
    SetupRandomCPUTensor<float>(&(cpu_tensors[i]), input_shapes[i]);
    phi::DenseTensor* dev_tensor =
        CreateTensor<float>(&scope, place, input_names[i], input_shapes[i]);
    paddle::framework::TensorCopySync(cpu_tensors[i], place, dev_tensor);
  }
  // Create output tensors.
  std::vector<int64_t> empty_shape;
  for (size_t j = 0; j < output_names.size(); ++j) {
    CreateTensor<float>(&scope, place, output_names[j], empty_shape);
  }

  fusion_group_op->Run(scope, place);

  auto* dev_ctx = platform::DeviceContextPool::Instance().Get(place);
  dev_ctx->Wait();

  // Check the output.
  CheckOutputs(
      &scope, output_names, &cpu_tensors, input_names.size(), cpu_kernel_func);
}

TEST(FusionGroupOp, elementwise) {
  if (!platform::dynload::HasNVRTC() || !platform::dynload::HasCUDADriver()) {
    return;
  }

  // z = relu(x + y)
  std::vector<std::string> input_names = {"x", "y"};
  std::vector<std::string> output_names = {"z"};
  std::vector<std::vector<int64_t>> input_shapes = {{256, 256}, {256, 256}};
  constexpr auto kernel = R"(
static inline __device__ float relu(float x) {
  return x * (x > 0);
}

extern "C" __global__
void elementwise_cuda_kernel_0(size_t n, float *x, float* y, float* z) {
  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n;
       tid += blockDim.x * gridDim.x) {
    float tmp_0 = x[tid];
    float tmp_1 = y[tid];
    float tmp_2 = tmp_0 + tmp_1;
    float tmp_3 = relu(tmp_2);
    z[tid] = tmp_3;
  }
})";

  auto elementwise_cpu_kernel_0 = [](size_t n,
                                     std::vector<void*> args) -> void {
    float* x = static_cast<float*>(args[0]);
    float* y = static_cast<float*>(args[1]);
    float* z = static_cast<float*>(args[2]);
    for (size_t i = 0; i < n; ++i) {
      float tmp_0 = x[i];
      float tmp_1 = y[i];
      float tmp_2 = tmp_0 + tmp_1;
      float tmp_3 = tmp_2 > 0 ? tmp_2 : 0;
      z[i] = tmp_3;
    }
  };

  TestMain(input_names,
           input_shapes,
           output_names,
           0,
           "elementwise_cuda_kernel_0",
           kernel,
           elementwise_cpu_kernel_0);
}

}  // namespace operators
}  // namespace paddle

USE_CUDA_ONLY_OP(fusion_group);
