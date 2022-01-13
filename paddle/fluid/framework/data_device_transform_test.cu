/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/init.h"

#include "paddle/fluid/framework/pten_utils.h"

namespace paddle {
namespace framework {

template <typename T>
struct AddFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return a + b; }
};

class OpKernelTestProtoAndCheckerMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("input", "input1 of test op");
    AddOutput("output", "output of test op");
    AddAttr<bool>("use_gpu", "force to use gpu kernel").SetDefault(false);
    AddComment("This is test op");
  }
};

class TestOpWithKernel : public OperatorWithKernel {
 public:
  using OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {}
  OpKernelType GetExpectedKernelType(
      const ExecutionContext& ctx) const override {
    if (Attr<bool>("use_gpu")) {
      VLOG(3) << "force use gpu kernel";
      return OpKernelType(proto::VarType::FP32, platform::CUDAPlace(0));
    } else {
      VLOG(3) << "use default kernel";
      return OpKernelType(proto::VarType::FP32,
                          ctx.Input<Tensor>("input")->place());
    }
  }
};

template <typename DeviceContext, typename T>
class TestKernel : public OpKernel<float> {
 public:
  void Compute(const ExecutionContext& ctx) const {
    std::cout << ctx.DebugString() << std::endl;

    const Tensor* input = ctx.Input<Tensor>("input");

    std::cout << "input place:" << input->place() << std::endl;
    auto* output = ctx.Output<framework::LoDTensor>("output");
    output->Resize(input->dims());
    output->mutable_data<T>(ctx.GetPlace());

    auto pt_input = paddle::experimental::MakePtenDenseTensor(*input);
    auto pt_out = paddle::experimental::MakePtenDenseTensor(*output);

    pten::funcs::TransformFunctor<AddFunctor<T>, T, DeviceContext> functor(
        *pt_input, *pt_input, pt_out.get(),
        ctx.template device_context<DeviceContext>(), AddFunctor<T>());
    functor.Run();
  }
};

}  // namespace framework
}  // namespace paddle

REGISTER_OP_WITHOUT_GRADIENT(
    test_op, paddle::framework::TestOpWithKernel,
    paddle::framework::OpKernelTestProtoAndCheckerMaker);
REGISTER_OP_CPU_KERNEL(
    test_op,
    paddle::framework::TestKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CUDA_KERNEL(
    test_op,
    paddle::framework::TestKernel<paddle::platform::CUDADeviceContext, float>);

static void BuildVar(const std::string& param_name,
                     std::initializer_list<const char*> arguments,
                     paddle::framework::proto::OpDesc::Var* var) {
  var->set_parameter(param_name);
  for (auto& arg_name : arguments) {
    *var->mutable_arguments()->Add() = arg_name;
  }
}

TEST(Operator, CPUtoGPU) {
  paddle::framework::InitDevices();

  paddle::framework::Scope scope;
  paddle::platform::CPUPlace cpu_place;

  // create an op to run on CPU
  paddle::framework::proto::OpDesc cpu_op_desc;
  cpu_op_desc.set_type("test_op");
  BuildVar("input", {"IN1"}, cpu_op_desc.add_inputs());
  BuildVar("output", {"OUT1"}, cpu_op_desc.add_outputs());

  auto cpu_op = paddle::framework::OpRegistry::CreateOp(cpu_op_desc);
  // prepare input
  auto* in_t = scope.Var("IN1")->GetMutable<paddle::framework::LoDTensor>();
  auto* src_ptr =
      in_t->mutable_data<float>({2, 3}, paddle::platform::CPUPlace());
  for (int i = 0; i < 2 * 3; ++i) {
    src_ptr[i] = static_cast<float>(i);
  }

  // get output
  auto* output = scope.Var("OUT1");
  cpu_op->Run(scope, cpu_place);

  auto* output_ptr = output->Get<paddle::framework::LoDTensor>().data<float>();
  for (int i = 0; i < 2 * 3; ++i) {
    ASSERT_EQ(output_ptr[i], static_cast<float>(i) * 2);
  }

  // create an op to run on GPU
  paddle::framework::proto::OpDesc gpu_op_desc;
  gpu_op_desc.set_type("test_op");
  BuildVar("input", {"OUT1"}, gpu_op_desc.add_inputs());
  BuildVar("output", {"OUT2"}, gpu_op_desc.add_outputs());

  auto attr = gpu_op_desc.mutable_attrs()->Add();
  attr->set_name("use_gpu");
  attr->set_type(paddle::framework::proto::AttrType::BOOLEAN);
  attr->set_b(true);

  auto gpu_op = paddle::framework::OpRegistry::CreateOp(gpu_op_desc);

  paddle::platform::CUDAPlace cuda_place(0);
  // get output
  auto* output2 = scope.Var("OUT2");
  gpu_op->Run(scope, cuda_place);
  VLOG(3) << "after gpu_op run";

  // auto* output2_ptr = output2->Get<LoDTensor>().data<float>();
  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  auto dev_ctx = pool.Get(cuda_place);

  paddle::framework::Tensor output_tensor;
  paddle::framework::TensorCopy(output2->Get<paddle::framework::LoDTensor>(),
                                paddle::platform::CPUPlace(), *dev_ctx,
                                &output_tensor);

  dev_ctx->Wait();
  float* output2_ptr = output_tensor.data<float>();
  for (int i = 0; i < 2 * 3; ++i) {
    ASSERT_EQ(output2_ptr[i], static_cast<float>(i) * 4);
  }
}
