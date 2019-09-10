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
#include "paddle/fluid/framework/ir/codegen.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/codegen_helper.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/device_code.h"
#include "paddle/fluid/platform/init.h"
#ifdef PADDLE_WITH_CUDA

TEST(codegen, cuda) {
  std::vector<int> mul_input{1, 2};
  std::vector<int> add_input{3, 4};
  int mul_out = 3;
  int add_out = 5;

  std::string op1 = "elementwise_mul";
  std::string op2 = "elementwise_add";
  paddle::framework::ir::OperationExpression opexp1(mul_input, mul_out, op1);
  paddle::framework::ir::OperationExpression opexp2(add_input, add_out, op2);

  std::vector<paddle::framework::ir::OperationExpression> fused_op = {opexp1,
                                                                      opexp2};
  paddle::framework::ir::CodeTemplate code_template(
      paddle::framework::ir::kernel);
  paddle::framework::ir::CodeGen codegen(code_template);
  paddle::framework::ir::TemplateVariable template_var;
  template_var.add("$kernel_name", EmitUniqueName(fused_op));
  template_var.add("$kernel_parameter", EmitDeclarationCode(fused_op, "float"));
  template_var.add("$kernel_compute", EmitComputeCode(fused_op));
  std::string saxpy_code = codegen.GetKernelCode(template_var);

  std::cout << saxpy_code << std::endl;
  paddle::framework::InitDevices(false, {0});
  paddle::platform::CUDAPlace place = paddle::platform::CUDAPlace(0);
  paddle::platform::CUDADeviceCode code(place, "fused_kernel35", saxpy_code);

  paddle::framework::Tensor cpu_x;
  paddle::framework::Tensor cpu_y;
  paddle::framework::Tensor cpu_z;
  paddle::framework::Tensor cpu_a;
  paddle::framework::Tensor cpu_b;

  auto dims = paddle::framework::make_ddim(
      {static_cast<int64_t>(256), static_cast<int64_t>(1024)});
  cpu_x.mutable_data<float>(dims, paddle::platform::CPUPlace());
  cpu_y.mutable_data<float>(dims, paddle::platform::CPUPlace());
  cpu_a.mutable_data<float>(dims, paddle::platform::CPUPlace());

  size_t n = cpu_x.numel();
  for (size_t i = 0; i < n; ++i) {
    cpu_x.data<float>()[i] = static_cast<float>(i);
  }
  for (size_t i = 0; i < n; ++i) {
    cpu_y.data<float>()[i] = static_cast<float>(0.5);
    cpu_a.data<float>()[i] = static_cast<float>(1.0);
  }

  paddle::framework::Tensor x;
  paddle::framework::Tensor y;
  paddle::framework::Tensor z;
  paddle::framework::Tensor a;
  paddle::framework::Tensor b;

  float* x_data = x.mutable_data<float>(dims, place);
  float* y_data = y.mutable_data<float>(dims, place);
  float* z_data = z.mutable_data<float>(dims, place);
  float* a_data = a.mutable_data<float>(dims, place);
  float* b_data = b.mutable_data<float>(dims, place);

  TensorCopySync(cpu_x, place, &x);
  TensorCopySync(cpu_y, place, &y);
  TensorCopySync(cpu_a, place, &a);

  code.Compile();

  std::vector<void*> args = {&n, &x_data, &y_data, &a_data, &z_data, &b_data};
  code.SetNumThreads(1024);
  code.SetWorkloadPerThread(1);
  code.Launch(n, &args);

  TensorCopySync(b, paddle::platform::CPUPlace(), &cpu_b);
  for (size_t i = 0; i < n; i++) {
    PADDLE_ENFORCE_EQ(cpu_b.data<float>()[i],
                      static_cast<float>(i) * 0.5 + 1.0);
  }
}
#endif
