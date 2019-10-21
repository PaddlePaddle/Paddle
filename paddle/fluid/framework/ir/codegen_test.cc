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
#include <cmath>
#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/codegen_helper.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/operators/math.h"
#include "paddle/fluid/platform/device_code.h"
#include "paddle/fluid/platform/init.h"
#ifdef PADDLE_WITH_CUDA

TEST(codegen, cuda) {
  std::vector<int> mul_input{1, 2};
  std::vector<int> add_input{3, 4};
  std::vector<int> sub_input{5, 6};
  std::vector<int> relu_input{7};
  std::vector<int> sigmoid_input{8};

  int mul_out = 3;
  int add_out = 5;
  int sub_out = 7;
  int relu_out = 8;
  int sigmoid_out = 9;

  std::string op1 = "elementwise_mul";
  std::string op2 = "elementwise_add";
  std::string op3 = "elementwise_sub";
  std::string op4 = "relu";
  std::string op5 = "sigmoid";
  paddle::framework::ir::OperationExpression opexp1(mul_input, mul_out, op1);
  paddle::framework::ir::OperationExpression opexp2(add_input, add_out, op2);
  paddle::framework::ir::OperationExpression opexp3(sub_input, sub_out, op3);
  paddle::framework::ir::OperationExpression opexp4(relu_input, relu_out, op4);
  paddle::framework::ir::OperationExpression opexp5(sigmoid_input, sigmoid_out,
                                                    op5);

  std::vector<paddle::framework::ir::OperationExpression> fused_op = {
      opexp1, opexp2, opexp3, opexp4, opexp5};
  paddle::framework::ir::CodeTemplate code_template(
      paddle::framework::ir::kernel_elementwise_template);
  paddle::framework::ir::CodeGenerator codegen(code_template);
  paddle::framework::ir::TemplateVariable template_var;
  template_var.Add("$name", EmitUniqueName(fused_op));
  template_var.Add("$parameter", EmitDeclarationCode(fused_op, "float"));
  template_var.Add("$compute", EmitComputeCode(fused_op));
  std::string saxpy_code = codegen.GenerateCode(template_var);

  std::cout << saxpy_code << std::endl;
  paddle::framework::InitDevices(false, {0});
  paddle::platform::CUDAPlace place = paddle::platform::CUDAPlace(0);
  paddle::platform::CUDADeviceCode code(place, EmitUniqueName(fused_op),
                                        saxpy_code);

  paddle::framework::Tensor cpu_a;
  paddle::framework::Tensor cpu_b;
  paddle::framework::Tensor cpu_c;
  paddle::framework::Tensor cpu_d;
  paddle::framework::Tensor cpu_e;
  paddle::framework::Tensor cpu_f;
  paddle::framework::Tensor cpu_g;
  paddle::framework::Tensor cpu_h;
  paddle::framework::Tensor cpu_o;

  auto dims = paddle::framework::make_ddim(
      {static_cast<int64_t>(256), static_cast<int64_t>(1024)});
  cpu_a.mutable_data<float>(dims, paddle::platform::CPUPlace());
  cpu_b.mutable_data<float>(dims, paddle::platform::CPUPlace());
  cpu_c.mutable_data<float>(dims, paddle::platform::CPUPlace());
  cpu_d.mutable_data<float>(dims, paddle::platform::CPUPlace());
  cpu_e.mutable_data<float>(dims, paddle::platform::CPUPlace());
  cpu_f.mutable_data<float>(dims, paddle::platform::CPUPlace());
  cpu_g.mutable_data<float>(dims, paddle::platform::CPUPlace());
  cpu_o.mutable_data<float>(dims, paddle::platform::CPUPlace());

  size_t n = cpu_a.numel();
  for (size_t i = 0; i < n; ++i) {
    cpu_a.data<float>()[i] = static_cast<float>(i);
  }
  for (size_t i = 0; i < n; ++i) {
    cpu_b.data<float>()[i] = static_cast<float>(0.5);
    cpu_d.data<float>()[i] = static_cast<float>(10.0);
    cpu_f.data<float>()[i] = static_cast<float>(0.0);
  }

  paddle::framework::Tensor a;
  paddle::framework::Tensor b;
  paddle::framework::Tensor c;
  paddle::framework::Tensor d;
  paddle::framework::Tensor e;
  paddle::framework::Tensor f;
  paddle::framework::Tensor g;
  paddle::framework::Tensor h;
  paddle::framework::Tensor o;

  float* a_data = a.mutable_data<float>(dims, place);
  float* b_data = b.mutable_data<float>(dims, place);
  float* c_data = c.mutable_data<float>(dims, place);
  float* d_data = d.mutable_data<float>(dims, place);
  float* e_data = e.mutable_data<float>(dims, place);
  float* f_data = f.mutable_data<float>(dims, place);
  float* g_data = g.mutable_data<float>(dims, place);
  float* h_data = h.mutable_data<float>(dims, place);
  float* o_data = o.mutable_data<float>(dims, place);

  TensorCopySync(cpu_a, place, &a);
  TensorCopySync(cpu_b, place, &b);
  TensorCopySync(cpu_d, place, &d);
  TensorCopySync(cpu_f, place, &f);

  code.Compile();

  std::vector<void*> args = {&n,      &a_data, &b_data, &d_data, &f_data,
                             &c_data, &e_data, &g_data, &h_data, &o_data};
  code.SetNumThreads(1024);
  code.SetWorkloadPerThread(1);
  code.Launch(n, &args);

  TensorCopySync(o, paddle::platform::CPUPlace(), &cpu_o);
  for (size_t i = 0; i < n; i++) {
    float result =
        (1.0 / (1.0 + std::exp(-std::max(
                          0.0, static_cast<float>(i) * 0.5 + 10.0 - 0.0))));
    PADDLE_ENFORCE_EQ(cpu_o.data<float>()[i], result);
  }
}
#endif
