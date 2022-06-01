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

#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "paddle/fluid/framework/ir/fusion_group/code_generator.h"
#include "paddle/fluid/framework/ir/fusion_group/operation.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/platform/device_code.h"
#include "paddle/fluid/platform/float16.h"

namespace phi {
class DenseTensor;
}  // namespace phi

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

namespace paddle {
namespace framework {
namespace ir {
namespace fusion_group {

// relu
inline float relu(float x) { return x > 0 ? x : 0.; }

inline float relu_grad_dx(float x, float out, float dout) {
  return out > 0 ? dout : 0;
}

// sigmoid
inline float sigmoid(float x) { return 1.0 / (1.0 + std::exp(-x)); }

inline float sigmoid_grad_dx(float x, float out, float dout) {
  return dout * out * (1 - out);
}

// tanh
inline float tanh(float x) { return 2.0 / (1.0 + std::exp(-2 * x)) - 1.0; }

inline float tanh_grad_dx(float x, float out, float dout) {
  return dout * (1.0 - out * out);
}

// elementwise_add
inline float elementwise_add(float x, float y) { return x + y; }

inline float elementwise_add_grad_dx(float x, float y, float out, float dout) {
  return dout;
}

inline float elementwise_add_grad_dy(float x, float y, float out, float dout) {
  return dout;
}

// elementwise_sub
inline float elementwise_sub(float x, float y) { return x - y; }

inline float elementwise_sub_grad_dx(float x, float y, float out, float dout) {
  return dout;
}

inline float elementwise_sub_grad_dy(float x, float y, float out, float dout) {
  return -dout;
}

// elementwise_mul
inline float elementwise_mul(float x, float y) { return x * y; }

inline float elementwise_mul_grad_dx(float x, float y, float out, float dout) {
  return dout * y;
}

inline float elementwise_mul_grad_dy(float x, float y, float out, float dout) {
  return dout * x;
}

void CheckOutput(const std::vector<OperationExpression>& expressions,
                 const std::vector<LoDTensor> cpu_tensors,
                 const std::vector<int> input_ids_of_subgraph,
                 const std::vector<int> output_ids_of_subgraph, int i,
                 float eps) {
  std::vector<float> var(cpu_tensors.size());
  for (auto id : input_ids_of_subgraph) {
    if (id >= 0) {
      var[id] = cpu_tensors[id].data<float>()[i];
    }
  }

  for (auto expression : expressions) {
    std::string op_type = expression.GetOpType();
    auto input_ids = expression.GetInputIds();
    auto output_ids = expression.GetOutputIds();
    if (op_type == "relu") {
      var[output_ids[0]] = relu(var[input_ids[0]]);
    } else if (op_type == "sigmoid") {
      var[output_ids[0]] = sigmoid(var[input_ids[0]]);
    } else if (op_type == "tanh") {
      var[output_ids[0]] = tanh(var[input_ids[0]]);
    } else if (op_type == "elementwise_add") {
      var[output_ids[0]] =
          elementwise_add(var[input_ids[0]], var[input_ids[1]]);
    } else if (op_type == "elementwise_sub") {
      var[output_ids[0]] =
          elementwise_sub(var[input_ids[0]], var[input_ids[1]]);
    } else if (op_type == "elementwise_mul") {
      var[output_ids[0]] =
          elementwise_mul(var[input_ids[0]], var[input_ids[1]]);
    } else if (op_type == "relu_grad") {
      var[output_ids[0]] =
          relu_grad_dx(0, var[input_ids[1]], var[input_ids[2]]);
    } else if (op_type == "sigmoid_grad") {
      var[output_ids[0]] =
          sigmoid_grad_dx(0, var[input_ids[1]], var[input_ids[2]]);
    } else if (op_type == "tanh_grad") {
      var[output_ids[0]] =
          tanh_grad_dx(0, var[input_ids[1]], var[input_ids[2]]);
    } else if (op_type == "elementwise_add_grad") {
      var[output_ids[0]] = elementwise_add_grad_dx(0, 0, 0, var[input_ids[3]]);
      var[output_ids[1]] = elementwise_add_grad_dy(0, 0, 0, var[input_ids[3]]);
    } else if (op_type == "elementwise_mul_grad") {
      var[output_ids[0]] =
          elementwise_mul_grad_dx(0, var[input_ids[1]], 0, var[input_ids[3]]);
      var[output_ids[1]] =
          elementwise_mul_grad_dy(var[input_ids[0]], 0, 0, var[input_ids[3]]);
    }
  }

  for (auto id : output_ids_of_subgraph) {
    float actual = cpu_tensors[id].data<float>()[i];
    float expect = var[id];
    if (fabs(actual - expect) > eps) {
      LOG(INFO) << "Precision check failed from i = " << id
                << ", expect: " << expect << ", actual: " << actual;
      EXPECT_LT(fabs(actual - expect), eps);
    }
  }
}

template <typename T>
void SetupRandomCPUTensor(LoDTensor* tensor) {
  static unsigned int seed = 100;
  std::mt19937 rng(seed++);
  std::uniform_real_distribution<double> uniform_dist(0, 1);

  T* ptr = tensor->data<T>();
  EXPECT_NE(ptr, nullptr);
  for (int64_t i = 0; i < tensor->numel(); ++i) {
    ptr[i] = static_cast<T>(uniform_dist(rng)) - static_cast<T>(0.5);
  }
}

}  // namespace fusion_group
}  // namespace ir
}  // namespace framework
}  // namespace paddle

namespace fusion_group = paddle::framework::ir::fusion_group;

template <typename T>
void TestMainImpl(std::string func_name, std::string code_str,
                  std::vector<paddle::framework::LoDTensor> cpu_tensors, int n,
                  std::vector<int> input_ids, std::vector<int> output_ids) {
  bool is_float16 = std::type_index(typeid(T)) ==
                    std::type_index(typeid(paddle::platform::float16));

  paddle::platform::CUDAPlace place = paddle::platform::CUDAPlace(0);
  paddle::platform::CUDADeviceCode device_code(place, func_name, code_str);
#ifdef PADDLE_WITH_HIP
  device_code.Compile(true);
#else
  device_code.Compile(is_float16);
#endif

  std::vector<paddle::framework::LoDTensor> gpu_tensors(cpu_tensors.size());
  std::vector<paddle::framework::LoDTensor> tmp_cpu_tensors(cpu_tensors.size());

  std::vector<T*> gpu_ptrs(gpu_tensors.size());
  std::vector<void*> args;
  args.push_back(&n);

  for (auto id : input_ids) {
    if (id >= 0) {
      gpu_ptrs[id] =
          gpu_tensors[id].mutable_data<T>(cpu_tensors[id].dims(), place);
      fusion_group::SetupRandomCPUTensor<float>(&cpu_tensors[id]);
      if (is_float16) {
        paddle::platform::float16* tmp_cpu_ptr =
            tmp_cpu_tensors[id].mutable_data<paddle::platform::float16>(
                cpu_tensors[id].dims(), paddle::platform::CPUPlace());
        const float* cpu_ptr = cpu_tensors[id].data<float>();
        for (int64_t i = 0; i < cpu_tensors[id].numel(); ++i) {
          tmp_cpu_ptr[i] = paddle::platform::float16(cpu_ptr[i]);
        }
        paddle::framework::TensorCopySync(tmp_cpu_tensors[id], place,
                                          &gpu_tensors[id]);
      } else {
        paddle::framework::TensorCopySync(cpu_tensors[id], place,
                                          &gpu_tensors[id]);
      }
      args.push_back(&gpu_ptrs[id]);
    }
  }

  for (auto id : output_ids) {
    gpu_ptrs[id] =
        gpu_tensors[id].mutable_data<T>(cpu_tensors[id].dims(), place);
    args.push_back(&gpu_ptrs[id]);
  }

  device_code.SetNumThreads(1024);
  device_code.SetWorkloadPerThread(1);
  device_code.Launch(n, &args);

  auto* dev_ctx = reinterpret_cast<paddle::platform::CUDADeviceContext*>(
      paddle::platform::DeviceContextPool::Instance().Get(place));
  dev_ctx->Wait();

  // Copy the results back to CPU.
  for (auto id : output_ids) {
    if (is_float16) {
      paddle::platform::float16* tmp_cpu_ptr =
          tmp_cpu_tensors[id].mutable_data<paddle::platform::float16>(
              cpu_tensors[id].dims(), paddle::platform::CPUPlace());
      paddle::framework::TensorCopySync(
          gpu_tensors[id], paddle::platform::CPUPlace(), &tmp_cpu_tensors[id]);

      float* cpu_ptr = cpu_tensors[id].mutable_data<float>(
          cpu_tensors[id].dims(), paddle::platform::CPUPlace());
      for (int64_t i = 0; i < cpu_tensors[id].numel(); ++i) {
        cpu_ptr[i] = static_cast<float>(tmp_cpu_ptr[i]);
      }
    } else {
      paddle::framework::TensorCopySync(
          gpu_tensors[id], paddle::platform::CPUPlace(), &cpu_tensors[id]);
    }
  }
}

void TestElementwiseMain(
    std::string func_name, std::string code_str,
    std::vector<fusion_group::OperationExpression> expressions,
    std::vector<int> input_ids, std::vector<int> output_ids,
    std::string dtype) {
  std::unordered_set<int> ids;
  for (auto id : input_ids) {
    ids.insert(id);
  }
  for (auto id : output_ids) {
    ids.insert(id);
  }

  // Prepare CPU tensors which always hold float.
  std::vector<paddle::framework::LoDTensor> cpu_tensors(ids.size());
  auto dims =
      phi::make_ddim({static_cast<int64_t>(256), static_cast<int64_t>(1024)});
  for (size_t i = 0; i < cpu_tensors.size(); ++i) {
    cpu_tensors[i].mutable_data<float>(dims, paddle::platform::CPUPlace());
  }

  int n = cpu_tensors[0].numel();
  if (dtype == "__half") {
    TestMainImpl<paddle::platform::float16>(func_name, code_str, cpu_tensors, n,
                                            input_ids, output_ids);
  } else {
    TestMainImpl<float>(func_name, code_str, cpu_tensors, n, input_ids,
                        output_ids);
  }

  // Check the results
  float eps = (dtype == "__half") ? 1E-2 : 1E-5;
  for (int i = 0; i < n; i++) {
    fusion_group::CheckOutput(expressions, cpu_tensors, input_ids, output_ids,
                              i, eps);
  }
}

void TestMain(std::string func_name,
              std::vector<fusion_group::OperationExpression> expressions,
              std::vector<int> input_ids, std::vector<int> output_ids,
              std::string dtype) {
  fusion_group::OperationMap::Init();
  fusion_group::CodeGenerator code_generator;
  std::string code_str = code_generator.Generate(func_name, expressions);
  VLOG(3) << code_str;

  LOG(INFO) << "dtype: " << dtype;
  TestElementwiseMain(func_name, code_str, expressions, input_ids, output_ids,
                      dtype);
}

void TestMain(fusion_group::SubGraph* subgraph, std::vector<int> input_ids,
              std::vector<int> output_ids, std::string dtype) {
  fusion_group::OperationMap::Init();
  fusion_group::CodeGenerator code_generator;
  std::string code_str = code_generator.Generate(subgraph);
  VLOG(3) << code_str;

  // Need to check the accuracy according to expressions.
  std::vector<fusion_group::OperationExpression> expressions =
      code_generator.ConvertToExpressions(subgraph);

  TestElementwiseMain(subgraph->GetFuncName(), code_str, expressions, input_ids,
                      output_ids, dtype);
}

TEST(code_generator, elementwise) {
  for (std::string dtype : {"float", "__half"}) {
    // t2 = t0 * t1
    // t4 = t2 + t3
    // t6 = t4 - t5
    // t7 = relu(t6)
    // t8 = sigmoid(t7)
    fusion_group::OperationExpression exp1("elementwise_mul", {0, 1}, {2},
                                           dtype, dtype);
    fusion_group::OperationExpression exp2("elementwise_add", {2, 3}, {4},
                                           dtype, dtype);
    fusion_group::OperationExpression exp3("elementwise_sub", {4, 5}, {6},
                                           dtype, dtype);
    fusion_group::OperationExpression exp4("relu", {6}, {7}, dtype, dtype);
    fusion_group::OperationExpression exp5("sigmoid", {7}, {8}, dtype, dtype);
    std::vector<fusion_group::OperationExpression> expressions = {
        exp1, exp2, exp3, exp4, exp5};

    // Expressions:
    //  Op(elementwise_mul), inputs:{0,1}, outputs:{2}
    //  Op(elementwise_add), inputs:{2,3}, outputs:{4}
    //  Op(elementwise_sub), inputs:{4,5}, outputs:{6}
    //  Op(relu), inputs:{6}, outputs:{7}
    //  Op(sigmoid), inputs:{7}, outputs:{8}
    std::vector<int> input_ids = {0, 1, 3, 5};
    std::vector<int> output_ids = {2, 4, 6, 7, 8};
    TestMain("elementwise_kernel_0", expressions, input_ids, output_ids, dtype);
  }
}

TEST(code_generator, elementwise_grad) {
  for (std::string dtype : {"float", "__half"}) {
    // The var order: t0, t1, t2, t3, t0', t1', t2', t3'
    // t2 = t0 * t1
    // t3 = relu(t2)
    // t2' = relu_grad(t2, t3, t3')
    // t0', t1' = elementwise_mul_grad(t0, t1, t2, t2')
    fusion_group::OperationExpression exp1("relu_grad", {-1, 3, 7}, {6}, dtype,
                                           dtype);
    fusion_group::OperationExpression exp2("elementwise_mul_grad", {0, 1, 2, 6},
                                           {4, 5}, dtype, dtype);
    std::vector<fusion_group::OperationExpression> expressions = {exp1, exp2};

    // Expressions:
    //  Op(relu_grad), inputs:{2,3,7}, outputs:{6}
    //  Op(elementwise_mul_grad), inputs:{0,1,2,6}, outputs:{4,5}
    std::vector<int> input_ids = {0, 1, 2, 3, 7};
    std::vector<int> output_ids = {4, 5, 6};
    TestMain("elementwise_grad_kernel_0", expressions, input_ids, output_ids,
             dtype);
  }
}

std::unique_ptr<paddle::framework::ir::Graph> BuildGraph(bool backward,
                                                         std::string dtype) {
  // inputs                     operator            output
  // --------------------------------------------------------
  // x0                         sigmoid          -> tmp_0
  // (tmp_0, x1)                elementwise_mul  -> tmp_1
  // x2                         tanh             -> tmp_2
  // (x3, tmp_2)                elementwise_mul  -> tmp_3
  // (tmp_1, tmp_3)             elementwise_add  -> tmp_4
  //
  // Expression: tmp_4 = sigmoid(x0) * x1 + tanh(x2) * x3
  // The var order (their ids may be different):
  //  backward is false - x0(0), x1(1), x2(2), x3(3);
  //                    - tmp_0(4), tmp_2(5), tmp_3(6), tmp_1(7), tmp_4(8)
  //  backward is true  - tmp_1(0), tmp_4@GRAD(1), tmp_3(2), tmp_4(3),
  //                      tmp_2(4), x3(5), x1(6), tmp_0(7), x0(8), x2(9)
  //                    - tmp_3@GRAD(10), tmp_1@GRAD(11), tmp_0@GRAD(12),
  //                      tmp_2@GRAD(13), x2@GRAD(14), x0@GRAD(15),
  //                      x3@GRAD(16), x1@GRAD(17)
  paddle::framework::ir::Layers layers;
  std::vector<int64_t> shape = {16, 32};
  auto* x0 = layers.data("x0", shape);
  auto* tmp_0 = layers.sigmoid(x0);
  auto* x1 = layers.data("x1", shape);
  auto* tmp_1 = layers.elementwise_mul(tmp_0, x1);
  auto* x2 = layers.data("x2", shape);
  auto* tmp_2 = layers.tanh(x2);
  auto* x3 = layers.data("x3", shape);
  auto* tmp_3 = layers.elementwise_mul(x3, tmp_2);
  auto* tmp_4 = layers.elementwise_add(tmp_1, tmp_3);

  std::vector<paddle::framework::VarDesc*> elementwise_vars = {
      tmp_0, tmp_1, tmp_2, tmp_3, tmp_4};
  for (auto* var : elementwise_vars) {
    var->SetShape(shape);
  }

  if (backward) {
    layers.backward({tmp_4});
  }

  std::unique_ptr<paddle::framework::ir::Graph> graph(
      new paddle::framework::ir::Graph(layers.main_program()));
  auto proto_dtype = (dtype == "__half")
                         ? paddle::framework::proto::VarType::FP16
                         : paddle::framework::proto::VarType::FP32;
  for (auto* n : graph->Nodes()) {
    if (n && n->IsVar() && n->Var()) {
      n->Var()->SetDataType(proto_dtype);
    }
  }
  return graph;
}

std::unordered_set<paddle::framework::ir::Node*> DistilGradNodes(
    const std::unique_ptr<paddle::framework::ir::Graph>& graph) {
  auto is_grad_op = [&](paddle::framework::ir::Node* n) -> bool {
    if (n && n->IsOp() && n->Op()) {
      std::string suffix = "_grad";
      std::string op_type = n->Op()->Type();
      size_t pos = op_type.rfind(suffix);
      return pos != std::string::npos &&
             pos == (op_type.length() - suffix.length());
    }
    return false;
  };

  std::unordered_set<paddle::framework::ir::Node*> grad_nodes;
  for (auto* n : graph->Nodes()) {
    if (is_grad_op(n)) {
      grad_nodes.insert(n);
    } else if (n && n->IsVar() && n->Var()) {
      // Remove forward op nodes from inputs
      std::vector<paddle::framework::ir::Node*> inputs;
      for (auto* in : n->inputs) {
        if (in && in->IsOp() && in->Op() && is_grad_op(in)) {
          inputs.push_back(in);
        }
      }
      n->inputs = inputs;
      // Remove forward op nodes from outputs
      std::vector<paddle::framework::ir::Node*> outputs;
      for (auto* out : n->outputs) {
        if (out && out->IsOp() && out->Op() && is_grad_op(out)) {
          outputs.push_back(out);
        }
      }
      n->outputs = outputs;
      grad_nodes.insert(n);
    }
  }
  return grad_nodes;
}

TEST(code_generator, subgraph) {
  for (std::string dtype : {"float", "__half"}) {
    std::unique_ptr<paddle::framework::ir::Graph> graph =
        BuildGraph(false, dtype);
    fusion_group::SubGraph subgraph(0, "elementwise_kernel_1", true,
                                    graph->Nodes());

    // Expressions generated by code_generator (they may be different):
    //  Op(sigmoid), inputs:{0}, outputs:{4}
    //  Op(elementwise_mul), inputs:{4,1}, outputs:{7}
    //  Op(tanh), inputs:{2}, outputs:{5}
    //  Op(elementwise_mul), inputs:{3,5}, outputs:{6}
    //  Op(elementwise_add), inputs:{7,6}, outputs:{8}
    std::vector<int> input_ids = {0, 1, 2, 3};
    std::vector<int> output_ids = {4, 5, 6, 7, 8};
    TestMain(&subgraph, input_ids, output_ids, dtype);
  }
}

TEST(code_generator, subgraph_grad) {
  for (std::string dtype : {"float", "__half"}) {
    std::unique_ptr<paddle::framework::ir::Graph> graph =
        BuildGraph(true, dtype);
    fusion_group::SubGraph subgraph(0, "elementwise_grad_kernel_1", true,
                                    DistilGradNodes(graph));

    // Expressions generated by code_generator (they may be different):
    //  Op(elementwise_add_grad), inputs:{1,2,3,0}, outputs:{11,10}
    //  Op(elementwise_mul_grad), inputs:{5,4,2,10}, outputs:{17,13}
    //  Op(elementwise_mul_grad), inputs:{7,6,1,11}, outputs:{12,15}
    //  Op(sigmoid_grad), inputs:{8,7,12}, outputs:{16}
    //  Op(tanh_grad), inputs:{9,4,13}, outputs:{14}
    std::vector<int> input_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int> output_ids = {10, 11, 12, 13, 14, 15, 16, 17};
    TestMain(&subgraph, input_ids, output_ids, dtype);
  }
}
#endif
