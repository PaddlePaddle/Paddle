// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <string>

#include "paddle/cinn/backends/codegen_cuda_dev.h"
#include "paddle/cinn/backends/codegen_cuda_host.h"
#include "paddle/cinn/backends/codegen_cuda_util.h"
#include "paddle/cinn/backends/cuda_util.h"
#include "paddle/cinn/backends/llvm/execution_engine.h"
#include "paddle/cinn/backends/llvm/runtime_symbol_registry.h"
#include "paddle/cinn/backends/llvm/simple_jit.h"
#include "paddle/cinn/backends/nvrtc/nvrtc_util.h"
#include "paddle/cinn/cinn.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/common/test_helper.h"
#include "paddle/cinn/hlir/framework/graph_compiler.h"
#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/op/use_ops.h"
#include "paddle/cinn/hlir/pe/nn.h"
#include "paddle/cinn/runtime/cinn_runtime.h"
#include "paddle/cinn/runtime/cuda/cuda_module.h"
namespace cinn {
namespace hlir {
namespace framework {

using common::_CINNValuePack_;
using common::CINNValue;
using common::CINNValuePack;
using framework::OpStrategy;
using framework::shape_t;
using framework::StrategyFunction;
using runtime::cuda::CUDAModule;

std::pair<ir::Module, std::string> GenReduceCode(
    const std::vector<int>& shape,
    const std::vector<int>& dim,
    const std::string& func_name,
    bool keep_dim = false,
    const std::string& op_name = "reduce_sum") {
  // code gen
  Context::Global().ResetNameId();
  auto reduce_sum = Operator::Get(op_name);
  auto strategy =
      Operator::GetAttrs<StrategyFunction>("CINNStrategy")[reduce_sum];

  // input tensor
  std::vector<Expr> shape_as_expr;
  for (auto value : shape) {
    shape_as_expr.emplace_back(value);
  }
  Placeholder<float> X("X", shape_as_expr);

  // set attrs
  NodeAttr attrs;
  attrs.attr_store["dim"] = dim;
  attrs.attr_store["keep_dim"] = keep_dim;
  std::vector<ir::Tensor> inputs{X.tensor()};
  std::vector<Type> out_type{Float(32)};

  std::vector<int> output_shape;
  for (int idx = 0; idx < shape.size(); ++idx) {
    if (std::find(dim.begin(), dim.end(), idx) != dim.end()) {
      if (keep_dim) {
        output_shape.push_back(1);
      }
    } else {
      output_shape.push_back(shape[idx]);
    }
  }

  auto target = common::DefaultNVGPUTarget();
  auto impl = OpStrategy::SelectImpl(
      strategy(attrs, inputs, out_type, {output_shape}, target));

  std::vector<ir::LoweredFunc> func;
  std::vector<std::string> input_output_nodes{"X", op_name};
  func = GetFuncFromImpl(
      impl,
      common::CINNValuePack{{common::CINNValue(X), common::CINNValue(op_name)}},
      inputs,
      input_output_nodes,
      func_name,
      target);

  Module::Builder builder(func_name + "_builder", target);
  for (auto& f : func) {
    builder.AddFunction(f);
  }
  // compile the module
  // Need to create a new compiler for every call of Build,
  // because the underneath jit engine does't support addIRModule repeatedly
  // now.
  auto module = builder.Build();
  auto host_module_device_module =
      backends::SplitCudaAndHostModule(module);  // NOLINT
  auto& host_module = std::get<0>(host_module_device_module);
  auto& device_module = std::get<1>(host_module_device_module);

  backends::CodeGenCUDA_Dev codegen(target);
  std::string source_code;
  source_code = codegen.Compile(device_module);
  // LOG(INFO) << "compiled code:\n" << device_module;

  return std::pair<ir::Module, std::string>(host_module, source_code);
}

// last dimension not in reduce
TEST(Operator, Operator_Reduce_Without_Last_Channel_Case_5) {
  std::vector<int> shape = {128, 112, 112, 128};
  std::vector<int> dim = {0, 1, 2};

  GenReduceCode(shape, dim, "Reduce_Without_Last_Channel_Case_5");
}

// last dimension not in reduce
TEST(Operator, Operator_Reduce_Without_Last_Channel_Case_4) {
  std::vector<int> shape = {16, 16, 8, 8, 16, 16};
  std::vector<int> dim = {0, 2, 3};

  GenReduceCode(shape, dim, "Reduce_Without_Last_Channel_Case_4");
}
// case 3
TEST(Operator, Operator_Reduce_Without_Last_Channel_Case_3) {
  std::vector<int> shape = {16, 16, 16, 16, 16};
  std::vector<int> dim = {0, 2};

  GenReduceCode(shape, dim, "Reduce_Without_Last_Channel_Case_3");
}
// case 2
TEST(Operator, Operator_Reduce_Without_Last_Channel_Case_2) {
  std::vector<int> shape = {16, 16, 16, 16};
  std::vector<int> dim = {0, 1};

  GenReduceCode(shape, dim, "Reduce_Without_Last_Channel_Case_2");
}
// case 1
TEST(Operator, Operator_Reduce_Without_Last_Channel_Case_1) {
  std::vector<int> shape = {16, 16, 16, 16};
  std::vector<int> dim = {1};

  GenReduceCode(shape, dim, "Reduce_Without_Last_Channel_Case_1");
}
// case 0
TEST(Operator, Operator_Reduce_Without_Last_Channel_Case_0) {
  std::vector<int> shape = {16, 16, 32};
  std::vector<int> dim = {1};

  GenReduceCode(shape, dim, "Reduce_Without_Last_Channel_Case_0");
}

TEST(Operator, Operator_Reduction_Case_Last_Dim_1) {
  std::vector<int> shape = {10, 100, 1};
  std::vector<int> dim = {0, 2};

  GenReduceCode(shape, dim, "reduce_cast_with_last_dim_1");
}

TEST(Operator, Operator_Reduction_Case_0) {
  std::vector<int> shape = {16, 16, 8, 16};
  std::vector<int> dim = {2, 3};

  GenReduceCode(shape, dim, "reduce_cast_0");
}

TEST(Operator, Operator_Reduction_Case_0_0) {
  std::vector<int> shape = {16, 16, 8, 16};
  std::vector<int> dim = {2, 3};

  GenReduceCode(shape, dim, "reduce_cast_0_0", true);
}

TEST(Operator, Operator_Reduction_Case_1) {
  std::vector<int> shape = {16, 16, 32, 32};
  std::vector<int> dim = {2, 3};

  GenReduceCode(shape, dim, "reduce_cast_1");
}

TEST(Operator, Operator_Reduction_Case_1_1) {
  std::vector<int> shape = {16, 16, 32, 32};
  std::vector<int> dim = {2, 3};

  GenReduceCode(shape, dim, "reduce_cast_1_1", true);
}

TEST(Operator, Operator_Reduction_Case_2) {
  std::vector<int> shape = {16, 16, 32, 32};
  std::vector<int> dim = {1};

  GenReduceCode(shape, dim, "reduce_cast_2", true);
}

TEST(Operator, Operator_Reduction_Case_2_1) {
  std::vector<int> shape = {16, 16, 32, 32};
  std::vector<int> dim = {-1};

  GenReduceCode(shape, dim, "reduce_cast_2_1", true);
}

TEST(Operator, Operator_Reduction_Case_3) {
  std::vector<int> shape = {16, 16, 64, 64};
  std::vector<int> dim = {1};

  GenReduceCode(shape, dim, "reduce_cast_3");
}

TEST(Operator, Operator_Reduction_Case_4) {
  std::vector<int> shape = {16, 16, 16, 16};
  std::vector<int> dim = {0, 2, 3};

  GenReduceCode(shape, dim, "reduce_cast_4");
}

TEST(Operator, Operator_Reduction_Case_4_4) {
  std::vector<int> shape = {16, 16, 16, 16};
  std::vector<int> dim = {0, 2, 3};

  GenReduceCode(shape, dim, "reduce_cast_4_4", true);
}

TEST(Operator, Operator_Reduction_Case_5) {
  std::vector<int> shape = {16, 16, 16, 16, 16, 32};
  std::vector<int> dim = {1, 3, 5};

  GenReduceCode(shape, dim, "reduce_cast_5");
}

TEST(Operator, Operator_Reduction_Case_5_5) {
  std::vector<int> shape = {16, 16, 16, 16, 16, 32};
  std::vector<int> dim = {1, 3, 5};

  GenReduceCode(shape, dim, "reduce_cast_5_5", true);
}

TEST(Operator, Operator_Reduction_Case_6_0) {
  std::vector<int> shape = {32, 32, 32};
  std::vector<int> dim = {0, 1, 2};

  GenReduceCode(shape, dim, "reduce_cast_6_0", false);
}

TEST(Operator, Operator_Reduction_Case_6_00) {
  std::vector<int> shape = {32, 32, 32, 32};
  std::vector<int> dim = {0, 1, 2};

  GenReduceCode(shape, dim, "reduce_cast_6_00", false);
}

TEST(Operator, Operator_Reduction_Case_6_10) {
  std::vector<int> shape = {32, 32, 32};
  std::vector<int> dim = {-2, -1, 0};

  GenReduceCode(shape, dim, "reduce_cast_6_10", true);
}

struct SumOp {
  float operator()(const float left, const float right) { return left + right; }
};
struct ProdOp {
  float operator()(const float left, const float right) { return left * right; }
};
struct MaxOp {
  float operator()(const float left, const float right) {
    return std::max(left, right);
  }
};
struct MinOp {
  float operator()(const float left, const float right) {
    return std::min(left, right);
  }
};

template <class Op>
void DoCpuReduce(const float* x,
                 std::vector<float>* sum0,
                 std::vector<float>* sum1,
                 const float init_val,
                 const int n,
                 const int c,
                 const int h,
                 const int w) {
  for (auto& val : *sum0) {
    val = init_val;
  }
  for (auto& val : *sum1) {
    val = init_val;
  }

  for (int idx = 0; idx < n; ++idx) {
    for (int idy = 0; idy < c; ++idy) {
      for (int idz = 0; idz < h; ++idz) {
        for (int ida = 0; ida < w; ++ida) {
          sum0->at(idy * w + ida) +=
              Op()(sum0->at(idy * w + ida),
                   x[idx * c * h * w + idy * h * w + idz * w + ida]);
          sum1->at(idy) = Op()(
              sum1->at(idy), x[idx * c * h * w + idy * h * w + idz * w + ida]);
        }
      }
    }
  }
}

template <class Op>
void TestCaseForReduce(const float init_val,
                       int n,
                       int c,
                       int h,
                       int w,
                       const std::string& test_name,
                       const std::string& op_name) {
  std::vector<int> shape = {n, c, h, w};
  std::vector<int> dim = {0, 2, 3};

  // get source code
  auto source_code =
      GenReduceCode(shape, dim, test_name, false, op_name).second;

  // nv jit compile to ptx
  backends::nvrtc::Compiler compiler;
  auto ptx = compiler(source_code);
  CHECK(!ptx.empty());

  // cuda_module load ptx
  runtime::cuda::CUDAModule cuda_module(ptx, CUDAModule::Kind::PTX);

  srand(time(NULL));
  CUDA_CALL(cudaSetDevice(0));

  // auto func_0   = reinterpret_cast<void (*)(cinn_pod_value_t*,
  // int)>(fn_reduce_sum);
  auto buffer_x =
      common::BufferBuilder(Float(32), {n, c, h, w}).set_random().Build();
  auto buffer_z = common::BufferBuilder(Float(32), {c}).set_random().Build();

  void *dev_x = nullptr, *dev_z = nullptr;
  CUDA_CALL(cudaMalloc(&dev_x, buffer_x->memory_size));
  CUDA_CALL(cudaMalloc(&dev_z, buffer_z->memory_size));
  CUDA_CALL(cudaMemcpy(
      dev_x, buffer_x->memory, buffer_x->memory_size, cudaMemcpyHostToDevice));
  dim3 grid;
  dim3 block;
  grid = {c, 1, 1};
  int block_dim_x = n * w * h > 1024 ? 1024 : n * w * h;
  block = {block_dim_x, 1, 1};

  void* args[] = {&dev_x, &dev_z};
  std::string new_test_name = "fn_" + test_name + "_kernel";
  cuda_module.LaunchKernel(0, new_test_name, grid, block, args);
  CUDA_CALL(cudaMemcpy(
      buffer_z->memory, dev_z, buffer_z->memory_size, cudaMemcpyDeviceToHost));

  std::vector<float> sum0(c * w);
  std::vector<float> sum1(c);
  DoCpuReduce<Op>(reinterpret_cast<float*>(buffer_x->memory),
                  &sum0,
                  &sum1,
                  init_val,
                  n,
                  c,
                  h,
                  w);

  std::vector<std::pair<std::vector<float>, float*>> results = {
      {sum1, reinterpret_cast<float*>(buffer_z->memory)}};
  for (auto& res : results) {
    for (int idx = 0; idx < res.first.size(); ++idx) {
      ASSERT_LT(abs(res.first[idx] - res.second[idx]) / res.first[idx], 1e-4);
    }
  }

  CUDA_CALL(cudaFree(dev_x));
  CUDA_CALL(cudaFree(dev_z));
}

TEST(Operator, Operator_Reduction_Case_6_1) {
  TestCaseForReduce<SumOp>(
      0.0f, 32, 32, 32, 32, "Operator_Reduction_Case_6_1", "reduce_sum");
}
TEST(Operator, Operator_Reduction_Case_6_2) {
  TestCaseForReduce<ProdOp>(
      1.0f, 1, 1, 1, 32, "Operator_Reduction_Case_6_2", "reduce_prod");
}
TEST(Operator, Operator_Reduction_Case_6_3) {
  TestCaseForReduce<MaxOp>(
      -1e38f, 32, 32, 32, 32, "Operator_Reduction_Case_6_3", "reduce_max");
}
TEST(Operator, Operator_Reduction_Case_6_4) {
  TestCaseForReduce<MinOp>(
      1e38f, 32, 32, 32, 32, "Operator_Reduction_Case_6_4", "reduce_min");
}
TEST(Operator, Operator_Reduction_Case_7) {
  int n = 32, c = 32, h = 16, w = 16;
  std::vector<int> shape = {n, c, h, w};
  std::vector<int> dim = {0, 1};

  std::string func_name = "reduce_cast_7";
  // get source code
  auto host_source = GenReduceCode(shape, dim, func_name);

  // compile to ptx
  backends::nvrtc::Compiler compiler;
  auto ptx = compiler(host_source.second);
  CHECK(!ptx.empty());

  // load ptx
  CUDA_CALL(cudaSetDevice(0));
  runtime::cuda::CUDAModule cuda_module(ptx,
                                        runtime::cuda::CUDAModule::Kind::PTX);
  std::string new_func_name = "fn_" + func_name;
  void* reduce_sum_kernel =
      cuda_module.GetFunction(0, new_func_name + "_kernel");
  CHECK(reduce_sum_kernel);

  // register cufunction and stream
  void* stream = nullptr;
  backends::GlobalSymbolRegistry::Global().RegisterFn(
      new_func_name + "_kernel_ptr_",
      reinterpret_cast<void*>(&reduce_sum_kernel));

  // gen host code
  auto jit = backends::SimpleJIT::Create();
  jit->Link<backends::CodeGenCUDA_Host>(host_source.first);

  auto fn_reduce_sum = jit->Lookup(new_func_name);
  CHECK(fn_reduce_sum);

  auto func_0 = reinterpret_cast<void (*)(void*, int, void*)>(fn_reduce_sum);

  srand(time(NULL));
  auto buffer_x =
      common::BufferBuilder(Float(32), {n, c, h, w}).set_random().Build();
  auto buffer_y = common::BufferBuilder(Float(32), {h, w}).set_random().Build();

  void *dev_x = nullptr, *dev_y = nullptr;
  CUDA_CALL(cudaMalloc(&dev_x, buffer_x->memory_size));
  CUDA_CALL(cudaMalloc(&dev_y, buffer_y->memory_size));

  CUDA_CALL(cudaMemcpy(
      dev_x, buffer_x->memory, buffer_x->memory_size, cudaMemcpyHostToDevice));

  cinn_buffer_t _x;
  cinn_buffer_t _y;

  _x.memory = static_cast<uint8_t*>(dev_x);
  _y.memory = static_cast<uint8_t*>(dev_y);

  _x.memory_size = buffer_x->memory_size;
  _y.memory_size = buffer_y->memory_size;

  cinn_pod_value_t x_arg(&_x), y_arg(&_y);
  cinn_pod_value_t args0[] = {x_arg, y_arg};

  func_0(args0, 2, stream);
  CUDA_CALL(cudaMemcpy(
      buffer_y->memory, dev_y, buffer_y->memory_size, cudaMemcpyDeviceToHost));

  CUDA_CALL(cudaFree(dev_x));
  CUDA_CALL(cudaFree(dev_y));
}

TEST(Operator, Operator_Reduction_Case_8) {
  std::vector<int> shape = {128, 1};
  std::vector<int> dim = {0};

  GenReduceCode(shape, dim, "Operator_Reduction_Case_8");
}

TEST(Operator, Operator_Reduction_Case_88) {
  std::vector<int> shape = {128, 1};
  std::vector<int> dim = {0};

  GenReduceCode(shape, dim, "Operator_Reduction_Case_88", true);
}

TEST(Operator, Operator_Reduction_Case_9) {
  std::vector<int> shape = {2560, 1};
  std::vector<int> dim = {0};

  GenReduceCode(shape, dim, "Operator_Reduction_Case_9");
}

TEST(Operator, Operator_Reduction_Case_99) {
  std::vector<int> shape = {2560, 1};
  std::vector<int> dim = {0};

  GenReduceCode(shape, dim, "Operator_Reduction_Case_99", true);
}

TEST(Operator, Operator_Reduction_Case_10) {
  std::vector<int> shape = {16, 2560, 1};
  std::vector<int> dim = {1};

  GenReduceCode(shape, dim, "Operator_Reduction_Case_10");
}

TEST(Operator, Operator_Reduction_Case_11) {
  std::vector<int> shape = {16, 128, 128, 1};
  std::vector<int> dim = {1, 2};

  GenReduceCode(shape, dim, "Operator_Reduction_Case_11");
}

TEST(Operator, Operator_Reduction_Case_Warp_Reduce) {
  int sm_count = common::DefaultNVGPUTarget().get_multi_processor_count();
  int max_threads_per_sm =
      common::DefaultNVGPUTarget().get_max_threads_per_sm();
  int warp_reduce_threshold = sm_count * max_threads_per_sm / 32;

  std::vector<int> shape = {warp_reduce_threshold + 10, 256};
  std::vector<int> dim = {1};

  auto res = GenReduceCode(shape, dim, "Operator_Reduction_Case_Warp_Reduce");
  CHECK(res.second.find("threadIdx.x < 32") != std::string::npos);
}

TEST(Operator, Operator_Reduction_Case_Block_Reduce) {
  int sm_count = common::DefaultNVGPUTarget().get_multi_processor_count();
  int max_threads_per_sm =
      common::DefaultNVGPUTarget().get_max_threads_per_sm();
  int warp_reduce_threshold = sm_count * max_threads_per_sm / 32;

  std::vector<int> shape = {warp_reduce_threshold - 10, 33};
  std::vector<int> dim = {1};

  auto res = GenReduceCode(shape, dim, "Operator_Reduction_Case_Block_Reduce");
  CHECK(res.second.find("threadIdx.x < 32") == std::string::npos);
}

TEST(Operator, Operator_Reduction_Case_Warp_Reduce_Case_1) {
  int sm_count = common::DefaultNVGPUTarget().get_multi_processor_count();
  int max_threads_per_sm =
      common::DefaultNVGPUTarget().get_max_threads_per_sm();
  int warp_reduce_threshold = sm_count * max_threads_per_sm / 32;

  std::vector<int> shape = {(warp_reduce_threshold + 32) / 2, 2, 10, 256};
  std::vector<int> dim = {2, 3};

  auto res =
      GenReduceCode(shape, dim, "Operator_Reduction_Case_Warp_Reduce_Case_1");
  CHECK(res.second.find("threadIdx.x < 32") != std::string::npos);
}

TEST(Operator, Operator_Reduction_Case_Block_Reduce_Case_1) {
  int sm_count = common::DefaultNVGPUTarget().get_multi_processor_count();
  int max_threads_per_sm =
      common::DefaultNVGPUTarget().get_max_threads_per_sm();
  int warp_reduce_threshold = sm_count * max_threads_per_sm / 32;

  std::vector<int> shape = {(warp_reduce_threshold - 32) / 2, 2, 10, 33};
  std::vector<int> dim = {2, 3};

  auto res =
      GenReduceCode(shape, dim, "Operator_Reduction_Case_Block_Reduce_Case_2");
  CHECK(res.second.find("threadIdx.x < 32") == std::string::npos);
}
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
