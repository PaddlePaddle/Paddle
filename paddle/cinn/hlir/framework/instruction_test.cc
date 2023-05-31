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

#include "paddle/cinn/hlir/framework/instruction.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/cinn/backends/llvm/simple_jit.h"
#include "paddle/cinn/common/test_helper.h"
#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/op/use_ops.h"
#include "paddle/cinn/runtime/cinn_runtime.h"

namespace cinn {
namespace hlir {
namespace framework {

std::unique_ptr<backends::SimpleJIT> GetLoweredFunc(int M, int N) {
  Expr m(M);
  Expr n(N);

  Placeholder<float> x("x", {m, n});
  Placeholder<float> y("y", {m, n});

  auto z = Compute(
      {m, n}, [=](Expr i, Expr j) { return x(i, j) + y(i, j); }, "z");

  auto stages = CreateStages({z});
  auto fn = Lower("fn", stages, {x, y, z});

  ir::Module::Builder builder("some_module", common::DefaultHostTarget());
  builder.AddFunction(fn);

  auto jit = backends::SimpleJIT::Create();
  jit->Link(builder.Build());
  return std::move(jit);
}

void InstantiateScope(int M, int N, Scope* scope) {
  for (auto& name : std::vector<std::string>({"x", "y", "z"})) {
    auto* var = scope->Var<Tensor>(name);
    auto& tensor = absl::get<Tensor>(*var);
    tensor->Resize(Shape{{M, N}});
    auto* data = tensor->mutable_data<float>(common::DefaultHostTarget());
    for (int i = 0; i < M * N; i++) {
      data[i] = (rand() * 1.f) / RAND_MAX;  // NOLINT
    }
  }
}

TEST(Instruction, basic) {
  const int M = 10;
  const int N = 20;

  Scope scope;
  InstantiateScope(M, N, &scope);
  // create Instruction
  Instruction instr(common::DefaultHostTarget(), &scope, {"x", "y"}, {"z"});
  auto jit = GetLoweredFunc(M, N);
  auto fn_ptr = jit->Lookup("fn");
  CHECK(fn_ptr);
  instr.SetLoweredFunc(reinterpret_cast<void*>(fn_ptr));
  // should call Finalize explicitly before Run
  ASSERT_DEATH(instr.Run(), "");
  instr.Finalize();
  instr.Run();

  // check result
  {
    auto* xd = scope.GetTensor("x")->data<float>();
    auto* yd = scope.GetTensor("y")->data<float>();
    auto* zd = scope.GetTensor("z")->data<float>();

    for (int i = 0; i < M * N; i++) {
      LOG_FIRST_N(INFO, 3) << "data: " << xd[i] << " + " << yd[i] << " = "
                           << zd[i];
      ASSERT_NEAR(xd[i] + yd[i], zd[i], 1e-5);
    }
  }
}

TEST(Instruction, RunWithRawPodArgs) {
  const int M = 10;
  const int N = 20;
  const auto& shape = Shape({M, N});

  std::map<std::string, cinn_pod_value_t> name2podargs;
  // case 1: create cinn_pod_value_t arguments dicrectly
  std::vector<cinn_buffer_t> args_buffer(
      3);  // store {"x", "y", "z"} buffer objects
  auto* default_memory_mng =
      MemoryManager::Global().RetrieveSafely(common::DefaultHostTarget().arch);

  int count = 0;
  for (const auto& name : std::vector<std::string>({"x", "y", "z"})) {
    auto* buffer = &args_buffer.at(count++);
    buffer->resize(
        reinterpret_cast<const cinn_dimension_t*>(shape.data().data()),
        shape.size());
    buffer->memory = reinterpret_cast<uint8_t*>(
        default_memory_mng->malloc(shape.numel() * sizeof(float)));
    auto* data = reinterpret_cast<float*>(buffer->memory);
    for (int i = 0; i < M * N; i++) {
      data[i] = (rand() * 1.f) / RAND_MAX;  // NOLINT
    }
    name2podargs.emplace(name, buffer);
  }

  // create Instruction
  auto jit = GetLoweredFunc(M, N);
  auto fn_ptr = jit->Lookup("fn");
  CHECK(fn_ptr);
  Instruction instr(
      common::DefaultHostTarget(), nullptr, {"x", "y"}, {"z"});  // empty scope
  instr.SetLoweredFunc(reinterpret_cast<void*>(fn_ptr));
  instr.Finalize();

  auto check_equal_by_element = [&]() {
    auto xd = reinterpret_cast<float*>(
        cinn_pod_value_to_buffer_p(&name2podargs.at("x"))->memory);
    auto yd = reinterpret_cast<float*>(
        cinn_pod_value_to_buffer_p(&name2podargs.at("y"))->memory);
    auto zd = reinterpret_cast<float*>(
        cinn_pod_value_to_buffer_p(&name2podargs.at("z"))->memory);
    for (int i = 0; i < M * N; ++i) {
      LOG_FIRST_N(INFO, 3) << "data: " << xd[i] << " + " << yd[i] << " = "
                           << zd[i];
      ASSERT_NEAR(xd[i] + yd[i], zd[i], 1e-5);
    }
  };

  // run with a arguments map passed
  instr.Run(&name2podargs, false, nullptr, false);
  // check instruction run correctly
  check_equal_by_element();

  // case 2: create cinn_pod_value_t arguments from scope;
  Scope scope;
  InstantiateScope(M, N, &scope);
  name2podargs.clear();

  for (auto& name : std::vector<std::string>({"x", "y", "z"})) {
    auto&& tensor = scope.GetTensor(name);
    name2podargs.emplace(name, tensor->buffer());
  }
  instr.Run(&name2podargs, false, nullptr, false);
  // check instruction run correctly
  check_equal_by_element();
}

#ifdef CINN_WITH_CUDNN

class TestInstruction : public Instruction {
 public:
  TestInstruction(const Target& target,
                  Scope* scope,
                  const std::vector<std::string>& in_args,
                  const std::vector<std::string>& out_args,
                  const std::string& func_name)
      : Instruction(target, scope, in_args, out_args, func_name) {}
  void SetArgs(const std::vector<int>& args) { args_ = args; }
  void SetPodArgs(const std::vector<cinn_pod_value_t>& pod_args) {
    pod_args_ = pod_args;
  }

  void RunX(std::string conv_type) {
    if (conv_type == "forward") {
      // input weight output
      runtime::cuda::cinn_call_cudnn_conv2d_forward(pod_args_.data(),
                                                    pod_args_.size(),
                                                    0,
                                                    1.0f,
                                                    0.0f,
                                                    args_[0],
                                                    args_[1],
                                                    args_[2],
                                                    args_[3],
                                                    args_[4],
                                                    args_[5],
                                                    args_[6],
                                                    args_[7],
                                                    args_[8],
                                                    args_[9],
                                                    args_[10],
                                                    args_[11],
                                                    args_[12],
                                                    args_[13],
                                                    args_[14],
                                                    args_[15],
                                                    args_[16],
                                                    args_[17],
                                                    args_[18],
                                                    stream_);
    } else if (conv_type == "backward_data") {
      // weight dy dx
      runtime::cuda::cinn_call_cudnn_conv2d_backward_data(pod_args_.data(),
                                                          pod_args_.size(),
                                                          0,
                                                          1.0f,
                                                          0.0f,
                                                          args_[0],
                                                          args_[1],
                                                          args_[2],
                                                          args_[3],
                                                          args_[4],
                                                          args_[5],
                                                          args_[6],
                                                          args_[7],
                                                          args_[8],
                                                          args_[9],
                                                          args_[10],
                                                          args_[11],
                                                          args_[12],
                                                          args_[13],
                                                          args_[14],
                                                          args_[15],
                                                          args_[16],
                                                          args_[17],
                                                          args_[18],
                                                          stream_);
    } else if (conv_type == "backward_filter") {
      // input dy dx
      runtime::cuda::cinn_call_cudnn_conv2d_backward_filter(pod_args_.data(),
                                                            pod_args_.size(),
                                                            0,
                                                            1.0f,
                                                            0.0f,
                                                            args_[0],
                                                            args_[1],
                                                            args_[2],
                                                            args_[3],
                                                            args_[4],
                                                            args_[5],
                                                            args_[6],
                                                            args_[7],
                                                            args_[8],
                                                            args_[9],
                                                            args_[10],
                                                            args_[11],
                                                            args_[12],
                                                            args_[13],
                                                            args_[14],
                                                            args_[15],
                                                            args_[16],
                                                            args_[17],
                                                            args_[18],
                                                            stream_);
    } else {
      LOG(FATAL) << "Unkown Conv Type!";
    }
    CUDA_CALL(cudaStreamSynchronize(stream_));
  }

 private:
  std::vector<int> args_;
  cudaStream_t stream_{nullptr};
  std::vector<cinn_pod_value_t> pod_args_;
};

TEST(Instruction, CONV_FORWARD) {
  int in = 32, ic = 32, ih = 128, iw = 128;
  int fn = 64, fc = 32, fh = 3, fw = 3;
  int on = 32, oc = 64, oh = 128, ow = 128;

  int ph = 1, pw = 1;
  int sh = 1, sw = 1;
  int dila_h = 1, dila_w = 1;

  int group = 1;
  std::vector<int> args = {in,
                           ic,
                           ih,
                           iw,
                           fn,
                           fc,
                           fh,
                           fw,
                           ph,
                           pw,
                           sh,
                           sw,
                           dila_h,
                           dila_w,
                           group,
                           on,
                           oc,
                           oh,
                           ow};

  // infer shape
  auto conv2d = Operator::Get("conv2d");
  auto strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  auto infer_shape_func =
      Operator::GetAttrs<InferShapeFunction>("infershape")[conv2d];

  CUDA_CALL(cudaSetDevice(0));
  auto buffer_x =
      common::BufferBuilder(Float(32), {in, ic, ih, iw}).set_random().Build();
  auto buffer_w =
      common::BufferBuilder(Float(32), {fn, fc, fh, fw}).set_random().Build();
  auto buffer_y =
      common::BufferBuilder(Float(32), {on, oc, oh, ow}).set_random().Build();

  void *dev_x = nullptr, *dev_w = nullptr, *dev_y = nullptr;
  CUDA_CALL(cudaMalloc(&dev_x, buffer_x->memory_size));
  CUDA_CALL(cudaMalloc(&dev_w, buffer_w->memory_size));
  CUDA_CALL(cudaMalloc(&dev_y, buffer_y->memory_size));

  CUDA_CALL(cudaMemcpy(
      dev_x, buffer_x->memory, buffer_x->memory_size, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(
      dev_w, buffer_w->memory, buffer_w->memory_size, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(
      dev_y, buffer_y->memory, buffer_y->memory_size, cudaMemcpyHostToDevice));

  cinn_buffer_t _x;
  cinn_buffer_t _w;
  cinn_buffer_t _y;

  _x.memory = static_cast<uint8_t*>(dev_x);
  _w.memory = static_cast<uint8_t*>(dev_w);
  _y.memory = static_cast<uint8_t*>(dev_y);

  _x.memory_size = buffer_x->memory_size;
  _w.memory_size = buffer_w->memory_size;
  _y.memory_size = buffer_y->memory_size;

  _x.type = cinn_float32_t();
  _w.type = cinn_float32_t();
  _y.type = cinn_float32_t();

  cinn_pod_value_t x(&_x);
  cinn_pod_value_t w(&_w);
  cinn_pod_value_t y(&_y);
  std::vector<cinn_pod_value_t> pod_args = {x, w, y};

  Scope scope;
  auto target = common::DefaultNVGPUTarget();
  std::vector<std::string> in_args, out_args;
  TestInstruction instr(target, &scope, in_args, out_args, "conv2d");

  instr.SetArgs(args);
  instr.SetPodArgs(pod_args);
  instr.RunX("forward");

  CUDA_CALL(cudaFree(dev_x));
  CUDA_CALL(cudaFree(dev_w));
  CUDA_CALL(cudaFree(dev_y));
}

TEST(Instruction, CONV_BACKWARD_DATA) {
  int in = 32, ic = 32, ih = 128, iw = 128;
  int fn = 64, fc = 32, fh = 3, fw = 3;
  int on = 32, oc = 64, oh = 128, ow = 128;

  int ph = 1, pw = 1;
  int sh = 1, sw = 1;
  int dila_h = 1, dila_w = 1;

  int group = 1;
  std::vector<int> args = {in,
                           ic,
                           ih,
                           iw,
                           fn,
                           fc,
                           fh,
                           fw,
                           ph,
                           pw,
                           sh,
                           sw,
                           dila_h,
                           dila_w,
                           group,
                           on,
                           oc,
                           oh,
                           ow};

  // infer shape
  auto conv2d = Operator::Get("conv2d");
  auto strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  auto infer_shape_func =
      Operator::GetAttrs<InferShapeFunction>("infershape")[conv2d];

  CUDA_CALL(cudaSetDevice(0));
  auto buffer_x =
      common::BufferBuilder(Float(32), {in, ic, ih, iw}).set_random().Build();
  auto buffer_w =
      common::BufferBuilder(Float(32), {fn, fc, fh, fw}).set_random().Build();
  auto buffer_y =
      common::BufferBuilder(Float(32), {on, oc, oh, ow}).set_random().Build();

  void *dev_x = nullptr, *dev_w = nullptr, *dev_y = nullptr;
  CUDA_CALL(cudaMalloc(&dev_x, buffer_x->memory_size));
  CUDA_CALL(cudaMalloc(&dev_w, buffer_w->memory_size));
  CUDA_CALL(cudaMalloc(&dev_y, buffer_y->memory_size));

  CUDA_CALL(cudaMemcpy(
      dev_x, buffer_x->memory, buffer_x->memory_size, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(
      dev_w, buffer_w->memory, buffer_w->memory_size, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(
      dev_y, buffer_y->memory, buffer_y->memory_size, cudaMemcpyHostToDevice));

  cinn_buffer_t _x;
  cinn_buffer_t _w;
  cinn_buffer_t _y;

  _x.memory = static_cast<uint8_t*>(dev_x);
  _w.memory = static_cast<uint8_t*>(dev_w);
  _y.memory = static_cast<uint8_t*>(dev_y);

  _x.type = cinn_float32_t();
  _w.type = cinn_float32_t();
  _y.type = cinn_float32_t();

  _x.memory_size = buffer_x->memory_size;
  _w.memory_size = buffer_w->memory_size;
  _y.memory_size = buffer_y->memory_size;

  cinn_pod_value_t x(&_x);
  cinn_pod_value_t w(&_w);
  cinn_pod_value_t y(&_y);
  // weight dy dx
  std::vector<cinn_pod_value_t> pod_args = {w, y, x};

  Scope scope;
  auto target = common::DefaultNVGPUTarget();
  std::vector<std::string> in_args, out_args;
  TestInstruction instr(target, &scope, in_args, out_args, "conv2d");

  instr.SetArgs(args);
  instr.SetPodArgs(pod_args);
  instr.RunX("backward_data");

  CUDA_CALL(cudaFree(dev_x));
  CUDA_CALL(cudaFree(dev_w));
  CUDA_CALL(cudaFree(dev_y));
}

TEST(Instruction, CONV_BACKWARD_FILTER) {
  int in = 32, ic = 32, ih = 128, iw = 128;
  int fn = 64, fc = 32, fh = 3, fw = 3;
  int on = 32, oc = 64, oh = 128, ow = 128;

  int ph = 1, pw = 1;
  int sh = 1, sw = 1;
  int dila_h = 1, dila_w = 1;

  int group = 1;
  std::vector<int> args = {in,
                           ic,
                           ih,
                           iw,
                           fn,
                           fc,
                           fh,
                           fw,
                           ph,
                           pw,
                           sh,
                           sw,
                           dila_h,
                           dila_w,
                           group,
                           on,
                           oc,
                           oh,
                           ow};

  // infer shape
  auto conv2d = Operator::Get("conv2d");
  auto strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  auto infer_shape_func =
      Operator::GetAttrs<InferShapeFunction>("infershape")[conv2d];

  absl::flat_hash_map<std::string, AttrType> attrs_map;
  attrs_map["padding"] = std::vector<int>({ph, pw});
  attrs_map["stride"] = std::vector<int>({sh, sw});
  attrs_map["dilation"] = std::vector<int>({dila_h, dila_w});
  attrs_map["data_format"] = std::string("NCHW");
  attrs_map["conv_type"] = std::string("backward_filter");
  attrs_map["output_shape"] = std::vector<int>({fn, fc, fh, fw});

  auto infer_shape =
      infer_shape_func({{in, ic, ih, iw}, {on, oc, oh, ow}}, attrs_map);
  ASSERT_EQ(infer_shape[0][0], fn);
  ASSERT_EQ(infer_shape[0][1], fc);
  ASSERT_EQ(infer_shape[0][2], fh);
  ASSERT_EQ(infer_shape[0][3], fw);

  CUDA_CALL(cudaSetDevice(0));
  auto buffer_x =
      common::BufferBuilder(Float(32), {in, ic, ih, iw}).set_random().Build();
  auto buffer_w =
      common::BufferBuilder(Float(32), {fn, fc, fh, fw}).set_random().Build();
  auto buffer_y =
      common::BufferBuilder(Float(32), {on, oc, oh, ow}).set_random().Build();

  void *dev_x = nullptr, *dev_w = nullptr, *dev_y = nullptr;
  CUDA_CALL(cudaMalloc(&dev_x, buffer_x->memory_size));
  CUDA_CALL(cudaMalloc(&dev_w, buffer_w->memory_size));
  CUDA_CALL(cudaMalloc(&dev_y, buffer_y->memory_size));

  CUDA_CALL(cudaMemcpy(
      dev_x, buffer_x->memory, buffer_x->memory_size, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(
      dev_w, buffer_w->memory, buffer_w->memory_size, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(
      dev_y, buffer_y->memory, buffer_y->memory_size, cudaMemcpyHostToDevice));

  cinn_buffer_t _x;
  cinn_buffer_t _w;
  cinn_buffer_t _y;

  _x.type = cinn_float32_t();
  _w.type = cinn_float32_t();
  _y.type = cinn_float32_t();

  _x.memory = static_cast<uint8_t*>(dev_x);
  _w.memory = static_cast<uint8_t*>(dev_w);
  _y.memory = static_cast<uint8_t*>(dev_y);

  _x.memory_size = buffer_x->memory_size;
  _w.memory_size = buffer_w->memory_size;
  _y.memory_size = buffer_y->memory_size;

  cinn_pod_value_t x(&_x);
  cinn_pod_value_t w(&_w);
  cinn_pod_value_t y(&_y);
  // input dy dw
  std::vector<cinn_pod_value_t> pod_args = {x, y, w};

  Scope scope;
  auto target = common::DefaultNVGPUTarget();
  std::vector<std::string> in_args, out_args;
  TestInstruction instr(target, &scope, in_args, out_args, "conv2d");

  instr.SetArgs(args);
  instr.SetPodArgs(pod_args);
  instr.RunX("backward_filter");

  CUDA_CALL(cudaFree(dev_x));
  CUDA_CALL(cudaFree(dev_w));
  CUDA_CALL(cudaFree(dev_y));
}

#endif
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
