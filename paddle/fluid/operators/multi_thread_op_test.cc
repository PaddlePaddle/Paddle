// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <condition_variable>  // NOLINT
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <vector>
#include "gtest/gtest.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/type_defs.h"

struct IInput {
  virtual ~IInput() {}
  virtual void Initialize(const std::string& name,
                          paddle::framework::Scope* scope) = 0;
  virtual void Share(const paddle::framework::Variable& in,
                     paddle::framework::Variable* out) = 0;
};

struct IOutput {
  virtual ~IOutput() {}

  virtual std::string CreateVar(const std::string& name,
                                paddle::framework::Scope* scope) = 0;
};

struct OpTest {
  std::unordered_map<std::string, std::vector<std::unique_ptr<IInput>>> params;
  std::unordered_map<std::string, std::vector<std::unique_ptr<IInput>>> inputs;

  std::string op_type;
  paddle::framework::AttributeMap attrs;
  std::unordered_map<std::string, std::vector<std::unique_ptr<IOutput>>>
      outputs;
};

struct OpTestBuilder {
  OpTest op_test;

  OpTestBuilder& Op(const std::string& type) {
    op_test.op_type = type;
    return *this;
  }

  OpTestBuilder& SetAttr(const std::string& name,
                         const paddle::framework::Attribute& attr) {
    op_test.attrs[name] = attr;
    return *this;
  }

  OpTestBuilder& AddInput(const std::string& name,
                          std::unique_ptr<IInput>&& in) {
    op_test.inputs[name].emplace_back(std::move(in));
    return *this;
  }

  template <typename T, typename... ARGS>
  OpTestBuilder& AddInput(const std::string& name, ARGS... args) {
    return AddInput(name, std::unique_ptr<IInput>(new T(args...)));
  }

  OpTestBuilder& AddOutput(const std::string& name,
                           std::unique_ptr<IOutput>&& out) {
    op_test.outputs[name].emplace_back(std::move(out));
    return *this;
  }

  template <typename T, typename... ARGS>
  OpTestBuilder& AddOutput(const std::string& name, ARGS... args) {
    return AddOutput(name, std::unique_ptr<IOutput>(new T(args...)));
  }

  OpTestBuilder& AddParam(const std::string& name,
                          std::unique_ptr<IInput>&& p) {
    op_test.params[name].emplace_back(std::move(p));
    return *this;
  }

  template <typename T, typename... ARGS>
  OpTestBuilder& AddParam(const std::string& name, ARGS... args) {
    return AddParam(name, std::unique_ptr<IInput>(new T(args...)));
  }
};

struct LoDTensorOutput : public IOutput {
  std::string CreateVar(const std::string& name,
                        paddle::framework::Scope* scope) override {
    scope->Var(name)->GetMutable<paddle::framework::LoDTensor>();
    return name;
  }
};

struct InplaceOutput : public IOutput {
  std::string prefix_;
  explicit InplaceOutput(const std::string& prefix) : prefix_(prefix) {}

  std::string CreateVar(const std::string& name,
                        paddle::framework::Scope* scope) override {
    return prefix_ + "_0";
  }
};

struct LoDTensorInput : public IInput {
  void Share(const paddle::framework::Variable& in,
             paddle::framework::Variable* out) override {
    auto& out_t = *out->GetMutable<paddle::framework::LoDTensor>();
    auto& in_t = in.Get<paddle::framework::LoDTensor>();
    out_t.ShareDataWith(in_t);
    out_t.set_lod(in_t.lod());
  }
};

USE_OP(uniform_random);

template <typename T>
struct UniformLoDTensorInput : public LoDTensorInput {
  float min_;
  float max_;
  std::vector<int> shape_;
  explicit UniformLoDTensorInput(std::vector<int> shape, float min = -1.0,
                                 float max = 1.0)
      : min_(min), max_(max), shape_(shape) {}
  void Initialize(const std::string& name,
                  paddle::framework::Scope* scope) override {
    scope->Var(name)->GetMutable<paddle::framework::LoDTensor>();
    paddle::framework::OpRegistry::CreateOp(
        "uniform_random", {}, {{"Out", {name}}},
        {{"min", min_},
         {"max", max_},
         {"shape", shape_},
         {"dtype", paddle::framework::ToDataType(typeid(T))}})
        ->Run(*scope, paddle::platform::CPUPlace());
  }
};

struct ExecutionFlag {
  std::mutex mtx_;
  std::condition_variable cv_;
  bool start_{false};

  void Wait() {
    std::unique_lock<std::mutex> lock(mtx_);
    cv_.wait(lock, [this] { return start_; });
  }

  void Post() {
    {
      std::lock_guard<std::mutex> guard(mtx_);
      start_ = true;
    }
    cv_.notify_all();
  }
};

inline static void TestMain(OpTest&& info, size_t max_num_thread, size_t iter,
                            size_t step = 1) {
  paddle::framework::Scope param_scope;

  std::unique_ptr<paddle::framework::Scope[]> working_scopes(
      new paddle::framework::Scope[max_num_thread]);

  std::map<std::string, std::vector<std::string>> inputs;

  for (auto& param : info.params) {
    std::string pname = param.first;
    for (size_t i = 0; i < param.second.size(); ++i) {
      std::unique_ptr<IInput>& in = param.second[i];
      auto vname = paddle::string::Sprintf("%s_%d", pname, i);
      inputs[pname].emplace_back(vname);
      in->Initialize(vname, &param_scope);
      for (size_t j = 0; j < max_num_thread; j += step) {
        in->Share(*param_scope.Var(vname), working_scopes[j].Var(vname));
      }
    }
  }

  for (auto& input : info.inputs) {
    std::string iname = input.first;
    for (size_t i = 0; i < input.second.size(); ++i) {
      auto vname = paddle::string::Sprintf("%s_%d", iname, i);
      inputs[iname].emplace_back(vname);
      std::unique_ptr<IInput>& in = input.second[i];
      for (size_t j = 0; j < max_num_thread; j += step) {
        in->Initialize(vname, &working_scopes[j]);
      }
    }
  }

  std::map<std::string, std::vector<std::string>> outputs;
  for (auto& output : info.outputs) {
    std::string oname = output.first;
    for (size_t i = 0; i < output.second.size(); ++i) {
      auto vname = paddle::string::Sprintf("%s_%d", oname, i);
      std::unique_ptr<IOutput>& out = output.second[i];
      for (size_t j = 0; j < max_num_thread; j += step) {
        vname = out->CreateVar(vname, &working_scopes[j]);
        if (j == 0) {
          outputs[oname].emplace_back(vname);
        }
      }
    }
  }

  std::unique_ptr<std::chrono::nanoseconds[]> times(
      new std::chrono::nanoseconds[max_num_thread]);

  auto thread_main = [&](size_t i, size_t iter, ExecutionFlag& flag) {
    auto& scope = working_scopes[i];
    auto op = paddle::framework::OpRegistry::CreateOp(info.op_type, inputs,
                                                      outputs, info.attrs);
    flag.Wait();
    paddle::platform::CPUPlace cpu;
    for (size_t j = 0; j < iter; ++j) {
      op->Run(scope, cpu);
    }
  };

  for (size_t i = 0; i < max_num_thread; i += step) {
    std::vector<std::thread> threads;
    ExecutionFlag flag;
    for (size_t j = 0; j < i + 1; ++j) {
      threads.emplace_back(
          [j, &flag, &thread_main, &iter] { thread_main(j, iter, flag); });
    }
    flag.Post();
    auto beg = std::chrono::high_resolution_clock::now();
    for (auto& th : threads) {
      th.join();
    }
    auto end = std::chrono::high_resolution_clock::now();
    times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg);
  }

  std::cerr << info.op_type << " speed up ratio:\n";
  for (size_t i = 0; i < max_num_thread; i += step) {
    std::cerr << "\t" << i + 1 << "\t";
    if (i == 0) {
      std::cerr << 1.0;
    } else {
      std::cerr << (i + 1.0d) * times[0].count() / times[i].count();
    }
    std::cerr << std::endl;
  }
}

USE_OP(mul);
TEST(OpMultiThread, mul) {
  OpTestBuilder builder;
  builder.Op("mul")
      .AddInput<UniformLoDTensorInput<float>>("X", std::vector<int>{64, 2500})
      .AddParam<UniformLoDTensorInput<float>>("Y", std::vector<int>{2500, 2})
      .AddOutput<LoDTensorOutput>("Out");
  TestMain(std::move(builder.op_test), 6, 1 << 12);
}

USE_OP(batch_norm);

template <int ChannelSize, bool is_test>
static inline void TestBatchNorm() {
  OpTestBuilder builder;
  std::cerr << "batch_norm: ChannelSize " << ChannelSize << ", is_test"
            << is_test << std::endl;
  builder.Op("batch_norm")
      .AddInput<UniformLoDTensorInput<float>>(
          "X", std::vector<int>{1, ChannelSize, 224, 224})
      .AddInput<UniformLoDTensorInput<float>>("Scale",
                                              std::vector<int>{ChannelSize})
      .AddInput<UniformLoDTensorInput<float>>("Bias",
                                              std::vector<int>{ChannelSize})
      .AddInput<UniformLoDTensorInput<float>>(
          "Mean", std::vector<int>{1, ChannelSize, 224, 224}, -1.0f, -0.1f)
      .AddInput<UniformLoDTensorInput<float>>(
          "Variance", std::vector<int>{1, ChannelSize, 224, 224})
      .AddOutput<LoDTensorOutput>("Y")
      .AddOutput<InplaceOutput>("MeanOut", "Mean")
      .AddOutput<InplaceOutput>("VarianceOut", "Variance")
      .AddOutput<LoDTensorOutput>("SavedMean")
      .AddOutput<LoDTensorOutput>("SavedVariance")
      .SetAttr("is_test", is_test);

  TestMain(std::move(builder.op_test), 6, 1 << 12);
}

TEST(OpMultiThread, batch_norm_channel_3) { TestBatchNorm<3, true>(); }
TEST(OpMultiThread, batch_norm_channel_10) { TestBatchNorm<10, true>(); }
TEST(OpMultiThread, batch_norm_channel_20) { TestBatchNorm<20, true>(); }
TEST(OpMultiThread, batch_norm_channel_30) { TestBatchNorm<30, true>(); }
