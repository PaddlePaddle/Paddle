/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/framework/executor.h"
#include <vector>
#include "gtest/gtest.h"
#include "paddle/framework/attribute.h"
#include "paddle/framework/backward.h"
#include "paddle/framework/block_desc.h"
#include "paddle/framework/grad_op_builder.h"
#include "paddle/framework/op_desc.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/operator.h"

USE_OP(elementwise_add);
USE_OP(gaussian_random);
USE_OP(feed);
USE_OP(fetch);
USE_OP(mul);
USE_OP(squared_l2_distance);

using std::string;
using namespace paddle::platform;
using namespace paddle::framework;

typedef paddle::framework::BlockDesc proto_block;
typedef paddle::framework::OpDesc proto_op;

struct SetAttrDescVisitor : public boost::static_visitor<void> {
  explicit SetAttrDescVisitor(OpDesc::Attr* attr) : attr_(attr) {}
  mutable OpDesc::Attr* attr_;
  void operator()(int v) const { attr_->set_i(v); }
  void operator()(float v) const { attr_->set_f(v); }
  void operator()(const std::string& v) const { attr_->set_s(v); }
  void operator()(bool b) const { attr_->set_b(b); }

  void operator()(const std::vector<int>& v) const {
    VectorToRepeated(v, attr_->mutable_ints());
  }
  void operator()(const std::vector<float>& v) const {
    VectorToRepeated(v, attr_->mutable_floats());
  }
  void operator()(const std::vector<std::string>& v) const {
    VectorToRepeated(v, attr_->mutable_strings());
  }
  void operator()(const std::vector<bool>& v) const {
    VectorToRepeated(v, attr_->mutable_bools());
  }
  void operator()(BlockDesc* desc) const { attr_->set_block_idx(desc->idx()); }
  void operator()(boost::blank) const { PADDLE_THROW("Unexpected branch"); }
};

void AddOp(const std::string& type, const VariableNameMap& inputs,
           const VariableNameMap& outputs, AttributeMap attrs,
           proto_block* block) {
  // insert output
  for (auto kv : outputs) {
    for (auto v : kv.second) {
      auto var = block->add_vars();
      var->set_name(v);
      auto var_lt = var->mutable_lod_tensor();
      var_lt->set_data_type(paddle::framework::DataType::FP32);
    }
  }

  // insert op
  auto op = block->add_ops();
  op->set_type(type);
  for (auto kv : inputs) {
    auto X = op->add_inputs();
    X->set_parameter(kv.first);
    for (auto argu : kv.second) {
      X->add_arguments(argu);
    }
  }
  for (auto kv : outputs) {
    auto X = op->add_outputs();
    X->set_parameter(kv.first);
    for (auto argu : kv.second) {
      X->add_arguments(argu);
    }
  }
  for (auto& attr : attrs) {
    auto* attr_desc = op->add_attrs();
    attr_desc->set_name(attr.first);
    attr_desc->set_type(
        static_cast<paddle::framework::AttrType>(attr.second.which() - 1));
    SetAttrDescVisitor visitor(attr_desc);
    boost::apply_visitor(visitor, attr.second);
  }
}

std::once_flag set_variable_flag;

// Tensors in feed value variable will only be in CPUPlace
// So we can  memcpy the data from vector<T> to feed_value
template <typename T>
void set_feed_variable(const std::vector<std::vector<T>>& inputs) {
  typedef std::vector<paddle::framework::Tensor> FeedInputs;
  Variable* g_feed_value = GetGlobalScope()->FindVar("feed_value");
  FeedInputs& feed_inputs = *(g_feed_value->GetMutable<FeedInputs>());
  auto size = inputs.size();
  feed_inputs.resize(size);
  for (size_t i = 0; i < size; i++) {
    T* dst = feed_inputs[i].mutable_data<T>(
        make_ddim({static_cast<int64_t>(inputs[i].size())}), CPUPlace());
    memcpy(dst, inputs[i].data(), inputs[i].size() * sizeof(T));
  }
}

// Tensors in fetch value variable will only be in CPUPlace
// So we can memcpy the data from fetch_value to vector<T>
template <typename T>
std::vector<std::vector<T>> get_fetch_variable() {
  typedef std::vector<paddle::framework::Tensor> FetchOutputs;
  Variable* g_fetch_value = GetGlobalScope()->FindVar("fetch_value");
  FetchOutputs& fetch_outputs = *(g_fetch_value->GetMutable<FetchOutputs>());

  auto size = fetch_outputs.size();
  std::vector<std::vector<T>> result;
  result.reserve(size);
  for (size_t i = 0; i < size; i++) {
    std::vector<T> tmp;
    tmp.resize(fetch_outputs[i].numel());
    memcpy(tmp.data(), fetch_outputs[i].data<T>(),
           fetch_outputs[i].numel() * sizeof(T));
    result.push_back(tmp);
  }

  return result;
}

class ExecutorTesterRandom : public ::testing::Test {
 public:
  virtual void SetUp() override {
    int input_dim = 5, batch_size = 2, embed_dim = 5;

    // init pdesc
    auto init_root_block = init_pdesc_.add_blocks();
    init_root_block->set_idx(0);
    init_root_block->set_parent_idx(-1);
    AddOp("gaussian_random", {}, {{"Out", {"w1"}}},
          {{"dims", std::vector<int>{input_dim, embed_dim}}}, init_root_block);
    AddOp("gaussian_random", {}, {{"Out", {"w2"}}},
          {{"dims", std::vector<int>{embed_dim, input_dim}}}, init_root_block);
    AddOp("fetch", {{"Input", {"w1"}}}, {},
          {{"dims", std::vector<int>{input_dim, embed_dim}}, {"col", 0}},
          init_root_block);
    AddOp("fetch", {{"Input", {"w2"}}}, {},
          {{"dims", std::vector<int>{embed_dim, input_dim}}, {"col", 1}},
          init_root_block);

    // run pdesc
    auto root_block = pdesc_.add_blocks();
    root_block->set_idx(0);
    root_block->set_parent_idx(-1);

    AddOp("gaussian_random", {}, {{"Out", {"a"}}},
          {{"dims", std::vector<int>{batch_size, input_dim}}}, root_block);
    AddOp("mul", {{"X", {"a"}}, {"Y", {"w1"}}}, {{"Out", {"b"}}}, {},
          root_block);
    AddOp("mul", {{"X", {"b"}}, {"Y", {"w2"}}}, {{"Out", {"a_out"}}}, {},
          root_block);
    AddOp("squared_l2_distance", {{"X", {"a"}}, {"Y", {"a_out"}}},
          {{"Out", {"l2_distance"}}, {"sub_result", {"l2_distance_sub"}}}, {},
          root_block);

    AppendBackward(pdesc_, {});
    // AddOp("fetch", {{"Input", {"sub_result"}}}, {},
    //       {{"dims", std::vector<int>{input_dim, batch_size}}, {"col", 0}},
    //       root_block);
    AddOp("fetch", {{"Input", {"l2_distance"}}}, {},
          {{"dims", std::vector<int>{batch_size}}, {"col", 1}}, root_block);
  }

 protected:
  ProgramDesc pdesc_;
  ProgramDesc init_pdesc_;
};

class ExecutorTesterFeedAndFetch : public ::testing::Test {
 public:
  virtual void SetUp() override {
    auto root_block = pdesc_.add_blocks();
    root_block->set_idx(0);
    root_block->set_parent_idx(-1);

    std::vector<int> dim{6};

    AddOp("feed", {}, {{"Out", {"a"}}}, {{"dims", dim}, {"col", 0}},
          root_block);
    AddOp("feed", {}, {{"Out", {"b"}}}, {{"dims", dim}, {"col", 1}},
          root_block);
    AddOp("fetch", {{"Input", {"a"}}}, {}, {{"dims", dim}, {"col", 0}},
          root_block);
    AddOp("fetch", {{"Input", {"b"}}}, {}, {{"dims", dim}, {"col", 1}},
          root_block);

    std::vector<float> vec1 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::vector<float> vec2 = {4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    inputs_.push_back(vec1);
    inputs_.push_back(vec2);
  }

 protected:
  ProgramDesc pdesc_;
  std::vector<std::vector<float>> inputs_;
};

#ifndef PADDLE_WITH_CUDA
TEST_F(ExecutorTesterRandom, CPU) {
  std::vector<Place> places;
  CPUPlace cpu_place;
  places.push_back(cpu_place);

  // We have a global Scope and BuddyAllocator, and we must ensure
  // global BuddyAllocator is initialized before global Scope. Thus,
  // global Scope will deconstruct before BuddyAllocator. Otherwise,
  // "pointer being freed was not allocated" error will appear.
  paddle::memory::Used(cpu_place);

  Executor* executor = new Executor(places);
  executor->Run(init_pdesc_, GetGlobalScope());
  executor->Run(pdesc_, GetGlobalScope());
  std::vector<std::vector<float>> result = get_fetch_variable<float>();

  for (auto& vec : result) {
    for (auto& num : vec) {
      std::cout << num << " ";
    }
    std::cout << std::endl;
  }
  delete executor;
}

TEST_F(ExecutorTesterFeedAndFetch, CPU) {
  std::vector<Place> places;
  CPUPlace cpu_place;
  places.push_back(cpu_place);

  // We have a global Scope and BuddyAllocator, and we must ensure
  // global BuddyAllocator is initialized before global Scope. Thus,
  // global Scope will deconstruct before BuddyAllocator. Otherwise,
  // "pointer being freed was not allocated" error will appear.
  paddle::memory::Used(cpu_place);

  Executor* executor = new Executor(places);

  // 3 mini-batch
  for (int i = 0; i < 3; i++) {
    set_feed_variable<float>(inputs_);
    executor->Run(pdesc_, GetGlobalScope());
    std::vector<std::vector<float>> result = get_fetch_variable<float>();
    PADDLE_ENFORCE_EQ(result.size(), inputs_.size());
    for (size_t i = 0; i < result.size(); ++i) {
      PADDLE_ENFORCE_EQ(result[i].size(), inputs_[i].size());
      for (size_t j = 0; j < result[i].size(); ++j) {
        PADDLE_ENFORCE_EQ(result[i][j], inputs_[i][j]);
      }
    }
  }

  delete executor;
}
#else
TEST_F(ExecutorTesterRandom, GPU) {
  std::vector<Place> places;
  GPUPlace gpu_place(0);
  places.push_back(gpu_place);

  // We have a global Scope and BuddyAllocator, and we must ensure
  // global BuddyAllocator is initialized before global Scope. Thus,
  // global Scope will deconstruct before BuddyAllocator. Otherwise,
  // "pointer being freed was not allocated" error will appear.
  // If paddle is compiled with GPU, both CPU and GPU BuddyAllocator
  // need to be used at first.
  paddle::memory::Used(CPUPlace());
  paddle::memory::Used(gpu_place);

  Executor* executor = new Executor(places);

  LOG(INFO) << "Run Init";
  executor->Run(init_pdesc_, GetGlobalScope());
  LOG(INFO) << "Run";
  executor->Run(pdesc_, GetGlobalScope());
  std::vector<std::vector<float>> result = get_fetch_variable<float>();

  for (auto& vec : result) {
    for (auto& num : vec) {
      std::cout << num << " ";
    }
    std::cout << std::endl;
  }
  delete executor;
}

TEST_F(ExecutorTesterFeedAndFetch, GPU) {
  std::vector<Place> places;
  GPUPlace gpu_place(0);
  places.push_back(gpu_place);
  // We have a global Scope and BuddyAllocator, and we must ensure
  // global BuddyAllocator is initialized before global Scope. Thus,
  // global Scope will deconstruct before BuddyAllocator. Otherwise,
  // "pointer being freed was not allocated" error will appear.
  // If paddle is compiled with GPU, both CPU and GPU BuddyAllocator
  // need to be used at first.
  paddle::memory::Used(CPUPlace());
  paddle::memory::Used(gpu_place);

  Executor* executor = new Executor(places);

  // 3 mini-batch
  for (int i = 0; i < 3; i++) {
    set_feed_variable<float>(inputs_);
    executor->Run(pdesc_, GetGlobalScope());
    std::vector<std::vector<float>> result = get_fetch_variable<float>();
    PADDLE_ENFORCE_EQ(result.size(), inputs_.size());
    for (size_t i = 0; i < result.size(); ++i) {
      PADDLE_ENFORCE_EQ(result[i].size(), inputs_[i].size());
      for (size_t j = 0; j < result[i].size(); ++j) {
        PADDLE_ENFORCE_EQ(result[i][j], inputs_[i][j]);
      }
    }
  }
  delete executor;
}
#endif
