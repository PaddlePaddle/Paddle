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
#include <memory>  // for unique_ptr
#include <mutex>   // for call_once
#include <vector>
#include "gtest/gtest.h"
#include "paddle/framework/attribute.h"
#include "paddle/framework/grad_op_builder.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/operator.h"

USE_OP(elementwise_add);
USE_OP(gaussian_random);
USE_OP(feed);

using std::string;
using namespace paddle::platform;
using namespace paddle::framework;

typedef paddle::framework::BlockDesc proto_block;
typedef paddle::framework::OpDesc proto_op;

void add_gaussian_random_op(string var_name, proto_block* block) {
  std::vector<int> dim{2, 3};

  // insert variable
  auto a = block->add_vars();
  a->set_name(var_name);
  auto a_lt = a->mutable_lod_tensor();
  a_lt->set_data_type(paddle::framework::DataType::FP32);
  for (int i : dim) {
    a_lt->add_dims(i);
  }

  // insert operation
  auto op = block->add_ops();
  op->set_type("gaussian_random");
  auto dims = op->add_attrs();
  dims->set_name("dims");
  dims->set_type(paddle::framework::AttrType::INTS);
  for (int i : dim) {
    dims->add_ints(i);
  }
  auto Out = op->add_outputs();
  Out->set_parameter("Out");
  Out->add_arguments(var_name);
}

void add_feed_op(string var_name, int index, proto_block* block) {
  std::vector<int> dim{3};

  // insert variable
  auto a = block->add_vars();
  a->set_name(var_name);
  auto a_lt = a->mutable_lod_tensor();
  a_lt->set_data_type(paddle::framework::DataType::FP32);
  for (int i : dim) {
    a_lt->add_dims(i);
  }

  // insert operation
  auto op = block->add_ops();
  op->set_type("feed");

  // set dims attr
  auto dims = op->add_attrs();
  dims->set_name("dims");
  dims->set_type(paddle::framework::AttrType::INTS);
  for (int i : dim) {
    dims->add_ints(i);
  }

  // set col attr
  auto col = op->add_attrs();
  col->set_name("col");
  col->set_type(paddle::framework::AttrType::INT);
  col->set_i(index);

  auto Out = op->add_outputs();
  Out->set_parameter("Out");
  Out->add_arguments(var_name);
}

std::once_flag set_variable_flag;

template <typename T>
void set_feed_variable(const std::vector<std::vector<T>>& inputs) {
  typedef std::vector<paddle::framework::Tensor> FeedInputs;
  Variable* g_feed_value = GetScope()->FindVar("feed_value");
  FeedInputs& feed_inputs = *(g_feed_value->GetMutable<FeedInputs>());
  auto size = inputs.size();

  std::call_once(set_variable_flag, [&]() {
    feed_inputs.reserve(size);
    for (size_t i = 0; i < size; i++) {
      paddle::framework::Tensor tmp;
      tmp.mutable_data<T>(make_ddim({static_cast<int64_t>(inputs[i].size())}),
                          CPUPlace());
      feed_inputs.push_back(tmp);
    }
  });

  for (size_t i = 0; i < size; i++) {
    memcpy(feed_inputs[i].data<T>(), inputs[i].data(),
           inputs[i].size() * sizeof(T));
  }
}

class ExecutorTesterRandom : public ::testing::Test {
 public:
  virtual void SetUp() override {
    auto root_block = pdesc_.add_blocks();
    root_block->set_idx(0);
    root_block->set_parent_idx(-1);

    add_gaussian_random_op("a", root_block);
    add_gaussian_random_op("b", root_block);

    auto c = root_block->add_vars();
    c->set_name("c");
    auto c_lt = c->mutable_lod_tensor();
    c_lt->set_data_type(paddle::framework::DataType::FP32);

    auto op = root_block->add_ops();
    op->set_type("elementwise_add");
    auto X = op->add_inputs();
    X->set_parameter("X");
    X->add_arguments("a");
    auto Y = op->add_inputs();
    Y->set_parameter("Y");
    Y->add_arguments("b");
    auto Out = op->add_outputs();
    Out->set_parameter("Out");
    Out->add_arguments("c");

    scope_ = GetScope();
  }

 protected:
  ProgramDesc pdesc_;
  Scope* scope_;
};

class ExecutorTesterFeed : public ::testing::Test {
 public:
  virtual void SetUp() override {
    auto root_block = pdesc_.add_blocks();
    root_block->set_idx(0);
    root_block->set_parent_idx(-1);

    add_feed_op("a", 0, root_block);
    add_feed_op("b", 1, root_block);

    auto c = root_block->add_vars();
    c->set_name("c");
    auto c_lt = c->mutable_lod_tensor();
    c_lt->set_data_type(paddle::framework::DataType::FP32);

    auto op = root_block->add_ops();
    op->set_type("elementwise_add");
    auto X = op->add_inputs();
    X->set_parameter("X");
    X->add_arguments("a");
    auto Y = op->add_inputs();
    Y->set_parameter("Y");
    Y->add_arguments("b");
    auto Out = op->add_outputs();
    Out->set_parameter("Out");
    Out->add_arguments("c");

    std::vector<float> vec1 = {1.0, 2.0, 3.0};
    std::vector<float> vec2 = {4.0, 5.0, 6.0};
    inputs_.push_back(vec1);
    inputs_.push_back(vec2);
  }

 protected:
  ProgramDesc pdesc_;
  std::vector<std::vector<float>> inputs_;
};

TEST_F(ExecutorTesterRandom, CPU) {
  std::vector<Place> places;
  CPUPlace cpu_place1, cpu_place2;
  places.push_back(cpu_place1);
  places.push_back(cpu_place2);

  Executor* executor = new Executor(places);
  executor->Run(pdesc_, scope_);
  delete executor;
}

TEST_F(ExecutorTesterFeed, CPU) {
  std::vector<Place> places;
  CPUPlace cpu_place;
  places.push_back(cpu_place);

  Executor* executor = new Executor(places);

  // 3 mini-batch
  for (int i = 0; i < 3; i++) {
    // need to set feed variable before Executor::Run
    set_feed_variable<float>(inputs_);
    executor->Run(pdesc_, GetScope());
  }

  delete executor;
}

#ifdef PADDLE_WITH_GPU
TEST_F(ExecutorTesterRandom, GPU) {
  std::vector<Place> places;
  GPUPlace gpu_place(0);
  places.push_back(gpu_place);

  Executor* executor = new Executor(places);
  executor->Run(pdesc_, scope_);
  delete executor;
}

TEST_F(ExecutorTesterFeed, GPU) {
  std::vector<Place> places;
  GPUPlace gpu_place(0);
  places.push_back(gpu_place);

  Executor* executor = new Executor(places);

  // need to set feed variable before Executor::Run
  set_feed_variable<float>(inputs_);
  executor->Run(pdesc_, scope_);

  delete executor;
}
#endif
