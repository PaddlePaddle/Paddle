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
#include "gtest/gtest.h"
#include "paddle/framework/attribute.h"

#include "paddle/framework/grad_op_builder.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/operator.h"

#include <vector>

USE_OP(elementwise_add);
USE_OP(gaussian_random);

using namespace paddle::platform;
using namespace paddle::framework;

typedef paddle::framework::BlockDesc proto_block;
typedef paddle::framework::OpDesc proto_op;

using std::string;

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

TEST(Executor, Init) {
  ProgramDesc pdesc;

  auto root_block = pdesc.add_blocks();
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

  CPUPlace cpu_place1, cpu_place2;
  std::vector<Place> places;
  places.push_back(cpu_place1);
  places.push_back(cpu_place2);

  Executor* executor = new Executor(places);
  Scope s;
  std::vector<Tensor>* outputs{nullptr};
  executor->Run(pdesc, &s, outputs);

  delete executor;
}
