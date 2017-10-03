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

#include <gtest/gtest.h>
#include "paddle/framework/grad_op_builder.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/operator.h"

USE_OP(elementwise_add);

using namespace paddle::platform;
using namespace paddle::framework;

TEST(Executor, Init) {
  ProgramDesc pdesc;

  auto root_block = pdesc.add_blocks();
  root_block->set_idx(0);
  root_block->set_parent_idx(-1);

  auto a = root_block->add_vars();
  a->set_name("a");
  auto a_lt = a->mutable_lod_tensor();
  a_lt->set_data_type(paddle::framework::DataType::FP32);
  a_lt->add_dims(640);
  a_lt->add_dims(640);

  auto b = root_block->add_vars();
  b->set_name("b");
  auto b_lt = b->mutable_lod_tensor();
  b_lt->set_data_type(paddle::framework::DataType::FP32);
  b_lt->add_dims(640);
  b_lt->add_dims(640);

  auto c = root_block->add_vars();
  c->set_name("c");
  auto c_lt = c->mutable_lod_tensor();
  c_lt->set_data_type(paddle::framework::DataType::FP32);
  c_lt->add_dims(640);
  c_lt->add_dims(640);

  auto op1 = root_block->add_ops();
  op1->set_type("elementwise_add");
  auto X = op1->add_inputs();
  X->set_parameter("X");
  X->add_arguments("a");
  auto Y = op1->add_inputs();
  Y->set_parameter("Y");
  Y->add_arguments("b");

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
