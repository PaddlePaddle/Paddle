/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <iostream>
#include <sstream>
#include "glog/logging.h"
#include "gtest/gtest.h"

#include "paddle/fluid/distributed/auto_parallel/dist_attr.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/phi/core/distributed/auto_parallel/proto_helper.h"

namespace phi {
namespace distributed {
namespace auto_parallel {

using paddle::framework::ProgramDesc;
using paddle::framework::VarDesc;

using paddle::distributed::auto_parallel::get_tensor_shape;
using paddle::distributed::auto_parallel::OperatorDistAttr;

TEST(DistAttr, ctor) {
  ProgramDesc program;
  auto* global_block = program.MutableBlock(0);
  auto* x = global_block->Var("X");
  x->SetType(paddle::framework::proto::VarType::DENSE_TENSOR);
  x->SetLoDLevel(0);
  x->SetDataType(paddle::framework::proto::VarType::FP32);
  x->SetShape({1000, 784});

  auto* y = global_block->Var("Y");
  y->SetType(paddle::framework::proto::VarType::DENSE_TENSOR);
  y->SetLoDLevel(0);
  y->SetDataType(paddle::framework::proto::VarType::FP32);
  y->SetShape({784, 100});

  auto* op = global_block->AppendOp();
  op->SetType("mul");
  op->SetInput("X", {x->Name()});
  op->SetInput("Y", {y->Name()});

  auto* out = global_block->Var("Out");
  out->SetType(paddle::framework::proto::VarType::DENSE_TENSOR);
  out->SetShape({1000, 100});
  op->SetOutput("Out", {out->Name()});

  auto get_dist_attr = [](const VarDesc* var_desc) {
    auto shape = get_tensor_shape(var_desc);
    return TensorDistAttr(shape);
  };

  std::vector<int64_t> shape = {2, 4};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(shape, process_ids, dim_names);

  std::vector<int64_t> shape2 = {2, 2};
  std::vector<int64_t> process_ids2 = {0, 1, 2, 3};
  std::vector<std::string> dim_names2 = {"a", "b"};
  ProcessMesh process_mesh2(shape2, process_ids2, dim_names2);

  auto x_dist_attr = get_dist_attr(x);
  auto y_dist_attr = get_dist_attr(y);
  auto out_dist_attr = get_dist_attr(out);
  x_dist_attr.set_process_mesh(process_mesh);
  x_dist_attr.set_dims_mapping(std::vector<int64_t>({0, -1}));
  x_dist_attr.set_batch_dim(0);
  x_dist_attr.set_chunk_id(0);
  x_dist_attr.set_dynamic_dims(std::vector<bool>({true, false}));
  x_dist_attr.mark_annotated("process_mesh");
  x_dist_attr.mark_annotated("dims_mapping");
  EXPECT_EQ(x_dist_attr.process_mesh(), process_mesh);
  EXPECT_EQ(x_dist_attr.dims_mapping(), std::vector<int64_t>({0, -1}));
  EXPECT_EQ(x_dist_attr.batch_dim(), 0);
  EXPECT_EQ(x_dist_attr.chunk_id(), 0);
  EXPECT_EQ(x_dist_attr.dynamic_dims(), std::vector<bool>({true, false}));
  EXPECT_EQ(x_dist_attr.is_annotated("process_mesh"), true);
  EXPECT_EQ(x_dist_attr.is_annotated("dims_mapping"), true);
  EXPECT_EQ(x_dist_attr.verify(get_tensor_shape(x)), true);
  x_dist_attr.clear_annotated();
  EXPECT_EQ(x_dist_attr.annotated().empty(), true);

  std::stringstream x_sstream;
  x_sstream << x_dist_attr;
  EXPECT_EQ(x_sstream.str(), x_dist_attr.to_string());
  auto x_proto = phi::distributed::to_proto(x_dist_attr);
  TensorDistAttr new_x_dist_attr = get_dist_attr(x);
  new_x_dist_attr.from_proto(x_proto);
  EXPECT_EQ(x_dist_attr, new_x_dist_attr);

  y_dist_attr.set_process_mesh(process_mesh);
  y_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, 0}));
  y_dist_attr.set_batch_dim(-1);
  y_dist_attr.set_chunk_id(0);
  y_dist_attr.set_dynamic_dims(std::vector<bool>({false, true}));
  x_dist_attr.mark_annotated("batch_dim");
  x_dist_attr.mark_annotated("dynamic_dims");
  EXPECT_EQ(y_dist_attr.process_mesh(), process_mesh);
  EXPECT_EQ(y_dist_attr.dims_mapping(), std::vector<int64_t>({-1, 0}));
  EXPECT_EQ(y_dist_attr.batch_dim(), -1);
  EXPECT_EQ(y_dist_attr.chunk_id(), 0);
  EXPECT_EQ(y_dist_attr.dynamic_dims(), std::vector<bool>({false, true}));
  EXPECT_EQ(x_dist_attr.is_annotated("batch_dim"), true);
  EXPECT_EQ(x_dist_attr.is_annotated("dynamic_dims"), true);
  EXPECT_EQ(x_dist_attr.verify(get_tensor_shape(y)), true);

  out_dist_attr.set_process_mesh(process_mesh);
  out_dist_attr.set_dims_mapping(std::vector<int64_t>({0, 1}));
  out_dist_attr.set_batch_dim(1);
  out_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));
  EXPECT_EQ(out_dist_attr.process_mesh(), process_mesh);
  EXPECT_EQ(out_dist_attr.dims_mapping(), std::vector<int64_t>({0, 1}));
  EXPECT_EQ(out_dist_attr.batch_dim(), 1);
  EXPECT_EQ(out_dist_attr.dynamic_dims(), std::vector<bool>({false, false}));
  EXPECT_EQ(out_dist_attr.verify(get_tensor_shape(out)), true);

  OperatorDistAttr mul_dist_attr(*op);
  EXPECT_EQ(mul_dist_attr.impl_type(),
            paddle::distributed::auto_parallel::kDefault);
  EXPECT_EQ(mul_dist_attr.impl_idx(), 0);
  EXPECT_EQ(mul_dist_attr.chunk_id(), 0);
  EXPECT_EQ(mul_dist_attr.is_recompute(), false);
  EXPECT_EQ(mul_dist_attr.is_annotated("process_mesh"), false);
  EXPECT_EQ(mul_dist_attr.is_annotated("impl_type"), false);
  EXPECT_EQ(mul_dist_attr.is_annotated("impl_idx"), false);
  mul_dist_attr.set_input_dist_attr(x->Name(), x_dist_attr);
  mul_dist_attr.set_input_dist_attr(y->Name(), y_dist_attr);
  mul_dist_attr.set_output_dist_attr(out->Name(), out_dist_attr);
  mul_dist_attr.set_process_mesh(process_mesh2);
  mul_dist_attr.set_impl_type("dist_mul");
  mul_dist_attr.set_impl_idx(0);
  mul_dist_attr.set_chunk_id(1);
  mul_dist_attr.set_is_recompute(true);
  mul_dist_attr.mark_annotated("process_mesh");
  mul_dist_attr.mark_annotated("impl_type");
  mul_dist_attr.mark_annotated("impl_idx");
  EXPECT_NE(mul_dist_attr.input_dist_attr(x->Name()), x_dist_attr);
  EXPECT_NE(mul_dist_attr.input_dist_attr(y->Name()), y_dist_attr);
  EXPECT_NE(mul_dist_attr.output_dist_attr(out->Name()), out_dist_attr);
  EXPECT_EQ(mul_dist_attr.process_mesh(), process_mesh2);
  EXPECT_EQ(mul_dist_attr.input_dist_attr(x->Name()).process_mesh(),
            process_mesh2);
  EXPECT_EQ(mul_dist_attr.input_dist_attr(y->Name()).process_mesh(),
            process_mesh2);
  EXPECT_EQ(mul_dist_attr.impl_type(), "dist_mul");
  EXPECT_EQ(mul_dist_attr.impl_idx(), 0);
  EXPECT_EQ(mul_dist_attr.chunk_id(), 1);
  EXPECT_EQ(mul_dist_attr.is_recompute(), true);
  EXPECT_EQ(mul_dist_attr.is_annotated("process_mesh"), true);
  EXPECT_EQ(mul_dist_attr.is_annotated("impl_type"), true);
  EXPECT_EQ(mul_dist_attr.is_annotated("impl_idx"), true);
  EXPECT_EQ(mul_dist_attr.verify(op), true);
  mul_dist_attr.clear_annotated();
  EXPECT_EQ(mul_dist_attr.annotated().empty(), true);

  std::stringstream mul_sstream;
  mul_sstream << mul_dist_attr;
  EXPECT_EQ(mul_sstream.str(), mul_dist_attr.to_string());
  auto mul_proto = mul_dist_attr.to_proto();
  OperatorDistAttr new_mul_dist_attr(*op);
  new_mul_dist_attr.from_proto(mul_proto);
  EXPECT_EQ(mul_dist_attr, new_mul_dist_attr);
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace phi
