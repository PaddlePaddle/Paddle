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

#include <gtest/gtest.h>

#include <functional>
#include <iostream>
#include <string>

#include "paddle/cinn/backends/llvm/execution_engine.h"
#include "paddle/cinn/cinn.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/common/test_helper.h"
#include "paddle/cinn/hlir/framework/graph_compiler.h"
#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/op/use_ops.h"
#include "paddle/cinn/hlir/pe/nn.h"
#include "paddle/cinn/runtime/flags.h"

namespace cinn {
namespace hlir {
namespace framework {

using CCompute =
    std::function<std::shared_ptr<ir::Tensor>(const std::vector<ir::Tensor>)>;

Module LowerToModule(const std::string test_name,
                     const std::string func_name,
                     const std::shared_ptr<OpImpl> &impl,
                     std::vector<std::string> input_names,
                     const std::string &output_name,
                     std::vector<ir::Tensor> &inputs,  // NOLINT
                     std::vector<common::CINNValue> cinn_inputs,
                     const Target &target) {
  Module::Builder builder("module", target);

  cinn_inputs.emplace_back(output_name);
  common::CINNValuePack cinn_input = common::CINNValuePack{cinn_inputs};
  input_names.push_back(output_name);

  auto funcs = framework::GetFuncFromImpl(
      impl, cinn_input, inputs, input_names, func_name, target);

  for (auto func : funcs) {
    LOG(INFO) << "Test" << test_name << "'s Strategy, func is :\n" << func;
    builder.AddFunction(func);
  }
  return builder.Build();
}

TEST(Operator, Operator_Pool2d_Test0) {
  auto pool2d = Operator::Get("pool2d");
  Operator temp = *pool2d;
  auto strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");

  Expr N(1), C(3), H(8), W(8);
  Placeholder<float> A("A", {N, C, H, W});

  NodeAttr attrs;
  std::vector<int> kernel_size = {2, 2};
  std::vector<int> stride_size = {2, 2};
  std::vector<int> padding_size = {1, 1, 1, 1};
  std::string pool_type = "max";
  attrs.attr_store["kernel_size"] = kernel_size;
  attrs.attr_store["stride_size"] = stride_size;
  attrs.attr_store["padding_size"] = padding_size;
  attrs.attr_store["pool_type"] = pool_type;
  std::vector<ir::Tensor> inputs{A.tensor()};
  std::vector<Type> type{Float(32)};
  common::Target target = common::DefaultHostTarget();
  auto impl = OpStrategy::SelectImpl(strategy[pool2d](
      attrs, inputs, type, {{1, 3, 10, 10}, {1, 3, 5, 5}}, target));

  std::string func_name = "pool2d";
  auto module = LowerToModule("Operator_Pool2d_Test0",
                              func_name,
                              impl,
                              {"A"},
                              "B",
                              inputs,
                              {common::CINNValue(A)},
                              target);

  auto jit = backends::ExecutionEngine::Create({});

  jit->Link(module);
  auto fn = jit->Lookup("fn_" + func_name);
  CHECK(fn);
  auto fn_ = reinterpret_cast<void (*)(void *, int32_t)>(fn);

  cinn_buffer_t *A_buf =
      common::BufferBuilder(Float(32), {1, 3, 8, 8}).set_random().Build();
  cinn_buffer_t *B_buf =
      common::BufferBuilder(Float(32), {1, 3, 10, 10}).set_random().Build();
  cinn_buffer_t *C_buf =
      common::BufferBuilder(Float(32), {1, 3, 5, 5}).set_random().Build();
  cinn_pod_value_t a_arg(A_buf), b_arg(B_buf), c_arg(C_buf);
  cinn_pod_value_t args[] = {a_arg, b_arg, c_arg};
  fn_(args, 3);

  ASSERT_EQ(impl->name, "strategy.pool2d.x86");
  ASSERT_EQ(
      pool2d->description,
      "Do pooling on the height and width dimension of the input tensor.");
}

TEST(Operator, Operator_Pool2d_Test1) {
  auto pool2d = Operator::Get("pool2d");
  Operator temp = *pool2d;
  auto strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");

  Expr N(1), C(3), H(8), W(8);
  Placeholder<float> A("A", {N, C, H, W});

  NodeAttr attrs;
  std::vector<int> kernel_size = {2, 2};
  std::vector<int> stride_size = {2, 2};
  std::vector<int> padding_size = {1, 1, 1, 1};
  std::string pool_type = "avg";
  attrs.attr_store["kernel_size"] = kernel_size;
  attrs.attr_store["stride_size"] = stride_size;
  attrs.attr_store["padding_size"] = padding_size;
  attrs.attr_store["pool_type"] = pool_type;
  attrs.attr_store["ceil_mode"] = true;
  attrs.attr_store["exclusive"] = false;
  std::vector<ir::Tensor> inputs{A.tensor()};
  std::vector<Type> type{Float(32)};
  common::Target target = common::DefaultHostTarget();
  auto impl = OpStrategy::SelectImpl(strategy[pool2d](
      attrs, inputs, type, {{1, 3, 11, 11}, {1, 3, 5, 5}}, target));

  std::string func_name = "pool2d";

  auto module = LowerToModule("Operator_Pool2d_Test1",
                              func_name,
                              impl,
                              {"A"},
                              "B",
                              inputs,
                              {common::CINNValue(A)},
                              target);

  auto jit = backends::ExecutionEngine::Create({});

  jit->Link(module);
  auto fn = jit->Lookup("fn_" + func_name);
  CHECK(fn);
  auto fn_ = reinterpret_cast<void (*)(void *, int32_t)>(fn);

  cinn_buffer_t *A_buf =
      common::BufferBuilder(Float(32), {1, 3, 8, 8}).set_random().Build();
  cinn_buffer_t *B_buf =
      common::BufferBuilder(Float(32), {1, 3, 11, 11}).set_random().Build();
  cinn_buffer_t *C_buf =
      common::BufferBuilder(Float(32), {1, 3, 5, 5}).set_random().Build();
  cinn_pod_value_t a_arg(A_buf), b_arg(B_buf), c_arg(C_buf);
  cinn_pod_value_t args[] = {a_arg, b_arg, c_arg};
  fn_(args, 3);

  ASSERT_EQ(impl->name, "strategy.pool2d.x86");
  ASSERT_EQ(
      pool2d->description,
      "Do pooling on the height and width dimension of the input tensor.");
}

TEST(Operator, Operator_Pool2d_Test2) {
  auto pool2d = Operator::Get("pool2d");
  Operator temp = *pool2d;
  auto strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");

  Expr N(1), H(8), W(8), C(3);
  Placeholder<float> A("A", {N, H, W, C});

  NodeAttr attrs;
  std::vector<int> kernel_size = {2, 2};
  std::vector<int> stride_size = {2, 2};
  std::vector<int> padding_size = {1, 1, 1, 1};
  std::string pool_type = "avg";
  std::string data_format = "NHWC";
  attrs.attr_store["kernel_size"] = kernel_size;
  attrs.attr_store["stride_size"] = stride_size;
  attrs.attr_store["padding_size"] = padding_size;
  attrs.attr_store["pool_type"] = pool_type;
  attrs.attr_store["ceil_mode"] = true;
  attrs.attr_store["exclusive"] = true;
  attrs.attr_store["data_format"] = data_format;
  std::vector<ir::Tensor> inputs{A.tensor()};
  std::vector<Type> type{Float(32)};
  common::Target target = common::DefaultHostTarget();
  auto impl = OpStrategy::SelectImpl(strategy[pool2d](
      attrs, inputs, type, {{1, 11, 11, 3}, {1, 5, 5, 3}}, target));

  std::string func_name = "pool2d";

  auto module = LowerToModule("Operator_Pool2d_Test2",
                              func_name,
                              impl,
                              {"A"},
                              "B",
                              inputs,
                              {common::CINNValue(A)},
                              target);

  auto jit = backends::ExecutionEngine::Create({});

  jit->Link(module);
  auto fn = jit->Lookup("fn_" + func_name);
  CHECK(fn);
  auto fn_ = reinterpret_cast<void (*)(void *, int32_t)>(fn);

  cinn_buffer_t *A_buf =
      common::BufferBuilder(Float(32), {1, 8, 8, 3}).set_random().Build();
  cinn_buffer_t *B_buf =
      common::BufferBuilder(Float(32), {1, 11, 11, 3}).set_random().Build();
  cinn_buffer_t *C_buf =
      common::BufferBuilder(Float(32), {1, 5, 5, 3}).set_random().Build();
  cinn_pod_value_t a_arg(A_buf), b_arg(B_buf), c_arg(C_buf);
  cinn_pod_value_t args[] = {a_arg, b_arg, c_arg};
  fn_(args, 3);

  ASSERT_EQ(impl->name, "strategy.pool2d.x86");
  ASSERT_EQ(
      pool2d->description,
      "Do pooling on the height and width dimension of the input tensor.");
}

TEST(Operator, Operator_Pool3d_Test0) {
  auto pool3d = Operator::Get("pool3d");
  Operator temp = *pool3d;
  auto strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");

  Expr N(1), D(8), H(8), W(8), C(3);
  Placeholder<float> A("A", {N, D, H, W, C});

  NodeAttr attrs;
  std::vector<int> kernel_size = {2, 2, 2};
  std::vector<int> stride_size = {2, 2, 2};
  std::vector<int> padding_size = {1, 1, 1, 1, 1, 1};
  std::string pool_type = "max";
  std::string data_format = "NDHWC";
  attrs.attr_store["kernel_size"] = kernel_size;
  attrs.attr_store["stride_size"] = stride_size;
  attrs.attr_store["padding_size"] = padding_size;
  attrs.attr_store["pool_type"] = pool_type;
  attrs.attr_store["ceil_mode"] = false;
  attrs.attr_store["exclusive"] = true;
  attrs.attr_store["data_format"] = data_format;
  std::vector<ir::Tensor> inputs{A.tensor()};
  std::vector<Type> type{Float(32)};
  common::Target target = common::DefaultHostTarget();
  auto impl = OpStrategy::SelectImpl(strategy[pool3d](
      attrs, inputs, type, {{1, 11, 11, 11, 3}, {1, 5, 5, 5, 3}}, target));

  std::string func_name = "pool3d";
  auto module = LowerToModule("Operator_Pool3d_Test0",
                              func_name,
                              impl,
                              {"A"},
                              "B",
                              inputs,
                              {common::CINNValue(A)},
                              target);

  auto jit = backends::ExecutionEngine::Create({});

  jit->Link(module);
  auto fn = jit->Lookup("fn_" + func_name);
  CHECK(fn);
  auto fn_ = reinterpret_cast<void (*)(void *, int32_t)>(fn);

  cinn_buffer_t *A_buf =
      common::BufferBuilder(Float(32), {1, 8, 8, 8, 3}).set_random().Build();
  cinn_buffer_t *B_buf =
      common::BufferBuilder(Float(32), {1, 11, 11, 11, 3}).set_random().Build();
  cinn_buffer_t *C_buf =
      common::BufferBuilder(Float(32), {1, 5, 5, 5, 3}).set_random().Build();
  cinn_pod_value_t a_arg(A_buf), b_arg(B_buf), c_arg(C_buf);
  cinn_pod_value_t args[] = {a_arg, b_arg, c_arg};
  fn_(args, 3);

  ASSERT_EQ(impl->name, "strategy.pool3d.x86");
  ASSERT_EQ(pool3d->description,
            "Do pooling on the depth, height and width dimension of the input "
            "tensor.");
}

TEST(Operator, Operator_Pool1d_Test0) {
  auto pool1d = Operator::Get("pool1d");
  Operator temp = *pool1d;
  auto strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");

  Expr N(1), W(8), C(3);
  Placeholder<float> A("A", {N, W, C});

  NodeAttr attrs;
  std::vector<int> kernel_size = {2};
  std::vector<int> stride_size = {2};
  std::vector<int> padding_size = {1, 1};
  std::string pool_type = "max";
  std::string data_format = "NWC";
  attrs.attr_store["kernel_size"] = kernel_size;
  attrs.attr_store["stride_size"] = stride_size;
  attrs.attr_store["padding_size"] = padding_size;
  attrs.attr_store["pool_type"] = pool_type;
  attrs.attr_store["ceil_mode"] = false;
  attrs.attr_store["exclusive"] = true;
  attrs.attr_store["data_format"] = data_format;
  std::vector<ir::Tensor> inputs{A.tensor()};
  std::vector<Type> type{Float(32)};
  common::Target target = common::DefaultHostTarget();
  auto impl = OpStrategy::SelectImpl(
      strategy[pool1d](attrs, inputs, type, {{1, 11, 3}, {1, 5, 3}}, target));

  std::string func_name = "pool1d";
  auto module = LowerToModule("Operator_Pool1d_Test0",
                              func_name,
                              impl,
                              {"A"},
                              "B",
                              inputs,
                              {common::CINNValue(A)},
                              target);

  auto jit = backends::ExecutionEngine::Create({});

  jit->Link(module);
  auto fn = jit->Lookup("fn_" + func_name);
  CHECK(fn);
  auto fn_ = reinterpret_cast<void (*)(void *, int32_t)>(fn);

  cinn_buffer_t *A_buf =
      common::BufferBuilder(Float(32), {1, 8, 3}).set_random().Build();
  cinn_buffer_t *B_buf =
      common::BufferBuilder(Float(32), {1, 11, 3}).set_random().Build();
  cinn_buffer_t *C_buf =
      common::BufferBuilder(Float(32), {1, 5, 3}).set_random().Build();
  cinn_pod_value_t a_arg(A_buf), b_arg(B_buf), c_arg(C_buf);
  cinn_pod_value_t args[] = {a_arg, b_arg, c_arg};
  fn_(args, 3);

  ASSERT_EQ(impl->name, "strategy.pool1d.x86");
  ASSERT_EQ(pool1d->description,
            "Do pooling on the width dimension of the input tensor.");
}

TEST(Operator, Operator_Select_Test0) {
  auto select = Operator::Get("select");
  Operator temp = *select;
  auto strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  auto infer_shape_func =
      Operator::GetAttrs<InferShapeFunction>("infershape")[select];

  Expr C(16), H(64), W(64);
  Placeholder<bool> condition("condition", {C, H, W});
  Placeholder<float> true_value("true_value", {C, H, W});
  Placeholder<float> false_value("false_value", {C, H, W});

  NodeAttr attrs;
  std::vector<ir::Tensor> inputs{
      condition.tensor(), true_value.tensor(), false_value.tensor()};
  std::vector<Type> type{Float(32)};
  const common::Target target = common::DefaultHostTarget();

  const std::vector<framework::shape_t> input_shapes = {
      {16, 64, 64}, {16, 64, 64}, {16, 64, 64}};
  auto infer_shape = infer_shape_func(input_shapes, attrs.attr_store);
  ASSERT_EQ(infer_shape[0][0], 16);
  ASSERT_EQ(infer_shape[0][1], 64);
  ASSERT_EQ(infer_shape[0][2], 64);

  auto impl = OpStrategy::SelectImpl(
      strategy[select](attrs, inputs, type, {{16, 64, 64}}, target));

  std::string func_name = "select";
  std::vector<std::string> input_names = {
      "condition", "true_value", "false_value"};
  std::vector<common::CINNValue> cinn_inputs = {common::CINNValue(condition),
                                                common::CINNValue(true_value),
                                                common::CINNValue(false_value)};

  auto module = LowerToModule("Operator_Select_Test0",
                              func_name,
                              impl,
                              std::move(input_names),
                              "output",
                              inputs,
                              cinn_inputs,
                              target);

  auto jit = backends::ExecutionEngine::Create({});

  jit->Link(module);
  auto fn = jit->Lookup("fn_" + func_name);
  CHECK(fn);
  auto fn_ = reinterpret_cast<void (*)(void *, int32_t)>(fn);

  cinn_buffer_t *A_buf =
      common::BufferBuilder(Bool(), {16, 64, 64}).set_random().Build();
  cinn_buffer_t *B_buf =
      common::BufferBuilder(Float(32), {16, 64, 64}).set_random().Build();
  cinn_buffer_t *C_buf =
      common::BufferBuilder(Float(32), {16, 64, 64}).set_random().Build();
  cinn_buffer_t *D_buf =
      common::BufferBuilder(Float(32), {16, 64, 64}).set_random().Build();

  cinn_pod_value_t a_arg(A_buf), b_arg(B_buf), c_arg(C_buf), d_arg(D_buf);
  cinn_pod_value_t args[] = {a_arg, b_arg, c_arg, d_arg};
  fn_(args, 4);

  auto condition_ = reinterpret_cast<int8_t *>(A_buf->memory);
  auto true_value_ = reinterpret_cast<float *>(B_buf->memory);
  auto false_value_ = reinterpret_cast<float *>(C_buf->memory);
  auto output_ = reinterpret_cast<float *>(D_buf->memory);

  for (int i = 0; i < A_buf->num_elements(); i++) {
    if (static_cast<bool>(condition_[i])) {
      ASSERT_EQ(output_[i], true_value_[i]);
    } else {
      ASSERT_EQ(output_[i], false_value_[i]);
    }
  }

  ASSERT_EQ(impl->name, "strategy.select.x86");
  ASSERT_EQ(select->description,
            "This operator implements the meta op 'Select'.");
}

TEST(Operator, Operator_Reverse_Test0) {
  auto reverse = Operator::Get("reverse");
  Operator temp = *reverse;
  auto strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");

  int c = 16, h = 64, w = 64;
  Expr C(c), H(h), W(w);
  Placeholder<float> A("A", {C, H, W});

  NodeAttr attrs;
  std::vector<int> axis = {1, 2};
  attrs.attr_store["axis"] = axis;
  std::vector<ir::Tensor> inputs{A.tensor()};
  std::vector<Type> type{Float(32)};
  common::Target target = common::DefaultHostTarget();

  auto impl = OpStrategy::SelectImpl(
      strategy[reverse](attrs, inputs, type, {{c, h, w}}, target));

  std::string func_name = "reverse";
  auto module = LowerToModule("Operator_Reverse_Test0",
                              func_name,
                              impl,
                              {"A"},
                              "B",
                              inputs,
                              {common::CINNValue(A)},
                              target);

  auto jit = backends::ExecutionEngine::Create({});

  jit->Link(module);
  auto fn = jit->Lookup("fn_" + func_name);
  CHECK(fn);
  auto fn_ = reinterpret_cast<void (*)(void *, int32_t)>(fn);

  cinn_buffer_t *A_buf =
      common::BufferBuilder(Float(32), {c, h, w}).set_random().Build();
  cinn_buffer_t *B_buf =
      common::BufferBuilder(Float(32), {c, h, w}).set_random().Build();
  cinn_pod_value_t a_arg(A_buf), b_arg(B_buf);
  cinn_pod_value_t args[] = {a_arg, b_arg};
  fn_(args, 2);

  auto input = reinterpret_cast<float *>(A_buf->memory);
  auto output = reinterpret_cast<float *>(B_buf->memory);

  for (int ida = 0; ida < c; ++ida) {
    for (int idb = 0; idb < h; ++idb) {
      for (int idc = 0; idc < w; ++idc) {
        int index = ida * h * w + idb * h + idc;
        int index_ = ida * h * w + (h - 1 - idb) * h + (w - 1 - idc);
        ASSERT_EQ(output[index], input[index_]);
      }
    }
  }

  ASSERT_EQ(impl->name, "strategy.reverse.x86");
  ASSERT_EQ(reverse->description,
            "This operator implements the meta op reverse.");
}

TEST(Operator, Operator_Transpose_Test0) {
  auto transpose = Operator::Get("transpose");
  Operator temp = *transpose;
  auto strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  auto infer_shape_func =
      Operator::GetAttrs<InferShapeFunction>("infershape")[transpose];

  int n = 16, c = 3, h = 32, w = 32;
  Expr N(n), C(c), H(h), W(w);
  Placeholder<float> A("A", {N, C, H, W});

  NodeAttr attrs;
  std::vector<int> axis = {0, 2, 3, 1};
  attrs.attr_store["axis"] = axis;
  std::vector<ir::Tensor> inputs{A.tensor()};
  std::vector<Type> type{Float(32)};
  common::Target target = common::DefaultHostTarget();

  auto infer_shape = infer_shape_func({{n, c, h, w}}, attrs.attr_store);
  ASSERT_EQ(infer_shape[0][0], n);
  ASSERT_EQ(infer_shape[0][1], h);
  ASSERT_EQ(infer_shape[0][2], w);
  ASSERT_EQ(infer_shape[0][3], c);

#ifndef CINN_WITH_CUDA
  using InferLayoutFunction =
      std::function<std::vector<std::vector<std::string>>(
          const std::vector<framework::shape_t> &,
          const std::vector<std::string> &,
          const framework::NodeAttr &,
          const Target &target)>;
  auto infer_layout_func =
      Operator::GetAttrs<InferLayoutFunction>("inferlayout")[transpose];
  auto infer_layout =
      infer_layout_func({{n, c, h, w}}, {"NCHW"}, attrs, target);
  ASSERT_EQ(infer_layout[0][0], "NHWC");
#endif

  auto input_shape = {n, c, h, w};
  auto output_shape = {n, h, w, c};

  auto impl = OpStrategy::SelectImpl(
      strategy[transpose](attrs, inputs, type, {output_shape}, target));

  std::string func_name = "transpose";
  auto module = LowerToModule("Operator_Transpose_Test0",
                              func_name,
                              impl,
                              {"A"},
                              "B",
                              inputs,
                              {common::CINNValue(A)},
                              target);

  auto jit = backends::ExecutionEngine::Create({});

  jit->Link(module);
  auto fn = jit->Lookup("fn_" + func_name);
  CHECK(fn);
  auto fn_ = reinterpret_cast<void (*)(void *, int32_t)>(fn);

  cinn_buffer_t *A_buf =
      common::BufferBuilder(Float(32), input_shape).set_random().Build();
  cinn_buffer_t *B_buf =
      common::BufferBuilder(Float(32), output_shape).set_random().Build();
  cinn_pod_value_t a_arg(A_buf), b_arg(B_buf);
  cinn_pod_value_t args[] = {a_arg, b_arg};
  fn_(args, 2);

  auto input = reinterpret_cast<float *>(A_buf->memory);
  auto output = reinterpret_cast<float *>(B_buf->memory);

  for (int idx = 0; idx < n; ++idx) {
    for (int idy = 0; idy < h; ++idy) {
      for (int idz = 0; idz < w; ++idz) {
        for (int id = 0; id < c; ++id) {
          // (n, h, w, c) (idx, idy, idz, id)
          int index = idx * (h * w * c) + idy * (w * c) + idz * c + id;
          // (n, c, h, w) (idx, id, idy, idz)
          int _index = idx * (c * h * w) + id * (h * w) + idy * h + idz;
          CHECK_EQ(output[index], input[_index]);
        }
      }
    }
  }

  ASSERT_EQ(impl->name, "strategy.transpose.x86");
  ASSERT_EQ(transpose->description,
            "This operator implements the meta op transpose.");
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
