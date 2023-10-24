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

#include "paddle/cinn/hlir/pe/transform.h"

#include <algorithm>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/op/op_util.h"
#include "paddle/cinn/hlir/pe/elementwise.h"
#include "paddle/cinn/hlir/pe/ir_schedule_pe.h"
#include "paddle/cinn/hlir/pe/nn.h"
#include "paddle/cinn/hlir/pe/schedule.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace hlir {
namespace op {
using common::_CINNValuePack_;
using common::CINNValue;
using common::CINNValuePack;
using framework::OpStrategy;
using framework::shape_t;
using framework::StrategyFunction;

std::shared_ptr<OpStrategy> StrategyForMatMul(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  const auto &attr_store = attrs.attr_store;
  bool trans_a = SafeGetAttr(attr_store, "trans_a", false);
  bool trans_b = SafeGetAttr(attr_store, "trans_b", false);
  float alpha = SafeGetAttr(attr_store, "alpha", 1.0f);

  const auto &shape_A = ToPodVector<int>(inputs[0]->shape);
  const auto &shape_B = ToPodVector<int>(inputs[1]->shape);

  const auto &new_shape =
      pe::utils::GetMatmulNewShapes({shape_A, shape_B}, trans_a, trans_b);

  const auto &new_shape_A = new_shape[0];
  const auto &new_shape_B = new_shape[1];
  const auto &output_shape = new_shape[2];

  framework::CINNCompute matmul_compute([=](lang::Args args,
                                            lang::RetValue *ret) {
    CHECK(!args.empty())
        << "The input arguments of Matmul compute is empty! Please check.\n";
    CINNValuePack pack_args = args[0];
    CHECK_GE(pack_args.size(), 2U)
        << "at least 2 input tensors for Matmul compute\n";
    Expr A = pack_args[0];
    Expr B = pack_args[1];
    CHECK(A.as_tensor());
    CHECK(B.as_tensor());

    CHECK_GE(pack_args.size(), 3);
    CHECK(pack_args[2].is_string());
    std::string tensor_name = pack_args[2].operator std::string();

    auto tensor_A = A.as_tensor_ref();
    auto tensor_B = B.as_tensor_ref();
    auto stages = CreateStages({tensor_A, tensor_B});

    auto new_shape_A_e = ToCinnExprs(new_shape_A);
    auto new_shape_B_e = ToCinnExprs(new_shape_B);

    auto new_A = tensor_A->Reshape(new_shape_A_e, stages);
    auto new_B = tensor_B->Reshape(new_shape_B_e, stages);

    std::vector<ir::Tensor> out;
    if (target.arch == Target::Arch::X86) {
#ifdef CINN_WITH_MKL_CBLAS
      out = pe::MatmulMKL(new_A,
                          new_B,
                          trans_a,
                          trans_b,
                          alpha,
                          UniqName("MatmulMKL_output"),
                          target);
#else
      out = pe::MatmulV2(new_A,
                         new_B,
                         trans_a,
                         trans_b,
                         alpha,
                         UniqName("MatmulV2_output"),
                         target);
#endif
    } else {
      out = pe::Matmul(new_A, new_B, trans_a, trans_b, alpha, tensor_name);
    }

    std::vector<CINNValue> res;
    for (auto &t : out) {
      stages->InsertLazily(t);
    }

    for (auto &t : out) {
      res.push_back(CINNValue(t));
    }
    res.push_back(CINNValue(stages));
    *ret = CINNValuePack{res};
  });

  framework::CINNSchedule matmul_schedule([=](lang::Args args,
                                              lang::RetValue *ret) {
    CHECK(!args.empty())
        << "The input argument of matmul schedule is empty! Please check.\n";
    CINNValuePack arg_pack = args[0];
    std::vector<CINNValue> results =
        pe::IRCudaScheduleMatMul(arg_pack, output_shape, target);
    *ret = CINNValuePack({results});
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(matmul_compute, matmul_schedule, "strategy.matmul.x86", 1);

  return strategy;
}

std::vector<std::vector<int>> InferShapeForMatMul(
    const std::vector<std::vector<int>> &inputs_shape,
    const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 2UL)
      << "The input's shape size should be 2! Please check again.";
  bool trans_a = SafeGetAttr(attrs, "trans_a", false);
  bool trans_b = SafeGetAttr(attrs, "trans_b", false);

  VLOG(4) << "During the matmul shape inference, origin shape_A: "
          << utils::Join(inputs_shape[0], ", ");
  VLOG(4) << "During the matmul shape inference, origin shape_B: "
          << utils::Join(inputs_shape[1], ", ");

  const auto &new_shape =
      pe::utils::GetMatmulNewShapes(inputs_shape, trans_a, trans_b);

  const auto &new_shape_A = new_shape[0];
  const auto &new_shape_B = new_shape[1];
  const auto &output_shape = new_shape[2];

  VLOG(4) << "During the matmul shape inference, new_shape_A: "
          << utils::Join(new_shape_A, ", ");
  VLOG(4) << "During the matmul shape inference, new_shape_B: "
          << utils::Join(new_shape_B, ", ");
  VLOG(4) << "During the matmul shape inference, output_shape: "
          << utils::Join(output_shape, ", ");

  std::vector<std::vector<int>> res{output_shape};
  return res;
}

std::vector<Type> InferDtypeForMatMul(const std::vector<Type> &inputs_type,
                                      const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_type.size(), 2UL)
      << "The input's type size should be 2! Please check again.";
  CHECK_EQ(inputs_type[0], inputs_type[1])
      << "The input's types should be equal! Please check again.";

  std::vector<Type> res{inputs_type[0]};
  return res;
}

std::vector<std::vector<std::string>> InferLayoutForMatMul(
    const std::vector<framework::shape_t> &input_shapes,
    const std::vector<std::string> &input_layouts,
    const framework::NodeAttr &attrs,
    const Target &target) {
  CHECK_EQ(input_layouts.size(), 2U)
      << "The input's layouts size is not 2! Please check again.";
  CHECK_EQ(input_shapes.size(), 2U) << "mul should have 2 input shapes";
  std::vector<std::string> new_input_layouts = input_layouts;
  for (int i = 0; i < input_shapes.size(); i++) {
    if (input_shapes[i].size() > 4) {
      // alter input layout back
      new_input_layouts[i] = "NCHW";
    }
  }

  return {{"", ""}, new_input_layouts};
}

std::shared_ptr<OpStrategy> StrategyForSplit(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  // get attribute
  std::vector<int> sections;
  int axis = 0;
  if (attrs.attr_store.find("num_or_sections") != attrs.attr_store.end()) {
    sections =
        absl::get<std::vector<int>>(attrs.attr_store.at("num_or_sections"));
  }
  if (attrs.attr_store.find("axis") != attrs.attr_store.end()) {
    axis = absl::get<int>(attrs.attr_store.at("axis"));
  }
  if (axis < 0) axis += static_cast<int>(output_shapes[0].size());

  CHECK(!output_shapes.empty())
      << "The Spilt Op's output shape list should not empty.";
  CHECK_LT(axis, static_cast<int>(output_shapes[0].size()));
  CHECK(!sections.empty())
      << "The Split op doesn't find [num_or_sections] attrbute! It it a "
         "mandatory attribute ! Please check.";

  framework::CINNCompute split_compute(
      [=](lang::Args args, lang::RetValue *ret) {
        CHECK(!args.empty())
            << "The input arguments of split compute is empty! Please check.";
        CINNValuePack pack_args = args[0];
        CHECK(!pack_args.empty())
            << "The input tensors of split compute is empty! Please check.";
        Expr A_expr = pack_args[0];
        CHECK(A_expr.as_tensor());
        ir::Tensor A = A_expr.as_tensor_ref();

        std::vector<std::string> tensor_names;
        CHECK_EQ(pack_args.size(), output_shapes.size() + 1);
        for (int idx = 1; idx < pack_args.size(); ++idx) {
          CHECK(pack_args[idx].is_string());
          tensor_names.push_back(pack_args[idx].operator std::string());
        }

        auto out = pe::Split(A, axis, output_shapes, tensor_names);
        auto stages = CreateStages(out);

        std::vector<CINNValue> res;
        for (int i = 0; i < out.size(); ++i) {
          res.emplace_back(out[i]);
        }
        res.emplace_back(stages);
        *ret = CINNValuePack{res};
      });

  framework::CINNSchedule split_schedule(
      [=](lang::Args args, lang::RetValue *ret) {
        CHECK(!args.empty())
            << "The input argument of split schedule is empty! Please check.";
        CINNValuePack arg_pack = args[0];
        std::vector<Expr> vec_ast;
        for (int i = 0; i < arg_pack.size(); i++) {
          if (arg_pack[i].is_expr()) {
            Expr temp = arg_pack[i];
            vec_ast.emplace_back(temp);
          }
        }
        CHECK(!vec_ast.empty());
        ir::ModuleExpr mod_expr(vec_ast);
        ir::IRSchedule ir_sch(mod_expr);
        ir_sch.MergeExprs();
        pe::IRCudaSplitSchedule(ir_sch, output_shapes, axis, target);
        std::vector<CINNValue> res{
            CINNValue(ir_sch.GetModule().GetExprs().at(0))};
        *ret = CINNValuePack{res};
      });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(split_compute, split_schedule, "strategy.split.x86", 1);

  return strategy;
}

std::vector<std::vector<int>> InferShapeForSplit(
    const std::vector<std::vector<int>> &inputs_shape,
    const framework::AttrMapType &attrs) {
  std::vector<int> sections;
  if (attrs.find("num_or_sections") != attrs.end()) {
    sections = absl::get<std::vector<int>>(attrs.at("num_or_sections"));
  } else {
    LOG(FATAL) << "The Split op doesn't find [num_or_sections] attrbute! It it "
                  "a mandatory attribute ! Please check.";
  }

  if (inputs_shape.empty()) {
    std::vector<std::vector<int>> ret;
    if (sections.size() == 1) {
      ret.resize(sections[0]);
    } else {
      ret.resize(sections.size());
    }
    return ret;
  }
  CHECK_GE(inputs_shape.size(), 1U)
      << "The input's shape size should be no less than 1! Please check again.";

  int axis = 0;
  if (attrs.find("axis") != attrs.end()) {
    axis = absl::get<int>(attrs.at("axis"));
    if (axis < 0) {
      axis += inputs_shape[0].size();
    }
  }

  // check sections valid
  int output_size = sections.size();
  int pivot = inputs_shape[0][axis];

  auto real_sections = sections;
  if (output_size == 1) {
    // if the 'sections' is a number, the tensor will split to 'sections'
    // sub-tensor, each sub-tensor length A[axis] / 'sections'
    output_size = sections[0];
    CHECK_EQ(pivot % output_size, 0)
        << "If the attribute 'num_or_sections' is a number, it should be "
           "divisible by the "
           "axis's dimension of inputs A ! Please check.";
    real_sections.assign(output_size, pivot / output_size);
  } else {
    // else the tensor will split to sections.size sub-tensor, each sub-tensor
    // length sections[i] The sections may have at most one '-1' in sections,
    // that means its value should be inferred by others.
    int section_sum = 0, neg_index = -1;
    for (int i = 0; i < output_size; ++i) {
      if (sections[i] > 0) {
        section_sum += sections[i];
      } else if (sections[i] == -1 && neg_index < 0) {
        neg_index = i;
      } else {
        if (sections[i] == 0) {
          LOG(FATAL) << "The attribute 'num_or_sections' should not has 0 ! "
                        "Please check.";
        } else {
          LOG(FATAL) << "The attribute 'num_or_sections' can only have at most "
                        "one '-1' ! Please check.";
        }
      }
    }

    if (neg_index >= 0) {
      // has '-1' in sections
      real_sections[neg_index] = pivot - section_sum;
    } else {
      CHECK_EQ(pivot, section_sum)
          << "The sum of attr sections should be equal with the axis's "
             "dimension value of "
             "inputs A in Split ! Please check.";
    }
  }

  std::vector<std::vector<int>> outputs_shape(output_size, inputs_shape[0]);
  for (int i = 0; i < output_size; ++i) {
    outputs_shape[i][axis] = real_sections[i];
  }
  return outputs_shape;
}

std::vector<Type> InferDtypeForSplit(const std::vector<Type> &inputs_type,
                                     const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty())
      << "The input's type size is 0! Please check again.";

  std::vector<int> sections;
  if (attrs.find("num_or_sections") != attrs.end()) {
    sections = absl::get<std::vector<int>>(attrs.at("num_or_sections"));
  } else {
    LOG(FATAL) << "The Split op doesn't find [num_or_sections] attrbute! It it "
                  "a mandatory attribute ! Please check.";
  }

  int output_size = sections.size();
  if (output_size == 1) {
    output_size = sections[0];
  }

  std::vector<Type> res(output_size, inputs_type[0]);
  return res;
}

std::vector<std::vector<std::string>> InferLayoutForSplit(
    const std::vector<framework::shape_t> &input_shapes,
    const std::vector<std::string> &input_layouts,
    const framework::NodeAttr &attrs,
    const Target &target) {
  CHECK(!input_layouts.empty())
      << "The input's layout size is 0! Please check again.";
  std::vector<int> sections;
  if (attrs.attr_store.find("num_or_sections") != attrs.attr_store.end()) {
    sections =
        absl::get<std::vector<int>>(attrs.attr_store.at("num_or_sections"));
  } else {
    LOG(FATAL) << "The Split op doesn't find [num_or_sections] attrbute! It it "
                  "a mandatory attribute ! Please check.";
  }

  int output_size = sections.size();
  if (output_size == 1) {
    output_size = sections[0];
  }

  std::vector<std::string> output_layout(output_size, input_layouts[0]);
  return {output_layout, input_layouts};
}

std::shared_ptr<OpStrategy> StrategyForConcat(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  framework::CINNCompute concat_compute([=](lang::Args args,
                                            lang::RetValue *ret) {
    CHECK(!args.empty())
        << "The input arguments of Concat compute is empty! Please check.\n";
    CHECK(!out_type.empty())
        << "Output type of Concat is empty! Please check.\n";
    CINNValuePack pack_args = args[0];
    int input_size = pack_args.size() - 1;
    CHECK_GE(input_size, 1UL)
        << "at least 2 input tensors for Concat compute\n";
    CHECK(!output_shapes.empty());
    int axis = 0;
    if (attrs.attr_store.count("axis")) {
      axis = absl::get<int>(attrs.attr_store.at("axis"));
    }

    std::vector<ir::Tensor> input_tensors;
    for (int i = 0; i < input_size; i++) {
      Expr tensor = pack_args[i];
      CHECK(tensor.as_tensor());
      input_tensors.push_back(tensor.as_tensor_ref());
    }

    CHECK(pack_args[input_size].is_string());
    std::string tensor_name = pack_args[input_size].operator std::string();

    auto stages = CreateStages(input_tensors);
    auto out = pe::Concat(input_tensors, axis, tensor_name);
    stages->InsertLazily(out);

    *ret = CINNValuePack({CINNValue(out), CINNValue(stages)});
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(concat_compute,
                    GetInjectiveScheduleFunc(output_shapes, target, false),
                    "strategy.concat.x86",
                    1);
  return strategy;
}

std::vector<std::vector<int>> InferShapeForConcat(
    const std::vector<std::vector<int>> &inputs_shape,
    const framework::AttrMapType &attrs) {
  CHECK_GE(inputs_shape.size(), 1UL)
      << "The input's shape size should be no less than 2! Please check again.";
  int axis = 0;
  for (auto &iter : attrs) {
    if (iter.first == "axis") {
      axis = absl::get<int>(iter.second);
      break;
    }
  }

  if (axis < 0) axis += inputs_shape[0].size();
  std::vector<int> output_shape = inputs_shape[0];
  CHECK(axis >= 0 && axis < inputs_shape[0].size())
      << "In Concat op, the attribute `axis` should be >= 0 and < input "
         "shape's size, please check!";

  int input_dim = inputs_shape[0].size();
  for (int i = 1; i < inputs_shape.size(); i++) {
    CHECK_EQ(inputs_shape[i].size(), input_dim)
        << "Dimensions of inputs tensors in Concat should be equal! Please "
           "check.";

    for (int j = 0; j < input_dim; j++) {
      if (j != axis) {
        CHECK_EQ(inputs_shape[0][j], inputs_shape[i][j])
            << "The " << j << "-th dimension of input[0] and input[" << i
            << "] should be the same, but here input[0].shape=["
            << cinn::utils::Join(inputs_shape[0], ", ") << "], input[" << i
            << "].shape=[" << cinn::utils::Join(inputs_shape[i], ", ")
            << "]! Please check.";
      }
    }

    output_shape[axis] += inputs_shape[i][axis];
  }

  std::vector<std::vector<int>> res{output_shape};
  return res;
}

std::vector<Type> InferDtypeForConcat(const std::vector<Type> &inputs_type,
                                      const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty())
      << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

std::vector<std::vector<std::string>> InferLayoutForConcat(
    const std::vector<framework::shape_t> &input_shapes,
    const std::vector<std::string> &input_layouts,
    const framework::NodeAttr &attrs,
    const Target &target) {
  CHECK_GE(input_layouts.size(), 1UL)
      << "The input's layout size is less than 2! Please check again.";
  return {{input_layouts[0]}, input_layouts};
}

std::shared_ptr<OpStrategy> StrategyForMul(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  CHECK_EQ(inputs.size(), 2UL) << "mul should have 2 input";
  const auto &attr_store = attrs.attr_store;
  int x_num_col_dims = SafeGetAttr(attr_store, "x_num_col_dims", 1);
  int y_num_col_dims = SafeGetAttr(attr_store, "y_num_col_dims", 1);
  bool is_infer = SafeGetAttr(attr_store, "is_infer", false);

  const auto &shape_A = ToPodVector<int>(inputs[0]->shape);
  const auto &shape_B = ToPodVector<int>(inputs[1]->shape);

  const auto &new_shape = pe::utils::GetMulNewShapes(
      {shape_A, shape_B}, x_num_col_dims, y_num_col_dims, is_infer);

  const auto &new_shape_A = new_shape[0];
  const auto &new_shape_B = new_shape[1];
  const auto &output_shape = new_shape[2];

  framework::CINNCompute mul_compute(
      [=](lang::Args args, lang::RetValue *ret) {
        CHECK(!args.empty())
            << "The input arguments of Mul compute is empty! Please check.\n";
        CINNValuePack pack_args = args[0];
        CHECK_GE(pack_args.size(), 2U)
            << "at least 2 input tensors for Mul compute\n";
        Expr A = pack_args[0];
        Expr B = pack_args[1];
        CHECK(A.as_tensor());
        CHECK(B.as_tensor());

        auto A_tensor = A.as_tensor_ref();
        auto B_tensor = B.as_tensor_ref();
        auto stages = CreateStages({A_tensor, B_tensor});

        auto new_shape_A_e = ToCinnExprs(new_shape_A);
        auto new_shape_B_e = ToCinnExprs(new_shape_B);

        auto new_A = A_tensor->Reshape(new_shape_A_e, stages);
        auto new_B = B_tensor->Reshape(new_shape_B_e, stages);

        std::vector<ir::Tensor> out;
        CHECK(pack_args.back().is_string());
        std::string tensor_name = pack_args.back().operator std::string();

        if (target.arch == Target::Arch::X86) {
#ifdef CINN_WITH_MKL_CBLAS
          out = pe::MatmulMKL(
              new_A, new_B, false, is_infer, 1.0f, tensor_name, target);
#else
          out = pe::MatmulV2(
              new_A, new_B, false, is_infer, 1.0f, tensor_name, target);
#endif
        } else {
          out = pe::Matmul(new_A, new_B, false, is_infer, 1.0f, tensor_name);
        }

        std::vector<CINNValue> res;
        for (auto &t : out) {
          stages->InsertLazily(t);
        }

        for (auto &t : out) {
          res.push_back(CINNValue(t));
        }
        res.push_back(CINNValue(stages));
        *ret = CINNValuePack{res};
      });

  framework::CINNSchedule mul_schedule([=](lang::Args args,
                                           lang::RetValue *ret) {
    CHECK(!args.empty())
        << "The input argument of matmul schedule is empty! Please check.\n";
    CINNValuePack arg_pack = args[0];
    std::vector<CINNValue> results =
        pe::IRCudaScheduleMatMul(arg_pack, output_shape, target);
    *ret = CINNValuePack({results});
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(mul_compute, mul_schedule, "strategy.mul.x86", 1);

  return strategy;
}

std::vector<std::vector<int>> InferShapeForMul(
    const std::vector<std::vector<int>> &inputs_shape,
    const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 2U)
      << "The input's shape size should be 2! Please check again.";
  CHECK_GE(inputs_shape[0].size(), 2U)
      << "Input matrix X's dim should be >= 2! Please check.";
  CHECK_GE(inputs_shape[1].size(), 2U)
      << "Input matrix Y's dim should be >= 2! Please check.";

  VLOG(4) << "During the matmul shape inference, origin shape_A: "
          << utils::Join(inputs_shape[0], ", ");
  VLOG(4) << "During the matmul shape inference, origin shape_B: "
          << utils::Join(inputs_shape[1], ", ");

  int x_num_col_dims = SafeGetAttr(attrs, "x_num_col_dims", 1);
  int y_num_col_dims = SafeGetAttr(attrs, "y_num_col_dims", 1);
  bool is_infer = SafeGetAttr(attrs, "is_infer", false);

  const auto &new_shape = pe::utils::GetMulNewShapes(
      inputs_shape, x_num_col_dims, y_num_col_dims, is_infer);

  const auto &new_shape_A = new_shape[0];
  const auto &new_shape_B = new_shape[1];
  const auto &output_shape = new_shape[2];

  VLOG(4) << "During the mul shape inference, new_shape_A: "
          << utils::Join(new_shape_A, ", ");
  VLOG(4) << "During the mul shape inference, new_shape_B: "
          << utils::Join(new_shape_B, ", ");
  VLOG(4) << "During the mul shape inference, output_shape: "
          << utils::Join(output_shape, ", ");

  int a_K = new_shape_A[1];
  int b_K = is_infer ? new_shape_B[1] : new_shape_B[0];

  CHECK_EQ(a_K, b_K) << "The K dimension of mul should be equal.";

  return {output_shape};
}

std::vector<Type> InferDtypeForMul(const std::vector<Type> &inputs_type,
                                   const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_type.size(), 2U)
      << "The input's type size should be 2! Please check again.";
  CHECK_EQ(inputs_type[0], inputs_type[1])
      << "The input's types should be equal! Please check again.";

  return {inputs_type[0]};
}

std::vector<std::vector<std::string>> InferLayoutForMul(
    const std::vector<framework::shape_t> &input_shapes,
    const std::vector<std::string> &input_layouts,
    const framework::NodeAttr &attrs,
    const Target &target) {
  CHECK_EQ(input_layouts.size(), 2U)
      << "The input's layouts size is not 2! Please check again.";
  CHECK_EQ(input_shapes.size(), 2U) << "mul should have 2 input shapes";
  std::vector<std::string> new_input_layouts = input_layouts;
  for (int i = 0; i < input_shapes.size(); i++) {
    if (input_shapes[i].size() > 4) {
      // alter input layout back
      new_input_layouts[i] = "NCHW";
    }
  }

  return {{"", ""}, new_input_layouts};
}

std::shared_ptr<OpStrategy> StrategyForCublasGemm(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  framework::CINNCompute gemm_compute(
      [attrs](lang::Args args, lang::RetValue *ret) {
        auto &attr_store = attrs.attr_store;
        CHECK(attr_store.contains("trans_a"))
            << "The cublas_gemm should have an attr named `trans_a`.";
        CHECK(attr_store.contains("trans_b"))
            << "The cublas_gemm should have an attr named `trans_b`.";
        CHECK(!args.empty())
            << "The input `args` of cublas_gemm is empty! Please check.";

        CINNValuePack input_args = args[0];
        CHECK_GE(input_args.size(), 3U)
            << "The input number of cublas_gemm should be equal to 3.";
        Expr lhs = input_args[0];
        Expr rhs = input_args[1];
        Expr bias = input_args[2];
        CHECK(lhs.as_tensor());
        CHECK(rhs.as_tensor());
        CHECK(bias.as_tensor());
        auto bias_tensor = bias.as_tensor_ref();
        // dummy gemm computation, which will be replaced by
        // cinn_gpu_cublas_gemm in the GemmRewriter pass.

        CHECK_EQ(input_args.size(), 4);
        CHECK(input_args[3].is_string());
        std::string tensor_name = input_args[3].operator std::string();
        auto out = pe::Identity(bias_tensor, tensor_name).front();
        auto stages = CreateStages(
            {lhs.as_tensor_ref(), rhs.as_tensor_ref(), bias_tensor});
        stages->InsertLazily(out);
        std::vector<CINNValue> res{CINNValue(out), CINNValue(stages)};
        *ret = CINNValuePack{res};
      });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(gemm_compute,
                    GetInjectiveScheduleFunc(output_shapes, target),
                    "strategy.cublas.gemm",
                    1);

  return strategy;
}

std::vector<shape_t> InferShapeForCublasGemm(
    const std::vector<std::vector<int>> &input_shapes,
    const framework::AttrMapType &attrs) {
  CHECK_EQ(input_shapes.size(), 3UL)
      << "cublas_gemm should have 3 input shapes";
  CHECK_EQ(input_shapes[0].size(), input_shapes[1].size());
  CHECK_EQ(input_shapes[0].size(), input_shapes[2].size());
  CHECK((input_shapes[0].size() == 2 || input_shapes[0].size() == 3));
  return {input_shapes[2]};
}

std::vector<Type> InferDtypeForCublasGemm(const std::vector<Type> &inputs_type,
                                          const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_type.size(), 3UL)
      << "The input's type size is 0! Please check again.";
  CHECK_EQ(inputs_type[0], inputs_type[1])
      << "The input A and B's types should be equal! Please check again.";
  CHECK_EQ(inputs_type[0], inputs_type[2])
      << "The input A and C's types should be equal! Please check again.";
  return {inputs_type[0]};
}

std::shared_ptr<OpStrategy> StrategyForLayoutTransform(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  framework::CINNCompute layout_transform_compute([=](lang::Args args,
                                                      lang::RetValue *ret) {
    std::string src_layout;
    std::string dst_layout;
    if (attrs.attr_store.find("src_layout") != attrs.attr_store.end()) {
      src_layout = absl::get<std::string>(attrs.attr_store.at("src_layout"));
    }
    if (attrs.attr_store.find("dst_layout") != attrs.attr_store.end()) {
      dst_layout = absl::get<std::string>(attrs.attr_store.at("dst_layout"));
    }
    CHECK(!args.empty()) << "The input argument of layout_transform compute is "
                            "empty! Please check.\n";
    CINNValuePack input_args = args[0];
    CHECK(!input_args.empty())
        << "at least one input tensor for layout_transform compute\n";
    Expr A = input_args[0];
    CHECK(A.as_tensor());

    CHECK_EQ(input_args.size(), 2);
    CHECK(input_args[1].is_string());
    std::string tensor_name = input_args[1].operator std::string();

    auto out = pe::LayoutTransform(
        A.as_tensor_ref(), src_layout, dst_layout, tensor_name);
    auto stages = CreateStages({A.as_tensor_ref()});
    std::vector<CINNValue> res;
    stages->InsertLazily(out);
    res = {CINNValue(out), CINNValue(stages)};
    *ret = CINNValuePack{res};
  });

  framework::CINNSchedule layout_transform_schedule([=](lang::Args args,
                                                        lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of CublasGemm schedule "
                            "is empty! Please check.";
    CINNValuePack arg_pack = args[0];
    std::vector<Expr> vec_ast;
    for (int i = 0; i < arg_pack.size(); i++) {
      if (arg_pack[i].is_expr()) {
        Expr temp = arg_pack[i];
        vec_ast.emplace_back(temp);
      }
    }
    CHECK(!vec_ast.empty());
    ir::ModuleExpr mod_expr(vec_ast);
    ir::IRSchedule ir_sch(mod_expr);
    ir_sch.MergeExprs();

    if (target.arch == Target::Arch::X86) {
      pe::IRScheduleInjectiveCPU(ir_sch, output_shapes.front(), target);
    } else {
      CINN_NOT_IMPLEMENTED
    }
    std::vector<CINNValue> res{CINNValue(ir_sch.GetModule().GetExprs().at(0))};
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  CHECK(out_type.size())
      << "Out_type of layout_transform op is empty! Please check.";
  strategy->AddImpl(layout_transform_compute,
                    layout_transform_schedule,
                    "strategy.layout_transform.x86",
                    1);
  return strategy;
}

std::vector<shape_t> InferShapeForLayoutTransform(
    const std::vector<shape_t> &inputs_shape,
    const framework::AttrMapType &attrs) {
  std::string src_layout;
  std::string dst_layout;
  if (attrs.find("src_layout") != attrs.end()) {
    src_layout = absl::get<std::string>(attrs.at("src_layout"));
  }
  if (attrs.find("dst_layout") != attrs.end()) {
    dst_layout = absl::get<std::string>(attrs.at("dst_layout"));
  }
  CHECK_EQ(inputs_shape.size(), 1UL);

  std::vector<Expr> input_shapes_expr;
  for (int shape : inputs_shape[0]) {
    input_shapes_expr.push_back(Expr(shape));
  }
  absl::flat_hash_map<int, std::vector<int>> split_index_map;
  std::vector<Expr> out_shapes =
      pe::InferShapeLayoutTransform(input_shapes_expr,
                                    ir::Layout(src_layout),
                                    ir::Layout(dst_layout),
                                    &split_index_map);
  VLOG(4) << "out_shapes: " << out_shapes;
  std::vector<int> output_shapes;
  for (auto &shape : out_shapes) {
    output_shapes.push_back(shape.as_int32());
  }
  return {output_shapes};
}

std::vector<Type> InferDtypeForLayoutTransform(
    const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty())
      << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

std::shared_ptr<OpStrategy> StrategyForReverse(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  // check output shape
  CHECK(!output_shapes.empty() && !output_shapes[0].empty())
      << "Output shape is empty! Please check.\n";
  // get axis[0, n_dim)
  std::vector<int> axis;
  if (attrs.attr_store.find("axis") != attrs.attr_store.end()) {
    axis = absl::get<std::vector<int>>(attrs.attr_store.at("axis"));
    for (auto &e : axis) {
      if (e >= static_cast<int>(output_shapes[0].size()) ||
          e < -1 * static_cast<int>(output_shapes[0].size())) {
        LOG(FATAL) << "axis is not in [0, n_dim), Please check.";
      }
      if (e < 0) {
        e += output_shapes[0].size();
      }
    }
  }

  framework::CINNCompute reverse_compute([=](lang::Args args,
                                             lang::RetValue *ret) {
    CHECK(!args.empty())
        << "The input argument of reverse compute is empty! Please check.\n";
    CINNValuePack input_args = args[0];
    CHECK(!input_args.empty())
        << "at least one input tensor for reverse compute\n";
    Expr A = input_args[0];
    CHECK(A.as_tensor());

    CHECK_EQ(input_args.size(), 2);
    CHECK(input_args[1].is_string());
    std::string tensor_name = input_args[1].operator std::string();

    auto out = pe::Reverse(A.as_tensor_ref(), axis, tensor_name);
    auto stages = CreateStages({A.as_tensor_ref(), out});
    *ret = CINNValuePack{{CINNValue(out), CINNValue(stages)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  CHECK(out_type.size()) << "Out_type of reverse op is empty! Please check.";
  strategy->AddImpl(reverse_compute,
                    GetInjectiveScheduleFunc(output_shapes, target),
                    "strategy.reverse.x86",
                    1);
  return strategy;
}

std::vector<framework::shape_t> InferShapeForReverse(
    const std::vector<framework::shape_t> &inputs_shape,
    const framework::AttrMapType &attrs) {
  CHECK(!inputs_shape.empty())
      << "The input's shape is empty! Please check again.";
  std::vector<framework::shape_t> res{inputs_shape[0]};
  if (attrs.find("axis") != attrs.end()) {
    auto axis = absl::get<std::vector<int>>(attrs.at("axis"));
    for (auto &e : axis) {
      if (e >= static_cast<int>(inputs_shape[0].size()) ||
          e < -1 * static_cast<int>(inputs_shape[0].size())) {
        LOG(FATAL) << "axis is not in [-n_dim, n_dim), Please check.";
      }
      if (e < 0) {
        e += inputs_shape[0].size();
      }
    }
  }
  return res;
}

std::vector<std::vector<std::string>> InferLayoutForReverse(
    const std::vector<framework::shape_t> &input_shapes,
    const std::vector<std::string> &input_layouts,
    const framework::NodeAttr &attrs,
    const Target &target) {
  if (attrs.attr_store.find("axis") != attrs.attr_store.end()) {
    auto axis = absl::get<std::vector<int>>(attrs.attr_store.at("axis"));
    for (auto &e : axis) {
      if (e >= static_cast<int>(input_shapes[0].size()) ||
          e < -1 * static_cast<int>(input_shapes[0].size())) {
        LOG(FATAL) << "axis is not in [-n_dim, n_dim), Please check.";
      }
    }
  }
  CHECK_EQ(input_layouts.size(), 1U)
      << "The input's layout size is not 1! Please check again.";
  return {input_layouts, input_layouts};
}

std::vector<std::vector<std::string>> InferLayoutForLayoutTransform(
    const std::vector<framework::shape_t> &input_shapes,
    const std::vector<std::string> &input_layouts,
    const framework::NodeAttr &attrs,
    const Target &target) {
  CHECK_EQ(input_layouts.size(), 1U)
      << "The input's layouts size is not 1! Please check again.";
  std::string dst_layout;
  std::string src_layout;
  if (attrs.attr_store.find("dst_layout") != attrs.attr_store.end()) {
    dst_layout = absl::get<std::string>(attrs.attr_store.at("dst_layout"));
  }
  if (attrs.attr_store.find("src_layout") != attrs.attr_store.end()) {
    src_layout = absl::get<std::string>(attrs.attr_store.at("src_layout"));
  }
  return {{dst_layout}, {src_layout}};
}

std::shared_ptr<OpStrategy> StrategyForTranspose(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  // check output shape
  CHECK(!output_shapes.empty() && !output_shapes[0].empty())
      << "Output shape is empty! Please check.\n";

  std::vector<int> axis;
  auto input_shape = inputs[0]->shape;
  if (attrs.attr_store.find("axis") != attrs.attr_store.end()) {
    axis = absl::get<std::vector<int>>(attrs.attr_store.at("axis"));
    CHECK_EQ(axis.size(), output_shapes[0].size())
        << "axis size is not equal output_shapes size! Please check setting.\n";
    // check axis and shape
    for (int idx = 0; idx < axis.size(); ++idx) {
      CHECK(axis[idx] >= 0 && axis[idx] < axis.size());
      for (int idy = idx + 1; idy < axis.size(); ++idy) {
        CHECK_NE(axis[idx], axis[idy]) << "axis can't repeat!";
      }
      CHECK_EQ(output_shapes[0][idx], input_shape[axis[idx]].as_int32())
          << "output shape is not equal! Please check!\n";
    }
  } else {
    LOG(FATAL) << "axis is not be set! Please check.";
  }

  framework::CINNCompute transpose_compute([=](lang::Args args,
                                               lang::RetValue *ret) {
    CHECK(!args.empty())
        << "The input argument of transpose compute is empty! Please check.\n";
    CINNValuePack input_args = args[0];
    CHECK(!input_args.empty())
        << "at least one input tensor for transpose compute\n";
    Expr A = input_args[0];
    CHECK(A.as_tensor());
    CHECK_EQ(input_args.size(), 2);
    CHECK(input_args[1].is_string());
    std::string tensor_name = input_args[1].operator std::string();

    auto out = pe::Transpose(A.as_tensor_ref(), axis, tensor_name);
    auto stages = CreateStages({out});
    *ret = CINNValuePack{{CINNValue(out), CINNValue(stages)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(transpose_compute,
                    GetInjectiveScheduleFunc(output_shapes, target),
                    "strategy.transpose.x86",
                    1);
  return strategy;
}

std::vector<framework::shape_t> InferShapeForTranspose(
    const std::vector<framework::shape_t> &inputs_shape,
    const framework::AttrMapType &attrs) {
  std::vector<framework::shape_t> result;
  CHECK(!inputs_shape.empty())
      << "The input's shape is empty! Please check again.";
  if (attrs.find("axis") != attrs.end()) {
    auto axis = absl::get<std::vector<int>>(attrs.at("axis"));
    CHECK_EQ(axis.size(), inputs_shape[0].size())
        << "input size and axis size is not equal!";
    std::vector<int> output_shape;
    for (int idx = 0; idx < axis.size(); ++idx) {
      CHECK(axis[idx] >= 0 && axis[idx] < axis.size());
      for (int idy = idx + 1; idy < axis.size(); ++idy) {
        CHECK_NE(axis[idx], axis[idy]) << "axis can't repeat!";
      }
      output_shape.push_back(inputs_shape[0][axis[idx]]);
    }
    result.push_back(output_shape);
  } else {
    LOG(FATAL) << "axis is not be set! Please check.";
  }
  return result;
}

std::vector<std::vector<std::string>> InferLayoutForTranspose(
    const std::vector<framework::shape_t> &input_shapes,
    const std::vector<std::string> &input_layouts,
    const framework::NodeAttr &attrs,
    const Target &target) {
  CHECK_EQ(input_shapes.size(), 1U)
      << "The input's shape size is not 1! Please check again.";
  CHECK_EQ(input_layouts.size(), 1U)
      << "The input's layout size is not 1! Please check again.";

  std::vector<int> axis;
  if (attrs.attr_store.find("axis") != attrs.attr_store.end()) {
    axis = absl::get<std::vector<int>>(attrs.attr_store.at("axis"));
    for (int idx = 0; idx < axis.size(); ++idx) {
      CHECK(axis[idx] >= 0 && axis[idx] < axis.size());
      for (int idy = idx + 1; idy < axis.size(); ++idy) {
        CHECK_NE(axis[idx], axis[idy]) << "axis can't repeat!";
      }
    }
  } else {
    LOG(FATAL) << "axis is not be set! Please check.";
  }

  std::vector<std::string> new_input_layouts = input_layouts;
  for (int i = 0; i < input_shapes.size(); i++) {
    if (input_shapes[i].size() > 4) {
      // alter input layout back
      new_input_layouts[i] = input_layouts[0].substr(0, 4);
    }
  }

  std::string output_layout = new_input_layouts[0];
  for (int idx = 0; idx < axis.size(); ++idx) {
    output_layout[idx] = new_input_layouts[0][axis[idx]];
  }

  return {{output_layout}, new_input_layouts};
}

std::shared_ptr<OpStrategy> StrategyForGather(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  CHECK(!output_shapes.empty() && !output_shapes[0].empty())
      << "The shape of output is empty! Please check again.";
  VLOG(4) << "The output passed in StrategyForGather: "
          << utils::Join(output_shapes[0], ", ");
  CHECK(!out_type.empty())
      << "The output type of Gather is empty! Please check again.\n";

  int axis = 0;
  if (attrs.attr_store.contains("axis")) {
    axis = absl::get<int>(attrs.attr_store.at("axis"));
  }
  axis = axis < 0 ? axis + static_cast<int>(inputs[0]->shape.size()) : axis;

  std::vector<Expr> output_shape;
  output_shape.reserve(output_shapes[0].size());
  for (int i : output_shapes[0]) {
    output_shape.emplace_back(i);
  }

  framework::CINNCompute gather_compute{
      [axis, output_shape = std::move(output_shape)](lang::Args args,
                                                     lang::RetValue *ret) {
        VLOG(4) << "The axis value used in gather_compute: " << axis;
        CHECK(!args.empty()) << "The input args are empty! Please check again.";
        CINNValuePack input_args = args[0];
        int input_size = input_args.size();
        CHECK_GE(input_size, 2U)
            << "Require 2 input tensors for Gather compute.";
        Expr x = input_args[0];
        CHECK(x.as_tensor());
        Expr index = input_args[1];
        CHECK(index.as_tensor());

        CHECK_EQ(input_args.size(), 3U);
        CHECK(input_args[2].is_string());
        std::string tensor_name = input_args[2].operator std::string();

        auto out = pe::Gather(x.as_tensor_ref(),
                              index.as_tensor_ref(),
                              output_shape,
                              axis,
                              tensor_name);
        auto stages = CreateStages({x.as_tensor_ref(), index.as_tensor_ref()});
        stages->InsertLazily(out);
        std::vector<CINNValue> res{CINNValue(out), CINNValue(stages)};
        *ret = CINNValuePack{res};
      }};

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(gather_compute,
                    GetInjectiveScheduleFunc(output_shapes, target),
                    "strategy.gather.x86",
                    1);
  return strategy;
}

std::vector<std::vector<int>> InferShapeForGather(
    const std::vector<std::vector<int>> &inputs_shape,
    const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 2U)
      << "The inputs' shape size should be equal to 2! Please check again.";
  std::vector<int> x_shape = inputs_shape[0];
  std::vector<int> index_shape = inputs_shape[1];
  int axis = absl::get<int>(attrs.at("axis"));
  VLOG(4) << "The axis value used in Gather: " << axis;

  CHECK(axis >= 0 && axis < static_cast<int>(x_shape.size()))
      << "The attribute `axis` in Gather should be >= 0 and < the size of the "
         "first input shape! Please check again.";

  std::vector<int> output_shape = x_shape;
  output_shape[axis] = index_shape[axis];
  VLOG(4) << "The output shape of gather: " << utils::Join(output_shape, ", ");

  return {std::move(output_shape)};
}

std::vector<Type> InferDtypeForGather(const std::vector<Type> &inputs_type,
                                      const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty())
      << "The input's type size is 0! Please check again.";
  return {inputs_type[0]};
}

std::vector<std::vector<std::string>> InferLayoutForGather(
    const std::vector<framework::shape_t> &input_shapes,
    const std::vector<std::string> &input_layouts,
    const framework::NodeAttr &attrs,
    const Target &target) {
  CHECK_EQ(input_layouts.size(), 2U)
      << "The input's layout size is not equal to 2! Please check again.";
  return {{input_layouts[0]}, input_layouts};
}

std::shared_ptr<OpStrategy> StrategyForScatterAssign(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  int axis = 0;
  if (attrs.attr_store.find("axis") != attrs.attr_store.end()) {
    axis = absl::get<int>(attrs.attr_store.at("axis"));
  }

  framework::CINNCompute scatter_assign_compute([=](lang::Args args,
                                                    lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input arguments of ScatterAssign compute is "
                            "empty! Please check.\n";
    CINNValuePack arg_pack = args[0];
    int input_size = arg_pack.size();
    CHECK_GE(input_size, 3U)
        << "at least 3 input tensors for ScatterAssign compute\n";
    CHECK(!output_shapes.empty());

    Expr expr_input = arg_pack[0];
    CHECK(expr_input.as_tensor());
    auto tensor_input = expr_input.as_tensor_ref();

    Expr expr_updates = arg_pack[1];
    CHECK(expr_updates.as_tensor());
    auto tensor_updates = expr_updates.as_tensor_ref();

    Expr expr_index = arg_pack[2];
    CHECK(expr_index.as_tensor());
    auto tensor_index = expr_index.as_tensor_ref();

    auto stages = CreateStages({tensor_input, tensor_updates, tensor_index});

    CHECK_EQ(arg_pack.size(), 4U);
    CHECK(arg_pack[3].is_string());
    std::string tensor_name = arg_pack[3].operator std::string();

    auto out = pe::ScatterAssign(
        tensor_input, tensor_updates, tensor_index, target, axis, tensor_name);

    std::vector<CINNValue> res;
    stages->InsertLazily(out);
    res.push_back(CINNValue(out));
    CHECK(!out_type.empty())
        << "Output type of ScatterAssign is empty! Please check.\n";
    res.push_back(CINNValue(stages));
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(scatter_assign_compute,
                    GetInjectiveScheduleFunc(output_shapes, target, false),
                    "strategy.scatter_assign.x86",
                    1);
  return strategy;
}

std::vector<std::vector<int>> InferShapeForScatterAssign(
    const std::vector<std::vector<int>> &inputs_shape,
    const framework::AttrMapType &attrs) {
  CHECK_GE(inputs_shape.size(), 3U)
      << "The input's shape size should be no less than 3! Please check again.";

  const auto &input_shape = inputs_shape[0];
  const auto &assign_shape = inputs_shape[1];
  const auto &index_shape = inputs_shape[2];

  int axis = 0;
  if (attrs.find("axis") != attrs.end()) {
    axis = absl::get<int>(attrs.at("axis"));
  }

  if (axis < 0) axis += input_shape.size();

  CHECK(axis >= 0 && axis < input_shape.size())
      << "In ScatterAssign op, the attribute `axis` should be >= 0 and < input "
         "shape's size! Please check.";
  CHECK_EQ(index_shape.size(), 1U)
      << "Dimensions of index tensor in ScatterAssign should be 1! Please "
         "check.";
  CHECK_EQ(input_shape.size(), assign_shape.size())
      << "Dimensions of inputs A and B in ScatterAssign should be equal! "
         "Please check.";
  CHECK_EQ(assign_shape[axis], index_shape[0])
      << "The first dimension of input B and index tensor in ScatterAssign "
         "should be equal! Please check.";
  for (int i = 0; i < input_shape.size(); ++i) {
    if (i != axis) {
      CHECK_EQ(input_shape[i], assign_shape[i])
          << "The " << i
          << "-th dimension of input A and B in ScatterAssign should be equal! "
             "Please check.";
    }
  }

  VLOG(4) << "Each input tensor's shape of ScatterAssign: A("
          << cinn::utils::Join(input_shape, ",") << "), B("
          << cinn::utils::Join(assign_shape, ",") << "), index("
          << cinn::utils::Join(index_shape, ",") << ")"
          << " at axis (" << axis << ")";

  return {input_shape};
}

std::vector<Type> InferDtypeForScatterAssign(
    const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty())
      << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

std::vector<std::vector<std::string>> InferLayoutForScatterAssign(
    const std::vector<framework::shape_t> &input_shapes,
    const std::vector<std::string> &input_layouts,
    const framework::NodeAttr &attrs,
    const Target &target) {
  CHECK_GE(input_layouts.size(), 3U)
      << "The input's layout size is less than 3! Please check again.";
  return {{input_layouts[0]}, input_layouts};
}

std::shared_ptr<OpStrategy> StrategyForScatterAdd(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  int axis = 0;
  if (attrs.attr_store.find("axis") != attrs.attr_store.end()) {
    axis = absl::get<int>(attrs.attr_store.at("axis"));
  }

  framework::CINNCompute scatter_add_compute([=](lang::Args args,
                                                 lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input arguments of ScatterAdd compute is "
                            "empty! Please check.\n";
    CINNValuePack arg_pack = args[0];
    int input_size = arg_pack.size();
    CHECK_GE(input_size, 3U)
        << "at least 3 input tensors for ScatterAdd compute\n";
    CHECK(!output_shapes.empty());

    Expr expr_input = arg_pack[0];
    CHECK(expr_input.as_tensor());
    auto tensor_input = expr_input.as_tensor_ref();

    Expr expr_updates = arg_pack[1];
    CHECK(expr_updates.as_tensor());
    auto tensor_updates = expr_updates.as_tensor_ref();

    Expr expr_index = arg_pack[2];
    CHECK(expr_index.as_tensor());
    auto tensor_index = expr_index.as_tensor_ref();

    auto stages = CreateStages({tensor_input, tensor_updates, tensor_index});

    CHECK_EQ(arg_pack.size(), 4U);
    CHECK(arg_pack[3].is_string());
    std::string tensor_name = arg_pack[3].operator std::string();

    auto out = pe::ScatterAdd(
        tensor_input, tensor_updates, tensor_index, target, axis, tensor_name);

    std::vector<CINNValue> res;
    stages->InsertLazily(out);
    res.push_back(CINNValue(out));
    CHECK(!out_type.empty())
        << "Output type of ScatterAdd is empty! Please check.\n";
    res.push_back(CINNValue(stages));
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(scatter_add_compute,
                    GetInjectiveScheduleFunc(output_shapes, target, false),
                    "strategy.scatter_add.x86",
                    1);
  return strategy;
}

std::vector<std::vector<int>> InferShapeForScatterAdd(
    const std::vector<std::vector<int>> &inputs_shape,
    const framework::AttrMapType &attrs) {
  CHECK_GE(inputs_shape.size(), 3U)
      << "The input's shape size should be no less than 3! Please check again.";

  const auto &input_shape = inputs_shape[0];
  const auto &updates_shape = inputs_shape[1];
  const auto &index_shape = inputs_shape[2];

  int axis = 0;
  if (attrs.find("axis") != attrs.end()) {
    axis = absl::get<int>(attrs.at("axis"));
  }

  if (axis < 0) axis += input_shape.size();

  CHECK(axis >= 0 && axis < input_shape.size())
      << "In ScatterAdd op, the attribute `axis` should be >= 0 and < input "
         "shape's size! Please check.";
  CHECK_EQ(index_shape.size(), 1U)
      << "Dimensions of index tensor in ScatterAdd should be 1! Please check.";
  CHECK_EQ(input_shape.size(), updates_shape.size())
      << "Dimensions of inputs A and B in ScatterAdd should be equal! Please "
         "check.";
  CHECK_EQ(updates_shape[axis], index_shape[0])
      << "The first dimension of input B and index tensor in ScatterAdd should "
         "be equal! Please check.";
  for (int i = 0; i < input_shape.size(); ++i) {
    if (i != axis) {
      CHECK_EQ(input_shape[i], updates_shape[i])
          << "The " << i
          << "-th dimension of input A and B in ScatterAdd should be equal! "
             "Please check.";
    }
  }

  VLOG(4) << "Each input tensor's shape of ScatterAdd: A("
          << cinn::utils::Join(input_shape, ",") << "), B("
          << cinn::utils::Join(updates_shape, ",") << "), index("
          << cinn::utils::Join(index_shape, ",") << ")"
          << " at axis (" << axis << ")";

  return {input_shape};
}

std::vector<Type> InferDtypeForScatterAdd(const std::vector<Type> &inputs_type,
                                          const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty())
      << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

std::vector<std::vector<std::string>> InferLayoutForScatterAdd(
    const std::vector<framework::shape_t> &input_shapes,
    const std::vector<std::string> &input_layouts,
    const framework::NodeAttr &attrs,
    const Target &target) {
  CHECK_GE(input_layouts.size(), 3U)
      << "The input's layout size is less than 3! Please check again.";
  return {{input_layouts[0]}, input_layouts};
}

std::shared_ptr<OpStrategy> StrategyForSlice(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  std::vector<int> starts, ends, axes, strides, decrease_axis;
  if (attrs.attr_store.find("starts") != attrs.attr_store.end()) {
    starts = absl::get<std::vector<int>>(attrs.attr_store.at("starts"));
  }
  if (attrs.attr_store.find("ends") != attrs.attr_store.end()) {
    ends = absl::get<std::vector<int>>(attrs.attr_store.at("ends"));
  }
  if (attrs.attr_store.find("axes") != attrs.attr_store.end()) {
    axes = absl::get<std::vector<int>>(attrs.attr_store.at("axes"));
  }
  if (attrs.attr_store.find("strides") != attrs.attr_store.end()) {
    strides = absl::get<std::vector<int>>(attrs.attr_store.at("strides"));
  }
  if (attrs.attr_store.find("decrease_axis") != attrs.attr_store.end()) {
    decrease_axis =
        absl::get<std::vector<int>>(attrs.attr_store.at("decrease_axis"));
  }

  CHECK(!starts.empty()) << "The Slice op doesn't find [starts] attrbute! It "
                            "it a mandatory attribute, please check.";
  CHECK(!ends.empty()) << "The Slice op doesn't find [ends] attrbute! It it a "
                          "mandatory attribute, please check.";
  CHECK_EQ(starts.size(), ends.size())
      << "The size of [starts] and [ends] must be identical! Please check.";
  if (!axes.empty()) {
    CHECK_EQ(starts.size(), axes.size())
        << "The size of [starts] and [axes] must be identical! Please check.";
  } else {
    for (int i = 0; i < starts.size(); i++) {
      axes.push_back(i);
    }
  }
  if (!strides.empty()) {
    CHECK_EQ(starts.size(), strides.size())
        << "The size of [starts] and [strides] must be identical! Please "
           "check.";
  } else {
    for (int i = 0; i < starts.size(); i++) {
      strides.push_back(1);
    }
  }

  std::vector<Expr> output_shape;
  for (auto &i : output_shapes[0]) {
    output_shape.push_back(Expr(i));
  }

  framework::CINNCompute slice_compute(
      [=](lang::Args args, lang::RetValue *ret) {
        CHECK(!args.empty())
            << "The input arguments of slice compute is empty! Please check.";
        CINNValuePack arg_pack = args[0];
        CHECK(!arg_pack.empty())
            << "The input tensors of slice compute is empty! Please check.";
        Expr A_expr = arg_pack[0];
        CHECK(A_expr.as_tensor());
        ir::Tensor A = A_expr.as_tensor_ref();

        CHECK_EQ(arg_pack.size(), 2U);
        CHECK(arg_pack[1].is_string());
        std::string tensor_name = arg_pack[1].operator std::string();

        auto out = pe::Slice(
            A, starts, axes, strides, decrease_axis, output_shape, tensor_name);
        auto stages = CreateStages({out});
        *ret = CINNValuePack{{CINNValue(out), CINNValue(stages)}};
      });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(slice_compute,
                    GetInjectiveScheduleFunc(output_shapes, target),
                    "strategy.slice.x86",
                    1);

  return strategy;
}

std::vector<std::vector<int>> InferShapeForSlice(
    const std::vector<std::vector<int>> &inputs_shape,
    const framework::AttrMapType &attrs) {
  CHECK(!inputs_shape.empty() && !inputs_shape[0].empty())
      << "The input's shape size is 0! Please check again.";
  std::vector<int> starts, ends, axes, strides, decrease_axis, infer_flags;
  for (auto &iter : attrs) {
    if (iter.first == "starts") {
      starts = absl::get<std::vector<int>>(iter.second);
    } else if (iter.first == "ends") {
      ends = absl::get<std::vector<int>>(iter.second);
    } else if (iter.first == "axes") {
      axes = absl::get<std::vector<int>>(iter.second);
    } else if (iter.first == "strides") {
      strides = absl::get<std::vector<int>>(iter.second);
    } else if (iter.first == "decrease_axis") {
      decrease_axis = absl::get<std::vector<int>>(iter.second);
    } else if (iter.first == "infer_flags") {
      infer_flags = absl::get<std::vector<int>>(iter.second);
    } else {
      LOG(ERROR) << "Unsupported attr: " << iter.first << std::endl;
    }
  }
  CHECK(!starts.empty()) << "The Slice op doesn't find [starts] attrbute! It "
                            "it a mandatory attribute, please check.";
  CHECK(!ends.empty()) << "The Slice op doesn't find [ends] attrbute! It it a "
                          "mandatory attribute, please check.";
  CHECK_EQ(starts.size(), ends.size())
      << "The size of [starts] and [ends] must be identical! Please check.";
  if (!axes.empty()) {
    CHECK_EQ(starts.size(), axes.size())
        << "The size of [starts] and [axes] must be identical! Please check.";
  } else {
    for (int i = 0; i < starts.size(); i++) {
      axes.push_back(i);
    }
  }
  if (!strides.empty()) {
    CHECK_EQ(starts.size(), strides.size())
        << "The size of [starts] and [strides] must be identical! Please "
           "check.";
  } else {
    for (int i = 0; i < starts.size(); i++) {
      strides.push_back(1);
    }
  }

  std::vector<int> output_shape = inputs_shape[0];
  for (int i = 0; i < axes.size(); i++) {
    if (ends[i] < 0) {
      ends[i] = output_shape[axes[i]] + ends[i];
    } else if (ends[i] > output_shape[axes[i]]) {
      ends[i] = output_shape[axes[i]];
    }
    if (starts[i] < -output_shape[axes[i]]) {
      starts[i] = 0;
    } else if (starts[i] < 0) {
      starts[i] = output_shape[axes[i]] + starts[i];
    } else if (starts[i] > output_shape[axes[i]]) {
      starts[i] = output_shape[axes[i]] - 1;
    }

    CHECK_NE(strides[i], 0)
        << "The value of [strides] of slice should not be 0 ! Please Check.";
    if (strides[i] > 0) {
      CHECK(ends[i] > starts[i])
          << "[ends] should greater than [starts] when strides > 0 ! But here "
          << ends[i] << " < " << starts[i] << ", Please Check.";
      output_shape[axes[i]] =
          (ends[i] - starts[i] + strides[i] - 1) / strides[i];
    } else {
      CHECK(ends[i] < starts[i])
          << "[ends] should less than [starts] when strides < 0 ! But here "
          << ends[i] << " > " << starts[i] << ",  Please Check.";
      output_shape[axes[i]] =
          (starts[i] - ends[i] + (-strides[i]) - 1) / (-strides[i]);
    }
  }

  if (decrease_axis.size() > 0) {
    std::vector<int> new_shape;
    for (int i = 0; i < output_shape.size(); ++i) {
      if (std::find(decrease_axis.cbegin(), decrease_axis.cend(), i) !=
          decrease_axis.cend()) {
        CHECK_EQ(output_shape[i], 1)
            << "Decrease dim should be 1, but now received " << output_shape[i];
      } else {
        new_shape.emplace_back(output_shape[i]);
      }
    }
    if (new_shape.empty()) {
      // output shape should not empty
      new_shape = {1};
    }

    output_shape = new_shape;
  }

  VLOG(4) << "Output shape of Slice is: "
          << cinn::utils::Join(output_shape, ",");
  std::vector<std::vector<int>> res{output_shape};
  return res;
}

std::vector<Type> InferDtypeForSlice(const std::vector<Type> &inputs_type,
                                     const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty())
      << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

std::vector<std::vector<std::string>> InferLayoutForSlice(
    const std::vector<framework::shape_t> &input_shapes,
    const std::vector<std::string> &input_layouts,
    const framework::NodeAttr &attrs,
    const Target &target) {
  CHECK_EQ(input_layouts.size(), 1U)
      << "The input's layout size is not 1! Please check again.";
  CHECK_EQ(input_shapes.size(), 1U)
      << "The input's shape size is not 1! Please check again.";
  std::vector<int> starts;
  std::vector<int> ends;
  std::vector<int> axes;
  for (auto &iter : attrs.attr_store) {
    if (iter.first == "starts") {
      starts = absl::get<std::vector<int>>(iter.second);
    } else if (iter.first == "ends") {
      ends = absl::get<std::vector<int>>(iter.second);
    } else if (iter.first == "axes") {
      axes = absl::get<std::vector<int>>(iter.second);
    }
  }
  std::string new_input_layouts = input_layouts[0];
  bool trans_back = false;
  if (input_shapes[0].size() > 4) {
    for (int i = 0; i < axes.size(); i++) {
      if (axes[i] == 1) {
        trans_back = true;
        break;
      }
    }
  }
  if (trans_back) {
    return {{"NCHW"}, {"NCHW"}};
  }
  return {input_layouts, input_layouts};
}

std::shared_ptr<OpStrategy> StrategyForSliceAssign(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  CHECK_EQ(inputs.size(), 2)
      << "the number of input tensors must be equal to 2";
  CHECK(!output_shapes.empty() && !output_shapes[0].empty())
      << "The shape of output is empty! Please check again.";
  VLOG(4) << "The output passed in StrategyForSliceAssign: "
          << utils::Join(output_shapes[0], ", ");
  CHECK(!out_type.empty())
      << "The output type of SliceAssign is empty! Please check again.\n";

  std::vector<int> starts, ends, axes, strides;
  if (attrs.attr_store.find("starts") != attrs.attr_store.end()) {
    starts = absl::get<std::vector<int>>(attrs.attr_store.at("starts"));
  }
  if (attrs.attr_store.find("ends") != attrs.attr_store.end()) {
    ends = absl::get<std::vector<int>>(attrs.attr_store.at("ends"));
  }
  if (attrs.attr_store.find("axes") != attrs.attr_store.end()) {
    axes = absl::get<std::vector<int>>(attrs.attr_store.at("axes"));
  }
  if (attrs.attr_store.find("strides") != attrs.attr_store.end()) {
    strides = absl::get<std::vector<int>>(attrs.attr_store.at("strides"));
  }

  CHECK(!starts.empty())
      << "The SliceAssign op doesn't find [starts] attrbute! It it a mandatory "
         "attribute, please check.";
  CHECK(!ends.empty()) << "The SliceAssign op doesn't find [ends] attrbute! It "
                          "it a mandatory attribute, please check.";
  CHECK_EQ(starts.size(), ends.size())
      << "The size of [starts] and [ends] must be identical! Please check.";
  if (!axes.empty()) {
    CHECK_EQ(starts.size(), axes.size())
        << "The size of [starts] and [axes] must be identical! Please check.";
  } else {
    for (int i = 0; i < starts.size(); i++) {
      axes.push_back(i);
    }
  }
  if (!strides.empty()) {
    CHECK_EQ(starts.size(), strides.size())
        << "The size of [starts] and [strides] must be identical! Please "
           "check.";
  } else {
    for (int i = 0; i < starts.size(); i++) {
      strides.push_back(1);
    }
  }

  framework::CINNCompute slice_assign_compute{
      [=](lang::Args args, lang::RetValue *ret) {
        CHECK(!args.empty()) << "The input args are empty! Please check again.";
        CINNValuePack arg_pack = args[0];
        int input_size = arg_pack.size();
        CHECK_GE(input_size, 2U)
            << "Require 2 input tensors for SliceAssign compute.";
        Expr input = arg_pack[0];
        CHECK(input.as_tensor());
        Expr assign = arg_pack[1];
        CHECK(assign.as_tensor());

        CHECK_EQ(arg_pack.size(), 3U);
        CHECK(arg_pack[2].is_string());
        std::string tensor_name = arg_pack[2].operator std::string();

        auto out = pe::SliceAssign(input.as_tensor_ref(),
                                   assign.as_tensor_ref(),
                                   axes,
                                   starts,
                                   ends,
                                   strides,
                                   tensor_name);
        auto stages = CreateStages({out});
        std::vector<CINNValue> res{CINNValue(out), CINNValue(stages)};
        *ret = CINNValuePack{res};
      }};

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(slice_assign_compute,
                    GetInjectiveScheduleFunc(output_shapes, target),
                    "strategy.slice_assign.x86",
                    1);
  return strategy;
}

std::vector<std::vector<int>> InferShapeForSliceAssign(
    const std::vector<std::vector<int>> &inputs_shape,
    const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 2U)
      << "The inputs' shape size should be equal to 2! Please check again.";
  return {inputs_shape[0]};
}

std::vector<Type> InferDtypeForSliceAssign(
    const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty())
      << "The input's type size is 0! Please check again.";
  return {inputs_type[0]};
}

std::vector<std::vector<std::string>> InferLayoutForSliceAssign(
    const std::vector<framework::shape_t> &input_shapes,
    const std::vector<std::string> &input_layouts,
    const framework::NodeAttr &attrs,
    const Target &target) {
  CHECK_EQ(input_layouts.size(), 2U)
      << "The input's layout size is not equal to 2! Please check again.";
  return {{input_layouts[0]}, {""}};
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(transform_ops) {
  CINN_REGISTER_OP(matmul)
      .describe(
          "This operator is used to perform (batched) matrix multiplication "
          "over the last two dimensions of the input "
          "tensors X and Y.")
      .set_num_inputs(2)
#ifdef CINN_WITH_CUDA
      .set_num_outputs(1)
#else
      .set_num_outputs(2)
#endif
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForMatMul)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForMatMul))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForMatMul))
#ifndef CINN_WITH_CUDA
      .set_attr("inferlayout",
                MakeOpFunction(cinn::hlir::op::InferLayoutForMatMul))
#endif
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kNonFusible)
      .set_support_level(4);

  CINN_REGISTER_OP(split)
      .describe(
          "This operator is used to split tensors X to 'sections' sub-tensor "
          "on specified axis.")
      .set_num_inputs(1)
      .set_num_outputs(0)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForSplit)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForSplit))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForSplit))
#ifndef CINN_WITH_CUDA
      .set_attr("inferlayout",
                MakeOpFunction(cinn::hlir::op::InferLayoutForSplit))
#endif
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kNonFusible)
      .set_support_level(4);

  CINN_REGISTER_OP(concat)
      .describe(
          "This operator is used to concat two input tensors X and Y on "
          "specified axis.")
      .set_num_inputs(2)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForConcat)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForConcat))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForConcat))
#ifndef CINN_WITH_CUDA
      .set_attr("inferlayout",
                MakeOpFunction(cinn::hlir::op::InferLayoutForConcat))
#endif
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kInjective)
      .set_support_level(4);

  CINN_REGISTER_OP(reverse)
      .describe("This operator implements the meta op reverse.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForReverse)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForReverse))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForLayoutTransform))
#ifndef CINN_WITH_CUDA
      .set_attr("inferlayout",
                MakeOpFunction(cinn::hlir::op::InferLayoutForReverse))
#endif
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kInjective)
      .set_support_level(4);

  CINN_REGISTER_OP(transpose)
      .describe("This operator implements the meta op transpose.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForTranspose)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForTranspose))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForLayoutTransform))
#ifndef CINN_WITH_CUDA
      .set_attr("inferlayout",
                MakeOpFunction(cinn::hlir::op::InferLayoutForTranspose))
#endif
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kInjective)
      .set_support_level(4);

  CINN_REGISTER_OP(mul)
      .describe(
          "This operator is used to perform matrix multiplication for input X "
          "and Y.")
      .set_num_inputs(2)
      .set_num_outputs(2)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForMul)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForMul))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForMul))
#ifndef CINN_WITH_CUDA
      .set_attr("inferlayout",
                MakeOpFunction(cinn::hlir::op::InferLayoutForMul))
#endif
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kNonFusible)
      .set_support_level(4);

#ifdef CINN_WITH_CUDA
  CINN_REGISTER_OP(cublas_gemm)
      .describe("This operator uses cublas to compute the gemm.")
      .set_num_inputs(3)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForCublasGemm)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForCublasGemm))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForCublasGemm))
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kNonFusible)
      .set_support_level(4);

  CINN_REGISTER_OP(cublas_matmul)
      .describe("This operator uses cublas to compute the matmul.")
      .set_num_inputs(2)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForMatMul)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForMatMul))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForMatMul))
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kNonFusible)
      .set_support_level(4);
#endif

  CINN_REGISTER_OP(layout_transform)
      .describe("This operator is used to transform op's layouts")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForLayoutTransform)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForLayoutTransform))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForLayoutTransform))
#ifndef CINN_WITH_CUDA
      .set_attr("inferlayout",
                MakeOpFunction(cinn::hlir::op::InferLayoutForLayoutTransform))
#endif
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kInjective)
      .set_support_level(4);

  CINN_REGISTER_OP(slice)
      .describe("This operator implements the slice layer")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForSlice)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForSlice))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForSlice))
#ifndef CINN_WITH_CUDA
      .set_attr("inferlayout",
                MakeOpFunction(cinn::hlir::op::InferLayoutForSlice))
#endif
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kInjective)
      .set_support_level(4);

  CINN_REGISTER_OP(slice_assign)
      .describe(
          "This operator is used to perform slice assign for tensor input and "
          "tensor assign.")
      .set_num_inputs(2)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForSliceAssign)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForSliceAssign))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForSliceAssign))
      .set_attr("inferlayout",
                MakeOpFunction(cinn::hlir::op::InferLayoutForSliceAssign))
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  CINN_REGISTER_OP(gather)
      .describe(
          "This operator is used to create a new tensor which indexes the "
          "`input` tensor along dimension `axis` using "
          "the entries in `index`.")
      .set_num_inputs(2)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForGather)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForGather))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForGather))
      .set_attr("inferlayout",
                MakeOpFunction(cinn::hlir::op::InferLayoutForGather))
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kInjective)
      .set_support_level(4);

  CINN_REGISTER_OP(scatter_assign)
      .describe(
          "This operator is used to assign tensor B to tensor A by index.")
      .set_num_inputs(3)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForScatterAssign)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForScatterAssign))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForScatterAssign))
      .set_attr("inferlayout",
                MakeOpFunction(cinn::hlir::op::InferLayoutForScatterAssign))
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kInjective)
      .set_support_level(4);

  CINN_REGISTER_OP(scatter_add)
      .describe(
          "This operator is used to add update tensor B into tensor A by "
          "index.")
      .set_num_inputs(3)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForScatterAdd)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForScatterAdd))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForScatterAdd))
      .set_attr("inferlayout",
                MakeOpFunction(cinn::hlir::op::InferLayoutForScatterAdd))
      // Because the scatter_add operator calls the external function by passing
      // pointers, the code generated by operator fusion will have out-of-bounds
      // access. It should not fuse with any other injective operators, though
      // scatter_add is injective. turn KNonFusible to kInjective will fail
      // /Paddle/python/paddle/base/tests/unittests/test_index_select_op.py
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kNonFusible)
      .set_support_level(4);

  return true;
}
