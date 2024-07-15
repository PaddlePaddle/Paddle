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
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/op/op_util.h"
#include "paddle/cinn/hlir/pe/elementwise.h"
#include "paddle/cinn/hlir/pe/ir_schedule_pe.h"
#include "paddle/cinn/hlir/pe/nn.h"
#include "paddle/cinn/hlir/pe/schedule.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"
#include "paddle/phi/core/enforce.h"

namespace cinn {
namespace hlir {
namespace op {
using cinn::common::_CINNValuePack_;
using cinn::common::CINNValue;
using cinn::common::CINNValuePack;
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

    auto new_shape_A_e = ToCinnExprs(new_shape_A);
    auto new_shape_B_e = ToCinnExprs(new_shape_B);

    auto new_A = tensor_A->Reshape(new_shape_A_e);
    auto new_B = tensor_B->Reshape(new_shape_B_e);

    std::vector<ir::Tensor> out;
    target.arch.Match(
        [&](common::UnknownArch) { CINN_NOT_IMPLEMENTED; },
        [&](common::X86Arch) {
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
        },
        [&](common::ARMArch) { CINN_NOT_IMPLEMENTED; },
        [&](common::NVGPUArch) {
          out = pe::Matmul(new_A, new_B, trans_a, trans_b, alpha, tensor_name);
        },
        [&](common::HygonDCUArchHIP) {
          out = pe::Matmul(new_A, new_B, trans_a, trans_b, alpha, tensor_name);
        });

    std::vector<CINNValue> res;

    for (auto &t : out) {
      res.push_back(CINNValue(t));
    }
    *ret = CINNValuePack{res};
  });

  framework::CINNSchedule matmul_schedule([=](lang::Args args,
                                              lang::RetValue *ret) {
    CHECK(!args.empty())
        << "The input argument of matmul schedule is empty! Please check.\n";
    CINNValuePack arg_pack = args[0];
    std::vector<CINNValue> results =
        pe::IRGpuScheduleMatMul(arg_pack, output_shape, target);
    *ret = CINNValuePack({results});
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(matmul_compute, matmul_schedule, "strategy.matmul.x86", 1);

  return strategy;
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
      << "The Split op doesn't find [num_or_sections] attribute! It it a "
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

        std::vector<CINNValue> res;
        for (int i = 0; i < out.size(); ++i) {
          res.emplace_back(out[i]);
        }
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

    auto out = pe::Concat(input_tensors, axis, tensor_name);

    *ret = CINNValuePack(std::vector<CINNValue>({CINNValue(out)}));
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(concat_compute,
                    GetInjectiveScheduleFunc(output_shapes, target, false),
                    "strategy.concat.x86",
                    1);
  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForConcatSymbolic(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<ir::Dim>> &output_shapes,
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

    auto out = pe::Concat(input_tensors, axis, tensor_name);

    *ret = CINNValuePack(std::vector<CINNValue>({CINNValue(out)}));
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      concat_compute, lang::PackedFunc(), "strategy.concat.x86", 1);
  return strategy;
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

  framework::CINNCompute mul_compute([=](lang::Args args, lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        !args.empty(),
        true,
        phi::errors::InvalidArgument(
            "The input arguments of Mul compute is empty! Please check.\n"));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_GE(pack_args.size(),
                      2U,
                      phi::errors::InvalidArgument(
                          "at least 2 input tensors for Mul compute\n"));
    Expr A = pack_args[0];
    Expr B = pack_args[1];
    PADDLE_ENFORCE_NOT_NULL(A.as_tensor(),
                            phi::errors::InvalidArgument(
                                "The A is not as tensor! Please check.\n"));
    PADDLE_ENFORCE_NOT_NULL(B.as_tensor(),
                            phi::errors::InvalidArgument(
                                "The B is not as tensor! Please check.\n"));

    auto A_tensor = A.as_tensor_ref();
    auto B_tensor = B.as_tensor_ref();

    auto new_shape_A_e = ToCinnExprs(new_shape_A);
    auto new_shape_B_e = ToCinnExprs(new_shape_B);

    auto new_A = A_tensor->Reshape(new_shape_A_e);
    auto new_B = B_tensor->Reshape(new_shape_B_e);

    std::vector<ir::Tensor> out;
    PADDLE_ENFORCE_EQ(pack_args.back().is_string(),
                      true,
                      phi::errors::InvalidArgument(
                          "The pack_args is not string! Please check.\n"));
    std::string tensor_name = pack_args.back().operator std::string();

    target.arch.Match(
        [&](common::UnknownArch) { CINN_NOT_IMPLEMENTED; },
        [&](common::X86Arch) {
#ifdef CINN_WITH_MKL_CBLAS
          out = pe::MatmulMKL(
              new_A, new_B, false, is_infer, 1.0f, tensor_name, target);
#else
          out = pe::MatmulV2(
              new_A, new_B, false, is_infer, 1.0f, tensor_name, target);
#endif
        },
        [&](common::ARMArch) { CINN_NOT_IMPLEMENTED; },
        [&](common::NVGPUArch) {
          out = pe::Matmul(new_A, new_B, false, is_infer, 1.0f, tensor_name);
        },
        [&](common::HygonDCUArchHIP) {
          out = pe::Matmul(new_A, new_B, false, is_infer, 1.0f, tensor_name);
        });

    std::vector<CINNValue> res;

    for (auto &t : out) {
      res.push_back(CINNValue(t));
    }
    *ret = CINNValuePack{res};
  });

  framework::CINNSchedule mul_schedule([=](lang::Args args,
                                           lang::RetValue *ret) {
    CHECK(!args.empty())
        << "The input argument of matmul schedule is empty! Please check.\n";
    CINNValuePack arg_pack = args[0];
    std::vector<CINNValue> results =
        pe::IRGpuScheduleMatMul(arg_pack, output_shape, target);
    *ret = CINNValuePack({results});
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(mul_compute, mul_schedule, "strategy.mul.x86", 1);

  return strategy;
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
        std::vector<CINNValue> res{CINNValue(out)};
        *ret = CINNValuePack{res};
      });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(gemm_compute,
                    GetInjectiveScheduleFunc(output_shapes, target),
                    "strategy.cublas.gemm",
                    1);

  return strategy;
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
    std::vector<CINNValue> res;
    res = {CINNValue(out)};
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

    if (std::holds_alternative<common::X86Arch>(target.arch)) {
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
        PADDLE_THROW(phi::errors::InvalidArgument(
            "axis is not in [0, n_dim), Please check."));
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
    *ret = CINNValuePack{{CINNValue(out)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  CHECK(out_type.size()) << "Out_type of reverse op is empty! Please check.";
  strategy->AddImpl(reverse_compute,
                    GetInjectiveScheduleFunc(output_shapes, target),
                    "strategy.reverse.x86",
                    1);
  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForReverseSymbolic(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<ir::Dim>> &output_shapes,
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
        PADDLE_THROW(::common::errors::InvalidArgument(
            "axis is not in [0, n_dim), Please check."));
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
    *ret = CINNValuePack{{CINNValue(out)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  CHECK(out_type.size()) << "Out_type of reverse op is empty! Please check.";
  strategy->AddImpl(
      reverse_compute, lang::PackedFunc(), "strategy.reverse.x86", 1);
  return strategy;
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
    PADDLE_THROW(
        phi::errors::InvalidArgument("axis is not be set! Please check."));
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
    *ret = CINNValuePack{{CINNValue(out)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(transpose_compute,
                    GetInjectiveScheduleFunc(output_shapes, target),
                    "strategy.transpose.x86",
                    1);
  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForTransposeSymbolic(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<ir::Dim>> &output_shapes,
    const Target &target) {
  // check output shape
  PADDLE_ENFORCE_EQ(output_shapes.empty(),
                    false,
                    ::common::errors::InvalidArgument(
                        "Output shape is empty! Please check.\n"));
  PADDLE_ENFORCE_EQ(output_shapes[0].empty(),
                    false,
                    ::common::errors::InvalidArgument(
                        "Output shape is empty! Please check.\n"));

  std::vector<int> axis;
  auto input_shape = inputs[0]->shape;
  if (attrs.attr_store.find("axis") != attrs.attr_store.end()) {
    axis = absl::get<std::vector<int>>(attrs.attr_store.at("axis"));
    PADDLE_ENFORCE_EQ(axis.size(),
                      output_shapes[0].size(),
                      ::common::errors::InvalidArgument(
                          "axis size is not equal output_shapes size! Please "
                          "check setting.\n"));
    // check axis and shape
    for (int idx = 0; idx < axis.size(); ++idx) {
      PADDLE_ENFORCE(axis[idx] >= 0 && axis[idx] < axis.size(),
                     ::common::errors::InvalidArgument(
                         "axis is not in the tensor shape."));
      for (int idy = idx + 1; idy < axis.size(); ++idy) {
        PADDLE_ENFORCE_NE(axis[idx],
                          axis[idy],
                          ::common::errors::InvalidArgument(
                              "The same axis parameter exists!"));
      }
    }
  } else {
    PADDLE_THROW(
        ::common::errors::InvalidArgument("axis is not be set! Please check."));
  }

  framework::CINNCompute transpose_compute([=](lang::Args args,
                                               lang::RetValue *ret) {
    PADDLE_ENFORCE(
        !args.empty(),
        ::common::errors::InvalidArgument("The input argument of transpose "
                                          "compute is empty! Please check.\n"));
    CINNValuePack input_args = args[0];
    PADDLE_ENFORCE(!input_args.empty(),
                   ::common::errors::InvalidArgument(
                       "at least one input tensor for transpose compute.\n"));
    Expr A = input_args[0];
    PADDLE_ENFORCE(
        A.as_tensor(),
        ::common::errors::InvalidArgument("The input argument is not Tensor."));
    PADDLE_ENFORCE_EQ(input_args.size(),
                      2,
                      ::common::errors::InvalidArgument(
                          "The input args size must be equal to 2."));
    PADDLE_ENFORCE(
        input_args[1].is_string(),
        ::common::errors::InvalidArgument(
            "The second argument must be of type string and is the name "
            "of the output tensor."));
    std::string tensor_name = input_args[1].operator std::string();

    auto out = pe::Transpose(A.as_tensor_ref(), axis, tensor_name);
    *ret = CINNValuePack{{CINNValue(out)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      transpose_compute, lang::PackedFunc(), "strategy.transpose.x86", 1);
  return strategy;
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
        std::vector<CINNValue> res{CINNValue(out)};
        *ret = CINNValuePack{res};
      }};

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(gather_compute,
                    GetInjectiveScheduleFunc(output_shapes, target),
                    "strategy.gather.x86",
                    1);
  return strategy;
}
std::shared_ptr<OpStrategy> StrategyForGatherSymbolic(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<ir::Dim>> &output_shapes,
    const Target &target) {
  PADDLE_ENFORCE_NE(output_shapes.size(),
                    0,
                    ::common::errors::InvalidArgument(
                        "The shape of output is empty! Please check again."));
  PADDLE_ENFORCE_NE(output_shapes[0].size(),
                    0,
                    ::common::errors::InvalidArgument(
                        "The shape of output is empty! Please check again."));

  VLOG(4) << "The output passed in StrategyForGather: "
          << utils::Join(output_shapes[0], ", ");
  PADDLE_ENFORCE_NE(
      out_type.size(),
      0,
      ::common::errors::InvalidArgument(
          "The output type of Gather is empty! Please check again."));

  int axis = 0;
  if (attrs.attr_store.contains("axis")) {
    axis = absl::get<int>(attrs.attr_store.at("axis"));
  }
  axis = axis < 0 ? axis + static_cast<int>(inputs[0]->shape.size()) : axis;

  std::vector<Expr> output_shape = ToCinnExprs(output_shapes[0]);

  framework::CINNCompute gather_compute{
      [axis, output_shape = std::move(output_shape)](lang::Args args,
                                                     lang::RetValue *ret) {
        VLOG(4) << "The axis value used in gather_compute: " << axis;
        PADDLE_ENFORCE_NE(args.size(),
                          0,
                          ::common::errors::InvalidArgument(
                              "The input args are empty! Please check again."));
        CINNValuePack input_args = args[0];
        int input_size = input_args.size();
        PADDLE_ENFORCE_GE(input_size,
                          2,
                          ::common::errors::InvalidArgument(
                              "Require 2 input tensors for Gather compute."));
        Expr x = input_args[0];
        PADDLE_ENFORCE_NE(x.as_tensor(),
                          nullptr,
                          ::common::errors::InvalidArgument(
                              "The first input args's type should be Tensor"));
        Expr index = input_args[1];
        PADDLE_ENFORCE_NE(index.as_tensor(),
                          nullptr,
                          ::common::errors::InvalidArgument(
                              "The first input args's type should be Tensor"));

        std::string tensor_name = input_args[2].operator std::string();

        auto out = pe::Gather(x.as_tensor_ref(),
                              index.as_tensor_ref(),
                              axis,
                              output_shape,
                              tensor_name);
        std::vector<CINNValue> res{CINNValue(out)};
        *ret = CINNValuePack{res};
      }};

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      gather_compute, lang::PackedFunc(), "strategy.gather.x86", 1);
  return strategy;
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

    CHECK_EQ(arg_pack.size(), 4U);
    CHECK(arg_pack[3].is_string());
    std::string tensor_name = arg_pack[3].operator std::string();

    auto out = pe::ScatterAssign(
        tensor_input, tensor_updates, tensor_index, target, axis, tensor_name);

    std::vector<CINNValue> res;
    res.push_back(CINNValue(out));
    CHECK(!out_type.empty())
        << "Output type of ScatterAssign is empty! Please check.\n";
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(scatter_assign_compute,
                    GetInjectiveScheduleFunc(output_shapes, target, false),
                    "strategy.scatter_assign.x86",
                    1);
  return strategy;
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

    CHECK_EQ(arg_pack.size(), 4U);
    CHECK(arg_pack[3].is_string());
    std::string tensor_name = arg_pack[3].operator std::string();

    auto out = pe::ScatterAdd(
        tensor_input, tensor_updates, tensor_index, target, axis, tensor_name);

    std::vector<CINNValue> res;
    res.push_back(CINNValue(out));
    CHECK(!out_type.empty())
        << "Output type of ScatterAdd is empty! Please check.\n";
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(scatter_add_compute,
                    GetInjectiveScheduleFunc(output_shapes, target, false),
                    "strategy.scatter_add.x86",
                    1);
  return strategy;
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

  CHECK(!starts.empty()) << "The Slice op doesn't find [starts] attribute! It "
                            "it a mandatory attribute, please check.";
  CHECK(!ends.empty()) << "The Slice op doesn't find [ends] attribute! It it a "
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
        *ret = CINNValuePack{{CINNValue(out)}};
      });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(slice_compute,
                    GetInjectiveScheduleFunc(output_shapes, target),
                    "strategy.slice.x86",
                    1);

  return strategy;
}

template <typename T = int>
std::vector<T> GetIntVectorFromAttr(const utils::Attribute &attr) {
  if (absl::holds_alternative<std::vector<int64_t>>(attr)) {
    const auto &attr_data = absl::get<std::vector<int64_t>>(attr);
    return std::vector<T>(attr_data.begin(), attr_data.end());
  } else if (absl::holds_alternative<std::vector<int>>(attr)) {
    const auto &attr_data = absl::get<std::vector<int>>(attr);
    return std::vector<T>(attr_data.begin(), attr_data.end());
  } else if (absl::holds_alternative<bool>(attr)) {
    return std::vector<T>{};
  } else {
    PADDLE_THROW(
        phi::errors::InvalidArgument("attribute's vector type is invalid!"));
  }
}
std::shared_ptr<OpStrategy> StrategyForSliceSymbolic(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<ir::Dim>> &output_shapes,
    const Target &target) {
  const std::vector<Expr> starts_expr = [&] {
    if (inputs.size() == 3) {
      const auto &value = inputs.at(1).self()->value();
      CHECK(value.has_value());
      return value.value();
    }
    if (attrs.attr_store.find("starts") != attrs.attr_store.end()) {
      return ToCinnExprs(GetIntVectorFromAttr(attrs.attr_store.at("starts")));
    } else {
      PADDLE_THROW(::common::errors::InvalidArgument(
          "The Slice op doesn't find [starts] attribute!"));
    }
  }();
  const std::vector<Expr> ends_expr = [&] {
    if (inputs.size() == 3) {
      const auto &value = inputs.at(2).self()->value();
      CHECK(value.has_value());
      return value.value();
    }
    if (attrs.attr_store.find("ends") != attrs.attr_store.end()) {
      return ToCinnExprs(GetIntVectorFromAttr(attrs.attr_store.at("ends")));
    } else {
      PADDLE_THROW(::common::errors::InvalidArgument(
          "The Slice op doesn't find [ends] attribute!"));
    }
  }();
  const std::vector<int> axes = [&] {
    std::vector<int> axes;
    if (attrs.attr_store.find("axes") != attrs.attr_store.end()) {
      axes = GetIntVectorFromAttr(attrs.attr_store.at("axes"));
    }
    if (axes.empty()) {
      for (int i = 0; i < starts_expr.size(); i++) {
        axes.push_back(i);
      }
    }
    return axes;
  }();
  const std::vector<Expr> strides_expr = [&] {
    std::vector<int> strides;
    if (attrs.attr_store.find("strides") != attrs.attr_store.end()) {
      strides = GetIntVectorFromAttr(attrs.attr_store.at("strides"));
    }
    if (strides.empty()) {
      for (int i = 0; i < starts_expr.size(); i++) {
        strides.push_back(1);
      }
    }
    return ToCinnExprs(strides);
  }();
  const std::vector<int> decrease_axis = [&] {
    if (attrs.attr_store.find("decrease_axis") != attrs.attr_store.end()) {
      return GetIntVectorFromAttr(attrs.attr_store.at("decrease_axis"));
    }
    return std::vector<int>{};
  }();

  CHECK(!starts_expr.empty())
      << "The Slice op doesn't find [starts] attribute! It "
         "it a mandatory attribute, please check.";
  CHECK(!ends_expr.empty())
      << "The Slice op doesn't find [ends] attribute! It it a "
         "mandatory attribute, please check.";
  CHECK_EQ(starts_expr.size(), ends_expr.size())
      << "The size of [starts] and [ends] must be identical! Please check.";
  CHECK_EQ(starts_expr.size(), axes.size())
      << "The size of [starts] and [axes] must be identical! Please check.";
  CHECK_EQ(starts_expr.size(), strides_expr.size())
      << "The size of [starts] and [strides] must be identical! Please "
         "check.";

  std::vector<Expr> output_shape;
  for (auto &i : output_shapes[0]) {
    output_shape.push_back(i->dim_expr);
    CHECK(output_shape.back().type().valid());
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

        const std::string tensor_name = [&] {
          if (arg_pack.size() == 2 || arg_pack.size() == 4) {
            CHECK(arg_pack.back().is_string());
            return arg_pack.back().operator std::string();
          }
          PADDLE_THROW(::common::errors::InvalidArgument(
              "The slice op doesn't find output tensor name! The size of "
              "arg_pack is %d.",
              arg_pack.size()));
        }();

        auto out = pe::SliceSymbolic(A,
                                     starts_expr,
                                     axes,
                                     strides_expr,
                                     decrease_axis,
                                     output_shape,
                                     tensor_name);
        VLOG(4) << "out: " << out;
        *ret = CINNValuePack{{CINNValue(out)}};
      });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(slice_compute, lang::PackedFunc(), "strategy.slice.x86", 1);

  return strategy;
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

  CHECK(!starts.empty()) << "The SliceAssign op doesn't find [starts] "
                            "attribute! It it a mandatory "
                            "attribute, please check.";
  CHECK(!ends.empty())
      << "The SliceAssign op doesn't find [ends] attribute! It "
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
        std::vector<CINNValue> res{CINNValue(out)};
        *ret = CINNValuePack{res};
      }};

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(slice_assign_compute,
                    GetInjectiveScheduleFunc(output_shapes, target),
                    "strategy.slice_assign.x86",
                    1);
  return strategy;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(transform_ops) {
  CINN_REGISTER_OP(split)
      .describe(
          "This operator is used to split tensors X to 'sections' sub-tensor "
          "on specified axis.")
      .set_num_inputs(1)
      .set_num_outputs(0)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForSplit)
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
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(
          "CINNStrategySymbolic", cinn::hlir::op::StrategyForConcatSymbolic)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kInjective)
      .set_support_level(4);

  CINN_REGISTER_OP(reverse)
      .describe("This operator implements the meta op reverse.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForReverse)
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(
          "CINNStrategySymbolic", cinn::hlir::op::StrategyForReverseSymbolic)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kInjective)
      .set_support_level(4);

  CINN_REGISTER_OP(transpose)
      .describe("This operator implements the meta op transpose.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForTranspose)
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(
          "CINNStrategySymbolic", cinn::hlir::op::StrategyForTransposeSymbolic)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kInjective)
      .set_support_level(4);

  CINN_REGISTER_OP(slice)
      .describe("This operator implements the slice layer")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForSlice)
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(
          "CINNStrategySymbolic", cinn::hlir::op::StrategyForSliceSymbolic)
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
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(
          "CINNStrategySymbolic", cinn::hlir::op::StrategyForGatherSymbolic)
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
