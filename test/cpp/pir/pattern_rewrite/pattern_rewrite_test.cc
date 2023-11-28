// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/constant_folding_pass.h"
#include "paddle/fluid/pir/transforms/dead_code_elimination_pass.h"
#include "paddle/fluid/pir/transforms/transform_general_functions.h"

#include "paddle/pir/core/builder.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/cast_utils.h"
#include "paddle/pir/core/dialect.h"
#include "paddle/pir/core/enforce.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/op_info.h"
#include "paddle/pir/core/parameter.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/core/value.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_manager.h"
#include "paddle/pir/pattern_rewrite/frozen_rewrite_pattern_set.h"
#include "paddle/pir/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/pattern_rewrite/pattern_match.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/kernel_registry.h"

// build Conv2dFusionOp
#include "paddle/fluid/pir/dialect/operator/interface/infermeta.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/transforms/fusion/conv2d_fuse_pass.h"
#include "paddle/fluid/pir/transforms/fusion/fc_fuse_pass.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/infermeta/multiary.h"
#include "paddle/pir/core/op_base.h"

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(sqrt, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(divide, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(multiply, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(subtract, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(full_int_array, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(reshape, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(fetch, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(conv2d, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(transpose, CPU, ALL_LAYOUT);

// Define op1.
class Operation1 : public pir::Op<Operation1> {
 public:
  using Op::Op;
  static const char *name() { return "test.Operation1"; }
  static constexpr uint32_t attributes_num = 2;
  static const char *attributes_name[attributes_num];  // NOLINT
  void VerifySig();
  static void InferShape() { VLOG(2) << "This is op2's InferShape interface."; }
};

void Operation1::VerifySig() {
  auto &attributes = this->attributes();
  if (attributes.count("op2_attr1") == 0 ||
      (!attributes.at("op2_attr1").isa<pir::StrAttribute>())) {
    throw("Type of attribute: parameter_name is not right.");
  }
  if (attributes.count("op2_attr2") == 0 ||
      (!attributes.at("op2_attr2").isa<pir::StrAttribute>())) {
    throw("Type of attribute: parameter_name is not right.");
  }
}
const char *Operation1::attributes_name[attributes_num] = {  // NOLINT
    "op2_attr1",
    "op2_attr2"};
IR_DECLARE_EXPLICIT_TYPE_ID(Operation1)
IR_DEFINE_EXPLICIT_TYPE_ID(Operation1)

// Define a dialect, op1 and op2 will be registered by this dialect.
class TestDialect : public pir::Dialect {
 public:
  explicit TestDialect(pir::IrContext *context)
      : pir::Dialect(name(), context, pir::TypeId::get<TestDialect>()) {
    initialize();
  }
  static const char *name() { return "test"; }

 private:
  void initialize() { RegisterOps<Operation1>(); }
};
IR_DECLARE_EXPLICIT_TYPE_ID(TestDialect)
IR_DEFINE_EXPLICIT_TYPE_ID(TestDialect)

// TODO(wilber): Add logical when ir support erase, replace or update.
class TestPatternRewrite : public pir::OpRewritePattern<Operation1> {
 public:
  using pir::OpRewritePattern<Operation1>::OpRewritePattern;

  void Rewrite(Operation1 op, pir::PatternRewriter &rewriter) const override {}
  bool Match(Operation1 op) const override { return false; }
};

class TestPatternRewrite2 : public pir::OpRewritePattern<Operation1> {
 public:
  using pir::OpRewritePattern<Operation1>::OpRewritePattern;
  bool MatchAndRewrite(
      Operation1 op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT
    return false;
  }
};

TEST(PatternRewrite, PatternBenefit) {
  pir::PatternBenefit benefit1(1);
  EXPECT_EQ(benefit1.benefit(), 1U);
  pir::PatternBenefit benefit2(2);
  EXPECT_EQ(benefit2.benefit(), 2U);

  EXPECT_TRUE(benefit2 > benefit1);
  EXPECT_TRUE(benefit2 >= benefit1);
  EXPECT_TRUE(benefit1 < benefit2);
  EXPECT_TRUE(benefit1 <= benefit2);
  EXPECT_TRUE(benefit1 != benefit2);
  pir::PatternBenefit benefit3(2);
  EXPECT_TRUE(benefit2 == benefit3);
}

TEST(RewritePattern, RewritePatternSet) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  auto *test_dialect = ctx->GetOrRegisterDialect<TestDialect>();
  test_dialect->RegisterOp<Operation1>();

  pir::RewritePatternSet ps(ctx);
  ps.Add<TestPatternRewrite>(ctx, 1);
  EXPECT_EQ(ps.native_patterns().size(), 1U);
  EXPECT_TRUE(ps.native_patterns().back()->debug_labels().empty());
  EXPECT_EQ(ps.native_patterns().back()->benefit(), 1U);
  ps.AddWithLabel<TestPatternRewrite2>({"TestPatternRewrite2"}, ctx, 2);
  EXPECT_EQ(ps.native_patterns().size(), 2U);
  EXPECT_EQ(ps.native_patterns().back()->debug_labels()[0],
            "TestPatternRewrite2");
  EXPECT_EQ(ps.native_patterns().back()->benefit(), 2U);

  ps.Clear();
  ps.Add<TestPatternRewrite, TestPatternRewrite2>(ctx, 2);
  EXPECT_EQ(ps.native_patterns().size(), 2U);
  EXPECT_EQ(ps.native_patterns()[0]->benefit(), 2U);
  EXPECT_EQ(ps.native_patterns()[1]->benefit(), 2U);
}

// TODO(wilber): Add actual case.
// TEST(PatternRewrite, PatternApplicator) {
//   pir::IrContext *ctx = pir::IrContext::Instance();
//   ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
//   auto *test_dialect = ctx->GetOrRegisterDialect<TestDialect>();
//   test_dialect->RegisterOp<Operation1>();
//   pir::RewritePatternSet ps(ctx);
//   ps.Add<TestPatternRewrite, TestPatternRewrite2>(ctx, 2);
//   pir::FrozenRewritePatternSet frozen_set(std::move(ps));
//   pir::PatternApplicator applicator(frozen_set);
//   applicator.ApplyDefaultCostModel();
// }

// // TODO(wilber): Add actual case.
TEST(PatternRewrite, FrozenRewritePatternSet) {
  pir::FrozenRewritePatternSet frozen_set;
  EXPECT_TRUE(frozen_set.match_any_op_native_patterns().empty());
  EXPECT_TRUE(frozen_set.op_specific_native_patterns().empty());

  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  auto *test_dialect = ctx->GetOrRegisterDialect<TestDialect>();
  test_dialect->RegisterOp<Operation1>();
  pir::RewritePatternSet ps(ctx);
  ps.Add<TestPatternRewrite, TestPatternRewrite2>(ctx, 2);

  pir::FrozenRewritePatternSet frozen_set2(std::move(ps));
  EXPECT_TRUE(frozen_set2.match_any_op_native_patterns().empty());
  const auto &pattern_maps = frozen_set2.op_specific_native_patterns();
  EXPECT_EQ(pattern_maps.size(), 1U);
  EXPECT_EQ(pattern_maps.at(ctx->GetRegisteredOpInfo("test.Operation1")).size(),
            2U);
}

class RedundantTransposeFusePattern
    : public pir::OpRewritePattern<paddle::dialect::TransposeOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::TransposeOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::TransposeOp op,
                       pir::PatternRewriter &rewriter) const override {
    auto prev_op = pir::GetDefiningOpForInput(op, 0);
    std::vector<int> axis_last = GetAxis(op);
    auto prev_trans_op = prev_op->dyn_cast<paddle::dialect::TransposeOp>();
    if (prev_trans_op) {
      std::vector<int> axis_first = GetAxis(prev_trans_op);
      IR_ENFORCE(axis_first.size() == axis_last.size(),
                 "tranpose op's perm rank should be same.");
      auto new_perm = GetPerm(axis_first, axis_last);
      rewriter.set_insertion_point(op);
      auto new_transpose_op = rewriter.Build<paddle::dialect::TransposeOp>(
          pir::GetDefiningOpForInput(prev_trans_op, 0)->result(0), new_perm);
      rewriter.ReplaceOp(op, {new_transpose_op.out()});
      return true;
    }

    return false;
  }

 private:
  std::vector<int> GetAxis(paddle::dialect::TransposeOp op) const {
    auto array_attr = op.attribute<pir::ArrayAttribute>("perm").AsVector();
    std::vector<int> axis(array_attr.size());
    for (size_t i = 0; i < array_attr.size(); ++i) {
      axis[i] = array_attr[i].dyn_cast<pir::Int32Attribute>().data();
    }
    return axis;
  }

  std::vector<int> GetPerm(const std::vector<int> &perm1,
                           const std::vector<int> &perm2) const {
    int n = static_cast<int>(perm1.size());
    std::vector<int> axis(n), axis1(n), axis2(n);
    std::iota(axis.begin(), axis.end(), 0);
    for (int i = 0; i < n; ++i) {
      axis1[i] = axis[perm1[i]];
    }
    for (int i = 0; i < n; ++i) {
      axis2[i] = axis1[perm2[i]];
    }
    return axis2;
  }
};

namespace paddle {
namespace dialect {
class Conv2dFusionOpTest : public pir::Op<Conv2dFusionOpTest,
                                          OpYamlInfoInterface,
                                          InferMetaInterface> {
 public:
  using Op::Op;
  static const char *name() { return "pd_op.conv2d_fusion_test"; }
  static const char *attributes_name[10];  // NOLINT
  static constexpr uint32_t attributes_num = 10;
  static OpInfoTuple GetOpInfo();
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::OpResult input_,
                    pir::OpResult filter_,
                    pir::OpResult bias_,
                    pir::OpResult residual_,
                    const std::vector<int> &strides,
                    const std::vector<int> &paddings_t,
                    std::string padding_algorithm,
                    const std::vector<int> &dilations_t,
                    int groups,
                    std::string data_format,
                    std::string activation,
                    bool exhaustive_search,
                    const std::vector<int> &channels,
                    int user_workspace_size);

  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::OpResult input_,
                    pir::OpResult filter_,
                    pir::OpResult bias_,
                    pir::OpResult residual_,
                    pir::AttributeMap attributes);
  void VerifySig();
  pir::Value input() { return operand_source(0); }
  pir::Value filter() { return operand_source(1); }
  pir::Value bias() { return operand_source(2); }
  pir::Value residual() { return operand_source(3); }
  pir::OpResult output() { return result(0); }
  pir::OpResult outputs() { return result(1); }
  pir::Attribute attribute(const std::string &name) {
    {
      PADDLE_ENFORCE(
          attributes().count(name) > 0,
          phi::errors::PreconditionNotMet("Attribute is not exist."));
      return attributes().at(name);
    }
  }
  template <typename T>
  T attribute(const std::string &name) {
    {
      PADDLE_ENFORCE(
          attributes().count(name) > 0 && attributes().at(name).isa<T>(),
          phi::errors::PreconditionNotMet("Attribute is not right."));
      return attributes().at(name).dyn_cast<T>();
    }
  }

  static void InferMeta(phi::InferMetaContext *infer_meta);
};

const char *Conv2dFusionOpTest::attributes_name[10] = {  // NOLINT
    "strides",
    "paddings_t",
    "padding_algorithm",
    "dilations_t",
    "groups",
    "data_format",
    "activation",
    "exhaustive_search",
    "channels",
    "user_workspace_size"};

OpInfoTuple Conv2dFusionOpTest::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      OpInputInfo("input",
                  "paddle::dialect::DenseTensorType",
                  false,
                  false,
                  false,
                  true),
      OpInputInfo("filter",
                  "paddle::dialect::DenseTensorType",
                  false,
                  false,
                  false,
                  true),
      OpInputInfo("bias",
                  "paddle::dialect::DenseTensorType",
                  false,
                  false,
                  false,
                  true),
      OpInputInfo("residual",
                  "paddle::dialect::DenseTensorType",
                  true,
                  false,
                  false,
                  true)};
  std::vector<paddle::dialect::OpAttributeInfo> attributes = {
      OpAttributeInfo(
          "strides", "pir::ArrayAttribute<pir::Int32Attribute>", ""),
      OpAttributeInfo(
          "paddings_t", "pir::ArrayAttribute<pir::Int32Attribute>", ""),
      OpAttributeInfo("padding_algorithm", "pir::StrAttribute", ""),
      OpAttributeInfo(
          "dilations_t", "pir::ArrayAttribute<pir::Int32Attribute>", ""),
      OpAttributeInfo("groups", "pir::Int32Attribute", ""),
      OpAttributeInfo("data_format", "pir::StrAttribute", ""),
      OpAttributeInfo("activation", "pir::StrAttribute", ""),
      OpAttributeInfo("exhaustive_search", "pir::BoolAttribute", ""),
      OpAttributeInfo(
          "channels", "pir::ArrayAttribute<pir::Int32Attribute>", ""),
      OpAttributeInfo("user_workspace_size", "pir::Int32Attribute", "")};
  std::vector<paddle::dialect::OpOutputInfo> outputs = {
      OpOutputInfo("output", "paddle::dialect::DenseTensorType", false, false),
      OpOutputInfo("outputs",
                   "pir::VectorType<paddle::dialect::DenseTensorType>",
                   false,
                   false)};
  paddle::dialect::OpRunTimeInfo run_time_info =
      OpRunTimeInfo("Conv2dFusionInferMeta",
                    {"input",
                     "filter",
                     "bias",
                     "residual",
                     "strides",
                     "paddings_t",
                     "padding_algorithm",
                     "dilations_t",
                     "groups",
                     "data_format",
                     "activation",
                     "exhaustive_search",
                     "channels",
                     "user_workspace_size"},
                    "ConvFusionKernel",
                    {"input",
                     "filter",
                     "bias",
                     "residual",
                     "strides",
                     "paddings_t",
                     "padding_algorithm",
                     "dilations_t",
                     "groups",
                     "data_format",
                     "activation",
                     "exhaustive_search",
                     "channels",
                     "user_workspace_size"},
                    {"input"},
                    {},
                    {},
                    {});

  return std::make_tuple(
      inputs, attributes, outputs, run_time_info, "conv2d_fusion_test");
}

void Conv2dFusionOpTest::Build(pir::Builder &builder,
                               pir::OperationArgument &argument,
                               pir::OpResult input_,
                               pir::OpResult filter_,
                               pir::OpResult bias_,
                               pir::OpResult residual_,
                               pir::AttributeMap attributes) {
  std::vector<int> strides;
  for (size_t i = 0;
       i < attributes.at("strides").dyn_cast<pir::ArrayAttribute>().size();
       i++) {
    strides.push_back(attributes.at("strides")
                          .dyn_cast<pir::ArrayAttribute>()
                          .at(i)
                          .dyn_cast<pir::Int32Attribute>()
                          .data());
  }

  std::vector<int> paddings_t;
  for (size_t i = 0;
       i < attributes.at("paddings_t").dyn_cast<pir::ArrayAttribute>().size();
       i++) {
    paddings_t.push_back(attributes.at("paddings_t")
                             .dyn_cast<pir::ArrayAttribute>()
                             .at(i)
                             .dyn_cast<pir::Int32Attribute>()
                             .data());
  }

  std::string padding_algorithm = attributes.at("padding_algorithm")
                                      .dyn_cast<pir::StrAttribute>()
                                      .AsString();
  std::vector<int> dilations_t;
  for (size_t i = 0;
       i < attributes.at("dilations_t").dyn_cast<pir::ArrayAttribute>().size();
       i++) {
    dilations_t.push_back(attributes.at("dilations_t")
                              .dyn_cast<pir::ArrayAttribute>()
                              .at(i)
                              .dyn_cast<pir::Int32Attribute>()
                              .data());
  }
  int groups = attributes.at("groups").dyn_cast<pir::Int32Attribute>().data();
  std::string data_format =
      attributes.at("data_format").dyn_cast<pir::StrAttribute>().AsString();
  std::string activation =
      attributes.at("activation").dyn_cast<pir::StrAttribute>().AsString();
  bool exhaustive_search =
      attributes.at("exhaustive_search").dyn_cast<pir::BoolAttribute>().data();
  std::vector<int> channels;
  for (size_t i = 0;
       i < attributes.at("channels").dyn_cast<pir::ArrayAttribute>().size();
       i++) {
    channels.push_back(attributes.at("channels")
                           .dyn_cast<pir::ArrayAttribute>()
                           .at(i)
                           .dyn_cast<pir::Int32Attribute>()
                           .data());
  }
  int user_workspace_size = attributes.at("user_workspace_size")
                                .dyn_cast<pir::Int32Attribute>()
                                .data();

  VLOG(4) << "Builder construction inputs";
  std::vector<pir::OpResult> argument_inputs = {
      input_, filter_, bias_, residual_};
  argument.AddInputs(argument_inputs.begin(), argument_inputs.end());

  VLOG(4) << "Builder construction attributes";
  std::vector<pir::Attribute> vec_strides;
  for (auto stride : strides) {
    pir::Attribute attr_strides =
        pir::Int32Attribute::get(pir::IrContext::Instance(), stride);

    vec_strides.push_back(attr_strides);
  }
  pir::Attribute attr_strides =
      pir::ArrayAttribute::get(pir::IrContext::Instance(), vec_strides);
  argument.AddAttribute("strides", attr_strides);
  std::vector<pir::Attribute> vec_paddings_t;
  for (auto padding : paddings_t) {
    pir::Attribute attr_paddings_t =
        pir::Int32Attribute::get(pir::IrContext::Instance(), padding);

    vec_paddings_t.push_back(attr_paddings_t);
  }
  pir::Attribute attr_paddings_t =
      pir::ArrayAttribute::get(pir::IrContext::Instance(), vec_paddings_t);
  argument.AddAttribute("paddings_t", attr_paddings_t);
  pir::Attribute attr_padding_algorithm =
      pir::StrAttribute::get(pir::IrContext::Instance(), padding_algorithm);
  argument.AddAttribute("padding_algorithm", attr_padding_algorithm);
  std::vector<pir::Attribute> vec_dilations_t;
  for (auto dilation : dilations_t) {
    pir::Attribute attr_dilations_t =
        pir::Int32Attribute::get(pir::IrContext::Instance(), dilation);

    vec_dilations_t.push_back(attr_dilations_t);
  }
  pir::Attribute attr_dilations_t =
      pir::ArrayAttribute::get(pir::IrContext::Instance(), vec_dilations_t);
  argument.AddAttribute("dilations_t", attr_dilations_t);
  pir::Attribute attr_groups =
      pir::Int32Attribute::get(pir::IrContext::Instance(), groups);
  argument.AddAttribute("groups", attr_groups);
  pir::Attribute attr_data_format =
      pir::StrAttribute::get(pir::IrContext::Instance(), data_format);
  argument.AddAttribute("data_format", attr_data_format);
  pir::Attribute attr_activation =
      pir::StrAttribute::get(pir::IrContext::Instance(), activation);
  argument.AddAttribute("activation", attr_activation);
  pir::Attribute attr_exhaustive_search =
      pir::BoolAttribute::get(pir::IrContext::Instance(), exhaustive_search);
  argument.AddAttribute("exhaustive_search", attr_exhaustive_search);
  std::vector<pir::Attribute> vec_channels;
  for (auto channel : channels) {
    pir::Attribute attr_channels =
        pir::Int32Attribute::get(pir::IrContext::Instance(), channel);

    vec_channels.push_back(attr_channels);
  }
  pir::Attribute attr_channels =
      pir::ArrayAttribute::get(pir::IrContext::Instance(), vec_channels);
  argument.AddAttribute("channels", attr_channels);
  pir::Attribute attr_user_workspace_size =
      pir::Int32Attribute::get(pir::IrContext::Instance(), user_workspace_size);
  argument.AddAttribute("user_workspace_size", attr_user_workspace_size);

  VLOG(4) << "Builder construction outputs";
  paddle::dialect::DenseTensorType input =
      input_.type().dyn_cast<paddle::dialect::DenseTensorType>();
  (void)input;
  paddle::dialect::DenseTensorType filter =
      filter_.type().dyn_cast<paddle::dialect::DenseTensorType>();
  (void)filter;
  paddle::dialect::DenseTensorType bias =
      bias_.type().dyn_cast<paddle::dialect::DenseTensorType>();
  (void)bias;
  // paddle::dialect::DenseTensorType residual =
  // residual_.type().dyn_cast<paddle::dialect::DenseTensorType>();
  // (void)residual;

  VLOG(4) << "Builder construction  dense_input";
  phi::DenseTensor dense_input(
      std::make_unique<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace())
          .get(),
      phi::DenseTensorMeta(TransToPhiDataType(input.dtype()),
                           input.dims(),
                           input.data_layout(),
                           input.lod(),
                           input.offset()));
  VLOG(4) << "Builder construction  meta_input";
  phi::MetaTensor meta_input(&dense_input);

  VLOG(4) << "Builder construction  dense_filter";
  phi::DenseTensor dense_filter(
      std::make_unique<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace())
          .get(),
      phi::DenseTensorMeta(TransToPhiDataType(filter.dtype()),
                           filter.dims(),
                           filter.data_layout(),
                           filter.lod(),
                           filter.offset()));
  VLOG(4) << "Builder construction  meta_filter";
  phi::MetaTensor meta_filter(&dense_filter);

  VLOG(4) << "Builder construction  dense_bias";
  phi::DenseTensor dense_bias(
      std::make_unique<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace())
          .get(),
      phi::DenseTensorMeta(TransToPhiDataType(bias.dtype()),
                           bias.dims(),
                           bias.data_layout(),
                           bias.lod(),
                           bias.offset()));
  VLOG(4) << "Builder construction  meta_bias";
  phi::MetaTensor meta_bias(&dense_bias);

  // VLOG(4) << "Builder construction  dense_residual";
  // phi::DenseTensor
  // dense_residual(std::make_unique<paddle::experimental::DefaultAllocator>(paddle::platform::CPUPlace()).get(),
  //                               phi::DenseTensorMeta(TransToPhiDataType(residual.dtype()),
  //                                                    residual.dims(),
  //                                                    residual.data_layout(),
  //                                                    residual.lod(),
  //                                                    residual.offset()));
  VLOG(4) << "Builder construction  meta_residual";
  // phi::MetaTensor meta_residual(&dense_residual);
  phi::MetaTensor meta_residual;
  phi::DenseTensor dense_output;
  phi::MetaTensor meta_output(&dense_output);
  std::vector<phi::DenseTensor> vec_dense_outputs((channels.size()),
                                                  phi::DenseTensor());
  std::vector<phi::MetaTensor> vec_meta_outputs;
  for (size_t i = 0; i < static_cast<size_t>(channels.size()); i++) {
    vec_meta_outputs.push_back(phi::MetaTensor(&vec_dense_outputs[i]));
  }
  std::vector<phi::MetaTensor *> meta_outputs;
  for (auto &vec_meta_output : vec_meta_outputs) {
    meta_outputs.push_back(&vec_meta_output);
  }

  phi::FusedConvInferMeta(meta_input,
                          meta_filter,
                          meta_bias,
                          meta_residual,
                          strides,
                          paddings_t,
                          padding_algorithm,
                          dilations_t,
                          groups,
                          data_format,
                          "float32",
                          "identity",
                          false,
                          false,
                          &meta_output,
                          phi::MetaConfig());

  std::vector<pir::Type> argument_outputs;
  auto output_dense_tensor_type = paddle::dialect::DenseTensorType::get(
      pir::IrContext::Instance(),
      TransToIrDataType(dense_output.dtype()),
      dense_output.dims(),
      dense_output.layout(),
      dense_output.lod(),
      dense_output.offset());
  LOG(INFO) << output_dense_tensor_type;

  argument_outputs.push_back(output_dense_tensor_type);

  std::vector<pir::Type> outputs_types;
  for (size_t i = 0; i < static_cast<size_t>(channels.size()); i++) {
    outputs_types.push_back(paddle::dialect::DenseTensorType::get(
        pir::IrContext::Instance(),
        TransToIrDataType(vec_dense_outputs[i].dtype()),
        vec_dense_outputs[i].dims(),
        vec_dense_outputs[i].layout(),
        vec_dense_outputs[i].lod(),
        vec_dense_outputs[i].offset()));
  }
  pir::Type outputs_vector_type =
      pir::VectorType::get(pir::IrContext::Instance(), outputs_types);
  argument_outputs.push_back(outputs_vector_type);
  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
}

void Conv2dFusionOpTest::VerifySig() {
  VLOG(4)
      << "Start Verifying inputs, outputs and attributes for: Conv2dFusionOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    PADDLE_ENFORCE_EQ(
        input_size,
        4u,
        phi::errors::PreconditionNotMet(
            "The size %d of inputs must be equal to 4.", input_size));
    PADDLE_ENFORCE((*this)
                       ->operand_source(0)
                       .type()
                       .isa<paddle::dialect::DenseTensorType>(),
                   phi::errors::PreconditionNotMet(
                       "Type validation failed for the 0th input."));
    PADDLE_ENFORCE((*this)
                       ->operand_source(1)
                       .type()
                       .isa<paddle::dialect::DenseTensorType>(),
                   phi::errors::PreconditionNotMet(
                       "Type validation failed for the 1th input."));
    PADDLE_ENFORCE((*this)
                       ->operand_source(2)
                       .type()
                       .isa<paddle::dialect::DenseTensorType>(),
                   phi::errors::PreconditionNotMet(
                       "Type validation failed for the 2th input."));
    if (auto val = (*this)->operand(3)) {
      PADDLE_ENFORCE(val.type().isa<paddle::dialect::DenseTensorType>(),
                     phi::errors::PreconditionNotMet(
                         "Type validation failed for the 3th input."));
    }
  }
  VLOG(4) << "Verifying attributes:";
  {
    auto &attributes = this->attributes();
    PADDLE_ENFORCE(attributes.count("strides") > 0 &&
                       attributes.at("strides").isa<pir::ArrayAttribute>(),
                   phi::errors::PreconditionNotMet(
                       "Type of attribute: strides is not right."));
    for (size_t i = 0;
         i < attributes.at("strides").dyn_cast<pir::ArrayAttribute>().size();
         i++) {
      PADDLE_ENFORCE(attributes.at("strides")
                         .dyn_cast<pir::ArrayAttribute>()
                         .at(i)
                         .isa<pir::Int32Attribute>(),
                     phi::errors::PreconditionNotMet(
                         "Type of attribute: strides is not right."));
    }
    PADDLE_ENFORCE(attributes.count("paddings_t") > 0 &&
                       attributes.at("paddings_t").isa<pir::ArrayAttribute>(),
                   phi::errors::PreconditionNotMet(
                       "Type of attribute: paddings_t is not right."));
    for (size_t i = 0;
         i < attributes.at("paddings_t").dyn_cast<pir::ArrayAttribute>().size();
         i++) {
      PADDLE_ENFORCE(attributes.at("paddings_t")
                         .dyn_cast<pir::ArrayAttribute>()
                         .at(i)
                         .isa<pir::Int32Attribute>(),
                     phi::errors::PreconditionNotMet(
                         "Type of attribute: paddings_t is not right."));
    }
    PADDLE_ENFORCE(
        attributes.count("padding_algorithm") > 0 &&
            attributes.at("padding_algorithm").isa<pir::StrAttribute>(),
        phi::errors::PreconditionNotMet(
            "Type of attribute: padding_algorithm is not right."));
    PADDLE_ENFORCE(attributes.count("dilations_t") > 0 &&
                       attributes.at("dilations_t").isa<pir::ArrayAttribute>(),
                   phi::errors::PreconditionNotMet(
                       "Type of attribute: dilations_t is not right."));
    for (size_t i = 0;
         i <
         attributes.at("dilations_t").dyn_cast<pir::ArrayAttribute>().size();
         i++) {
      PADDLE_ENFORCE(attributes.at("dilations_t")
                         .dyn_cast<pir::ArrayAttribute>()
                         .at(i)
                         .isa<pir::Int32Attribute>(),
                     phi::errors::PreconditionNotMet(
                         "Type of attribute: dilations_t is not right."));
    }
    PADDLE_ENFORCE(attributes.count("groups") > 0 &&
                       attributes.at("groups").isa<pir::Int32Attribute>(),
                   phi::errors::PreconditionNotMet(
                       "Type of attribute: groups is not right."));
    PADDLE_ENFORCE(attributes.count("data_format") > 0 &&
                       attributes.at("data_format").isa<pir::StrAttribute>(),
                   phi::errors::PreconditionNotMet(
                       "Type of attribute: data_format is not right."));
    PADDLE_ENFORCE(attributes.count("activation") > 0 &&
                       attributes.at("activation").isa<pir::StrAttribute>(),
                   phi::errors::PreconditionNotMet(
                       "Type of attribute: activation is not right."));
    PADDLE_ENFORCE(
        attributes.count("exhaustive_search") > 0 &&
            attributes.at("exhaustive_search").isa<pir::BoolAttribute>(),
        phi::errors::PreconditionNotMet(
            "Type of attribute: exhaustive_search is not right."));
    PADDLE_ENFORCE(attributes.count("channels") > 0 &&
                       attributes.at("channels").isa<pir::ArrayAttribute>(),
                   phi::errors::PreconditionNotMet(
                       "Type of attribute: channels is not right."));
    for (size_t i = 0;
         i < attributes.at("channels").dyn_cast<pir::ArrayAttribute>().size();
         i++) {
      PADDLE_ENFORCE(attributes.at("channels")
                         .dyn_cast<pir::ArrayAttribute>()
                         .at(i)
                         .isa<pir::Int32Attribute>(),
                     phi::errors::PreconditionNotMet(
                         "Type of attribute: channels is not right."));
    }
    PADDLE_ENFORCE(
        attributes.count("user_workspace_size") > 0 &&
            attributes.at("user_workspace_size").isa<pir::Int32Attribute>(),
        phi::errors::PreconditionNotMet(
            "Type of attribute: user_workspace_size is not right."));
  }
  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    PADDLE_ENFORCE_EQ(
        output_size,
        2u,
        phi::errors::PreconditionNotMet(
            "The size %d of outputs must be equal to 2.", output_size));
    PADDLE_ENFORCE(
        (*this)->result(0).type().isa<paddle::dialect::DenseTensorType>(),
        phi::errors::PreconditionNotMet(
            "Type validation failed for the 0th output."));
    auto output_1_type = (*this)->result(1).type();
    if (auto vec_type = output_1_type.dyn_cast<pir::VectorType>()) {
      for (size_t i = 0; i < vec_type.size(); i++) {
        PADDLE_ENFORCE(vec_type[i].isa<paddle::dialect::DenseTensorType>(),
                       phi::errors::PreconditionNotMet(
                           "Type validation failed for the 1th output."));
      }
    } else {
      PADDLE_ENFORCE(output_1_type.isa<paddle::dialect::DenseTensorType>(),
                     phi::errors::PreconditionNotMet(
                         "Type validation failed for the 1th output."));
    }
  }
  VLOG(4) << "End Verifying for: Conv2dFusionOp.";
}

void Conv2dFusionOpTest::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::FusedConvInferMeta);
  fn(infer_meta);
}
}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::Conv2dFusionOpTest)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::Conv2dFusionOpTest)

class Conv2dFusionTestDialect : public pir::Dialect {
 public:
  explicit Conv2dFusionTestDialect(pir::IrContext *context)
      : pir::Dialect(name(), context, pir::TypeId::get<TestDialect>()) {
    initialize();
  }
  static const char *name() { return "con2d fusion test"; }

 private:
  void initialize() { RegisterOps<paddle::dialect::Conv2dFusionOpTest>(); }
};
IR_DECLARE_EXPLICIT_TYPE_ID(Conv2dFusionTestDialect)
IR_DEFINE_EXPLICIT_TYPE_ID(Conv2dFusionTestDialect)

class Conv2dAddFusePattern
    : public pir::OpRewritePattern<paddle::dialect::AddOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::AddOp>::OpRewritePattern;
  bool MatchAndRewrite(
      paddle::dialect::AddOp op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT
    // The next op should be add.
    paddle::dialect::Conv2dOp conv2d_op =
        pir::GetDefiningOpForInput(op, 0)
            ->dyn_cast<paddle::dialect::Conv2dOp>();
    if (!conv2d_op) return false;

    pir::OpResult conv2d_out = conv2d_op.out();
    if (!conv2d_out.HasOneUse()) return false;

    pir::Value conv2d_filter = conv2d_op.filter();

    pir::OpResult conv2d_filter_result =
        conv2d_filter.dyn_cast<pir::OpResult>();
    IR_ENFORCE(conv2d_filter_result);

    pir::Value add_input = op.x();
    IR_ENFORCE(add_input == conv2d_out);

    pir::Value y = op.y();
    pir::OpResult bias = y.dyn_cast<pir::OpResult>();
    auto conv2d_attributes = conv2d_op.attributes();
    std::vector<std::string> conv2d_fusion_attrStr = {"strides",
                                                      "paddings_t",
                                                      "padding_algorithm",
                                                      "dilations_t",
                                                      "groups",
                                                      "data_format",
                                                      "activation",
                                                      "exhaustive_search",
                                                      "channels",
                                                      "user_workspace_size"};
    std::vector<pir::Attribute> con2d_fusing_attr = {
        conv2d_attributes.at("strides"),
        conv2d_attributes.at("paddings"),
        conv2d_attributes.at("padding_algorithm"),
        conv2d_attributes.at("dilations"),
        conv2d_attributes.at("groups"),
        conv2d_attributes.at("data_format"),
        rewriter.str_attr("identity"),
        rewriter.bool_attr(true),
        rewriter.array_attr(std::vector<pir::Attribute>{}),
        rewriter.int32_attr(0)};
    pir::AttributeMap conv2d_fusion_attributes;
    for (size_t i = 0; i < conv2d_fusion_attrStr.size(); ++i) {
      conv2d_fusion_attributes[conv2d_fusion_attrStr[i]] = con2d_fusing_attr[i];
    }

    pir::OpResult tmpResidual;

    auto conv2d_fuse_op = rewriter.Build<paddle::dialect::Conv2dFusionOpTest>(
        pir::GetDefiningOpForInput(conv2d_op, 0)->result(0),
        conv2d_filter_result,
        bias,
        tmpResidual,
        conv2d_fusion_attributes);
    rewriter.ReplaceOp(op, std::vector<pir::Value>{conv2d_fuse_op.output()});
    return true;
  }
};

class TestPass : public pir::PatternRewritePass {
 public:
  TestPass() : pir::PatternRewritePass("test_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<RedundantTransposeFusePattern>(context);
    return ps;
  }
};

void BuildProgram(pir::Builder &builder) {  // NOLINT
  paddle::dialect::FullOp full_input_op =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{4, 3, 16, 16},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  paddle::dialect::FullOp full_filter_op =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64, 3, 3, 3},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  paddle::dialect::FullOp full_mean_op = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{64}, 1.5, phi::DataType::FLOAT32, phi::CPUPlace());

  paddle::dialect::FullOp full_variance_op =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  paddle::dialect::FullOp full_scale_op =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  paddle::dialect::FullOp full_bias_op = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{64}, 1.5, phi::DataType::FLOAT32, phi::CPUPlace());

  paddle::dialect::Conv2dOp conv2d_op =
      builder.Build<paddle::dialect::Conv2dOp>(full_input_op.out(),
                                               full_filter_op.out());

  paddle::dialect::BatchNorm_Op batch_norm_op =
      builder.Build<paddle::dialect::BatchNorm_Op>(conv2d_op.out(),
                                                   full_mean_op.out(),
                                                   full_variance_op.out(),
                                                   full_scale_op.out(),
                                                   full_bias_op.out(),
                                                   true,
                                                   0.9,
                                                   1e-6,
                                                   "NCHW",
                                                   false,
                                                   false);

  auto transpose1_op = builder.Build<paddle::dialect::TransposeOp>(
      batch_norm_op.out(), std::vector<int>{0, 2, 3, 1});

  auto transpose2_op = builder.Build<paddle::dialect::TransposeOp>(
      transpose1_op.out(), std::vector<int>{0, 3, 1, 2});

  builder.Build<paddle::dialect::FetchOp>(transpose2_op.out(), "out", 0);
}

TEST(pattern_rewrite, Patterns) {
  pir::IrContext *ctx = pir::IrContext::Instance();

  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  auto *test_dialect = ctx->GetOrRegisterDialect<Conv2dFusionTestDialect>();
  test_dialect->RegisterOp<paddle::dialect::Conv2dFusionOpTest>();

  pir::Program program(ctx);
  pir::Builder builder = pir::Builder(ctx, program.block());
  BuildProgram(builder);

  EXPECT_EQ(program.block()->size(), 11u);
  paddle::framework::Scope scope;
  pir::PassManager pm(ctx);
  pm.AddPass(std::make_unique<TestPass>());
<<<<<<< Updated upstream
  pm.AddPass(pir::CreateConv2dFusePass());
=======
  pm.AddPass(pir::CreateFcFusePass());
>>>>>>> Stashed changes
  pm.AddPass(pir::CreateConstantFoldingPass(&scope));
  pm.AddPass(pir::CreateDeadCodeEliminationPass());
  pm.EnablePassTiming();
  pm.EnableIRPrinting();
  //   pm.EnableIRPrinting(std::make_unique<pir::PassManager::IRPrinterOption>(
  //       [](pir::Pass *pass, pir::Operation *op) {
  //         return pass->name() == "constant_folding_pass";
  //       },
  //       [](pir::Pass *pass, pir::Operation *op) {
  //         return pass->name() == "constant_folding_pass";
  //       },
  //       true,
  //       true));

  CHECK_EQ(pm.Run(&program), true);
  EXPECT_EQ(program.block()->size(), 2u);
}

void BuildConstantFoldingProgram(pir::Program *program,
                                 pir::IrContext *ctx,
                                 paddle::framework::Scope *scope) {
  pir::Builder builder = pir::Builder(ctx, program->block());

  pir::Type fp32_dtype = pir::Float32Type::get(ctx);
  phi::DDim dims = {2, 2};
  phi::DataLayout data_layout = phi::DataLayout::NCHW;
  phi::LoD lod = {{0, 1, 2}};
  size_t offset = 0;
  pir::Type dense_tensor_dtype = paddle::dialect::DenseTensorType::get(
      ctx, fp32_dtype, dims, data_layout, lod, offset);

  phi::DenseTensorMeta meta(
      phi::DataType::FLOAT32, dims, data_layout, lod, offset);
  paddle::platform::DeviceContext *dev_ctx =
      paddle::platform::DeviceContextPool::Instance().Get(
          paddle::platform::CPUPlace());

  auto op1 = builder.Build<pir::ParameterOp>("a", dense_tensor_dtype);
  auto op2 = builder.Build<pir::ParameterOp>("b", dense_tensor_dtype);

  auto op3 =
      builder.Build<paddle::dialect::AddOp>(op1->result(0), op2->result(0));

  auto op4 = builder.Build<pir::ParameterOp>("c", dense_tensor_dtype);

  auto op5 =
      builder.Build<paddle::dialect::AddOp>(op3->result(0), op4->result(0));
  builder.Build<paddle::dialect::FetchOp>(op5.out(), "out", 0);

  auto *tensor_a = scope->Var("a")->GetMutable<phi::DenseTensor>();
  auto *tensor_b = scope->Var("b")->GetMutable<phi::DenseTensor>();
  auto *tensor_c = scope->Var("c")->GetMutable<phi::DenseTensor>();

  tensor_a->set_meta(meta);
  tensor_b->set_meta(meta);
  tensor_c->set_meta(meta);

  dev_ctx->Alloc(tensor_a, phi::DataType::FLOAT32);
  dev_ctx->Alloc(tensor_b, phi::DataType::FLOAT32);
  dev_ctx->Alloc(tensor_c, phi::DataType::FLOAT32);
}

TEST(constant_folding, ConstantFolding) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();

  pir::Program program(ctx);
  paddle::framework::Scope scope;
  BuildConstantFoldingProgram(&program, ctx, &scope);

  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateConstantFoldingPass(&scope));
  pm.AddPass(pir::CreateDeadCodeEliminationPass());
  pm.EnableIRPrinting();

  CHECK_EQ(pm.Run(&program), true);
  EXPECT_EQ(program.block()->size(), 2u);
}

void BuildConcatProgram(pir::Program *program, pir::IrContext *ctx) {
  pir::Builder builder = pir::Builder(ctx, program->block());
  auto x = builder
               .Build<paddle::dialect::FullOp>(std::vector<int64_t>({16, 16}),
                                               2.0,
                                               phi::DataType::FLOAT32,
                                               phi::GPUPlace())
               .result(0);

  auto y = builder
               .Build<paddle::dialect::FullOp>(std::vector<int64_t>({16, 16}),
                                               2.0,
                                               phi::DataType::FLOAT32,
                                               phi::GPUPlace())
               .result(0);

  auto t1 =
      builder.Build<pir::CombineOp>(std::vector<pir::Value>({x, y})).result(0);

  auto out1 = builder.Build<paddle::dialect::ConcatOp>(t1, 1).result(0);

  auto z = builder
               .Build<paddle::dialect::FullOp>(std::vector<int64_t>({16, 16}),
                                               2.0,
                                               phi::DataType::FLOAT32,
                                               phi::GPUPlace())
               .result(0);

  auto w = builder
               .Build<paddle::dialect::FullOp>(std::vector<int64_t>({16, 16}),
                                               2.0,
                                               phi::DataType::FLOAT32,
                                               phi::GPUPlace())
               .result(0);

  auto t2 =
      builder.Build<pir::CombineOp>(std::vector<pir::Value>({z, w})).result(0);

  auto out2 = builder.Build<paddle::dialect::ConcatOp>(t2, 1).result(0);

  auto out = builder.Build<paddle::dialect::AddOp>(out1, out2).result(0);

  builder.Build<paddle::dialect::FetchOp>(out, "out", 0);
}

TEST(constant_folding, ConstantFolding_Combine) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();

  pir::Program program(ctx);
  paddle::framework::Scope scope;
  BuildConcatProgram(&program, ctx);

  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateConstantFoldingPass(&scope));
  pm.AddPass(pir::CreateDeadCodeEliminationPass());
  pm.EnableIRPrinting();

  CHECK_EQ(pm.Run(&program), true);
  // EXPECT_EQ(program.block()->size(), 6u);
}
