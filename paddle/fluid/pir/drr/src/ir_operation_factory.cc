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

#include <any>

#include "paddle/common/layout.h"

#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#ifdef PADDLE_WITH_DNNL
#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#endif
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_context.h"
#include "paddle/fluid/pir/drr/src/attr_type_uilts.h"
#include "paddle/fluid/pir/drr/src/ir_operation_factory.h"

#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/value.h"

#include "paddle/phi/core/enforce.h"

namespace paddle::drr {

void OperationFactory::RegisterManualOpCreator() {
  RegisterOperationCreator(
      "pd_op.fused_gemm_epilogue",
      [](const std::vector<pir::Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        return rewriter.Build<paddle::dialect::FusedGemmEpilogueOp>(
            inputs[0], inputs[1], inputs[2], attrs);
      });
  RegisterOperationCreator(
      "pd_op.fused_gemm_epilogue_grad",
      [](const std::vector<pir::Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        return rewriter.Build<paddle::dialect::FusedGemmEpilogueGradOp>(
            inputs[0], inputs[1], inputs[2], inputs[3], attrs);
      });
  RegisterOperationCreator("builtin.combine",
                           [](const std::vector<pir::Value>& inputs,
                              const pir::AttributeMap& attrs,
                              pir::PatternRewriter& rewriter) {
                             return rewriter.Build<pir::CombineOp>(inputs);
                           });
  RegisterOperationCreator("builtin.split",
                           [](const std::vector<pir::Value>& inputs,
                              const pir::AttributeMap& attrs,
                              pir::PatternRewriter& rewriter) {
                             return rewriter.Build<pir::SplitOp>(inputs[0]);
                           });

  RegisterOperationCreator(
      "builtin.slice",
      [](const std::vector<pir::Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        return rewriter.Build<pir::SliceOp>(
            inputs[0],
            attrs.at("index").dyn_cast<pir::Int32Attribute>().data());
      });
  RegisterOperationCreator(
      "pd_op.scale",
      [](const std::vector<pir::Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        if (inputs.size() == 2) {
          return rewriter.Build<paddle::dialect::ScaleOp>(
              inputs[0],
              inputs[1],
              attrs.at("bias").dyn_cast<pir::FloatAttribute>().data(),
              attrs.at("bias_after_scale")
                  .dyn_cast<pir::BoolAttribute>()
                  .data());
        }
        return rewriter.Build<paddle::dialect::ScaleOp>(inputs[0], attrs);
      });
  RegisterOperationCreator(
      "pd_op.slice",
      [](const std::vector<pir::Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        if (inputs.size() == 3) {
          PADDLE_ENFORCE_NE(attrs.find("axes"),
                            attrs.end(),
                            common::errors::InvalidArgument(
                                "'axes' Attribute is expected for SliceOp. "));
          std::vector<int64_t> axes;
          for (size_t i = 0;
               i < attrs.at("axes").dyn_cast<pir::ArrayAttribute>().size();
               i++) {
            axes.push_back(attrs.at("axes")
                               .dyn_cast<pir::ArrayAttribute>()
                               .at(i)
                               .dyn_cast<pir::Int64Attribute>()
                               .data());
          }

          PADDLE_ENFORCE_NE(
              attrs.find("infer_flags"),
              attrs.end(),
              common::errors::InvalidArgument(
                  "'infer_flags' Attribute is expected for SliceOp. "));
          std::vector<int64_t> infer_flags;
          for (size_t i = 0;
               i <
               attrs.at("infer_flags").dyn_cast<pir::ArrayAttribute>().size();
               i++) {
            infer_flags.push_back(attrs.at("infer_flags")
                                      .dyn_cast<pir::ArrayAttribute>()
                                      .at(i)
                                      .dyn_cast<pir::Int64Attribute>()
                                      .data());
          }

          PADDLE_ENFORCE_NE(
              attrs.find("decrease_axis"),
              attrs.end(),
              common::errors::InvalidArgument(
                  "'decrease_axis' Attribute is expected for SliceOp. "));
          std::vector<int64_t> decrease_axis;
          for (size_t i = 0;
               i <
               attrs.at("decrease_axis").dyn_cast<pir::ArrayAttribute>().size();
               i++) {
            decrease_axis.push_back(attrs.at("decrease_axis")
                                        .dyn_cast<pir::ArrayAttribute>()
                                        .at(i)
                                        .dyn_cast<pir::Int64Attribute>()
                                        .data());
          }
          return rewriter.Build<paddle::dialect::SliceOp>(inputs[0],
                                                          inputs[1],
                                                          inputs[2],
                                                          axes,
                                                          infer_flags,
                                                          decrease_axis);
        }
        return rewriter.Build<paddle::dialect::SliceOp>(inputs[0], attrs);
      });
#ifdef PADDLE_WITH_DNNL
  RegisterOperationCreator(
      "onednn_op.conv2d_transpose_bias",
      [](const std::vector<pir::Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        if (inputs.size() == 4) {
          PADDLE_ENFORCE_EQ(
              attrs.find("strides") != attrs.end(),
              true,
              common::errors::InvalidArgument("'strides' Attribute is expected "
                                              "for Conv2dTransposeBiasOp. "));
          std::vector<int> strides;
          for (size_t i = 0;
               i < attrs.at("strides").dyn_cast<pir::ArrayAttribute>().size();
               i++) {
            strides.push_back(attrs.at("strides")
                                  .dyn_cast<pir::ArrayAttribute>()
                                  .at(i)
                                  .dyn_cast<pir::Int32Attribute>()
                                  .data());
          }

          PADDLE_ENFORCE_EQ(attrs.find("paddings") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'paddings' Attribute is expected "
                                "for Conv2dTransposeBiasOp. "));
          std::vector<int> paddings;
          for (size_t i = 0;
               i < attrs.at("paddings").dyn_cast<pir::ArrayAttribute>().size();
               i++) {
            paddings.push_back(attrs.at("paddings")
                                   .dyn_cast<pir::ArrayAttribute>()
                                   .at(i)
                                   .dyn_cast<pir::Int32Attribute>()
                                   .data());
          }

          PADDLE_ENFORCE_EQ(attrs.find("output_padding") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'output_padding' Attribute is expected for "
                                "Conv2dTransposeBiasOp. "));
          std::vector<int> output_padding;
          for (size_t i = 0; i < attrs.at("output_padding")
                                     .dyn_cast<pir::ArrayAttribute>()
                                     .size();
               i++) {
            output_padding.push_back(attrs.at("output_padding")
                                         .dyn_cast<pir::ArrayAttribute>()
                                         .at(i)
                                         .dyn_cast<pir::Int32Attribute>()
                                         .data());
          }

          PADDLE_ENFORCE_EQ(attrs.find("padding_algorithm") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'padding_algorithm' Attribute is expected for "
                                "Conv2dTransposeBiasOp. "));
          std::string padding_algorithm = attrs.at("padding_algorithm")
                                              .dyn_cast<pir::StrAttribute>()
                                              .AsString();

          PADDLE_ENFORCE_EQ(attrs.find("groups") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'groups' Attribute is expected for "
                                "Conv2dTransposeBiasOp. "));
          int groups =
              attrs.at("groups").dyn_cast<pir::Int32Attribute>().data();

          PADDLE_ENFORCE_EQ(attrs.find("dilations") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'dilations' Attribute is expected "
                                "for Conv2dTransposeBiasOp. "));
          std::vector<int> dilations;
          for (size_t i = 0;
               i < attrs.at("dilations").dyn_cast<pir::ArrayAttribute>().size();
               i++) {
            dilations.push_back(attrs.at("dilations")
                                    .dyn_cast<pir::ArrayAttribute>()
                                    .at(i)
                                    .dyn_cast<pir::Int32Attribute>()
                                    .data());
          }

          PADDLE_ENFORCE_EQ(attrs.find("data_format") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'data_format' Attribute is expected for "
                                "Conv2dTransposeBiasOp. "));
          std::string data_format =
              attrs.at("data_format").dyn_cast<pir::StrAttribute>().AsString();

          PADDLE_ENFORCE_EQ(
              attrs.find("is_test") != attrs.end(),
              true,
              common::errors::InvalidArgument("'is_test' Attribute is expected "
                                              "for Conv2dTransposeBiasOp. "));
          bool is_test =
              attrs.at("is_test").dyn_cast<pir::BoolAttribute>().data();

          PADDLE_ENFORCE_EQ(attrs.find("force_fp32_output") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'force_fp32_output' Attribute is expected "
                                "for Conv2dTransposeBiasOp. "));
          bool force_fp32_output = attrs.at("force_fp32_output")
                                       .dyn_cast<pir::BoolAttribute>()
                                       .data();

          PADDLE_ENFORCE_EQ(attrs.find("mkldnn_data_type") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'mkldnn_data_type' Attribute is expected "
                                "for Conv2dTransposeBiasOp. "));
          std::string mkldnn_data_type = attrs.at("mkldnn_data_type")
                                             .dyn_cast<pir::StrAttribute>()
                                             .AsString();

          PADDLE_ENFORCE_EQ(attrs.find("fuse_relu") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'fuse_relu' Attribute is expected "
                                "for Conv2dTransposeBiasOp. "));
          bool fuse_relu =
              attrs.at("fuse_relu").dyn_cast<pir::BoolAttribute>().data();

          PADDLE_ENFORCE_EQ(attrs.find("fuse_activation") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'fuse_activation' Attribute is expected "
                                "for Conv2dTransposeBiasOp. "));
          std::string fuse_activation = attrs.at("fuse_activation")
                                            .dyn_cast<pir::StrAttribute>()
                                            .AsString();

          PADDLE_ENFORCE_EQ(attrs.find("fuse_alpha") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'fuse_alpha' Attribute is expected "
                                "for Conv2dTransposeBiasOp. "));
          bool fuse_alpha =
              attrs.at("fuse_alpha").dyn_cast<pir::FloatAttribute>().data();

          PADDLE_ENFORCE_EQ(attrs.find("fuse_beta") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'fuse_beta' Attribute is expected "
                                "for Conv2dTransposeBiasOp. "));
          bool fuse_beta =
              attrs.at("fuse_beta").dyn_cast<pir::FloatAttribute>().data();

          return rewriter.Build<paddle::onednn::dialect::Conv2dTransposeBiasOp>(
              inputs[0],
              inputs[1],
              inputs[2],
              inputs[3],
              strides,
              paddings,
              output_padding,
              padding_algorithm,
              groups,
              dilations,
              data_format,
              is_test,
              force_fp32_output,
              mkldnn_data_type,
              fuse_relu,
              fuse_activation,
              fuse_alpha,
              fuse_beta);
        }

        return rewriter.Build<paddle::onednn::dialect::Conv2dTransposeBiasOp>(
            inputs[0], inputs[1], inputs[2], attrs);
      });
  RegisterOperationCreator(
      "onednn_op.reshape",
      [](const std::vector<pir::Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        if (inputs.size() == 2) {
          PADDLE_ENFORCE_EQ(attrs.find("mkldnn_data_type") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'mkldnn_data_type' Attribute is expected "
                                "for ReshapeOp. "));
          std::string mkldnn_data_type = attrs.at("mkldnn_data_type")
                                             .dyn_cast<pir::StrAttribute>()
                                             .AsString();
          PADDLE_ENFORCE_EQ(attrs.find("use_quantizer") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'use_quantizer' Attribute is expected "
                                "for ReshapeOp. "));
          bool use_quantizer =
              attrs.at("use_quantizer").dyn_cast<pir::BoolAttribute>().data();

          return rewriter.Build<paddle::onednn::dialect::ReshapeOp>(
              inputs[0], inputs[1], mkldnn_data_type, use_quantizer);
        }
        return rewriter.Build<paddle::onednn::dialect::ReshapeOp>(inputs[0],
                                                                  attrs);
      });

  RegisterOperationCreator(
      "onednn_op.reshape_",
      [](const std::vector<pir::Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        if (inputs.size() == 2) {
          PADDLE_ENFORCE_EQ(attrs.find("mkldnn_data_type") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'mkldnn_data_type' Attribute is expected "
                                "for Reshape_Op. "));
          std::string mkldnn_data_type = attrs.at("mkldnn_data_type")
                                             .dyn_cast<pir::StrAttribute>()
                                             .AsString();
          PADDLE_ENFORCE_EQ(attrs.find("use_quantizer") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'use_quantizer' Attribute is expected "
                                "for Reshape_Op. "));
          bool use_quantizer =
              attrs.at("use_quantizer").dyn_cast<pir::BoolAttribute>().data();

          return rewriter.Build<paddle::onednn::dialect::Reshape_Op>(
              inputs[0], inputs[1], mkldnn_data_type, use_quantizer);
        }
        return rewriter.Build<paddle::onednn::dialect::Reshape_Op>(inputs[0],
                                                                   attrs);
      });

  RegisterOperationCreator(
      "onednn_op.pool2d",
      [](const std::vector<pir::Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        if (inputs.size() == 2) {
          PADDLE_ENFORCE_EQ(attrs.find("strides") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'strides' Attribute is expected for "
                                "Pool2dOp. "));
          std::vector<int> strides;
          for (size_t i = 0;
               i < attrs.at("strides").dyn_cast<pir::ArrayAttribute>().size();
               i++) {
            strides.push_back(attrs.at("strides")
                                  .dyn_cast<pir::ArrayAttribute>()
                                  .at(i)
                                  .dyn_cast<pir::Int32Attribute>()
                                  .data());
          }

          PADDLE_ENFORCE_EQ(attrs.find("paddings") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'paddings' Attribute is expected for "
                                "Pool2dOp. "));
          std::vector<int> paddings;
          for (size_t i = 0;
               i < attrs.at("paddings").dyn_cast<pir::ArrayAttribute>().size();
               i++) {
            paddings.push_back(attrs.at("paddings")
                                   .dyn_cast<pir::ArrayAttribute>()
                                   .at(i)
                                   .dyn_cast<pir::Int32Attribute>()
                                   .data());
          }

          PADDLE_ENFORCE_EQ(attrs.find("ceil_mode") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'ceil_mode' Attribute is expected "
                                "for Pool2dOp. "));
          bool ceil_mode =
              attrs.at("ceil_mode").dyn_cast<pir::BoolAttribute>().data();

          PADDLE_ENFORCE_EQ(attrs.find("exclusive") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'exclusive' Attribute is expected "
                                "for Pool2dOp. "));
          bool exclusive =
              attrs.at("exclusive").dyn_cast<pir::BoolAttribute>().data();

          PADDLE_ENFORCE_EQ(attrs.find("data_format") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'data_format' Attribute is expected "
                                "for Pool2dOp. "));
          std::string data_format =
              attrs.at("data_format").dyn_cast<pir::StrAttribute>().AsString();

          PADDLE_ENFORCE_EQ(attrs.find("pooling_type") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'pooling_type' Attribute is expected "
                                "for Pool2dOp. "));
          std::string pooling_type =
              attrs.at("pooling_type").dyn_cast<pir::StrAttribute>().AsString();

          PADDLE_ENFORCE_EQ(attrs.find("global_pooling") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'global_pooling' Attribute is expected "
                                "for Pool2dOp. "));
          bool global_pooling =
              attrs.at("global_pooling").dyn_cast<pir::BoolAttribute>().data();

          PADDLE_ENFORCE_EQ(attrs.find("adaptive") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'adaptive' Attribute is expected "
                                "for Pool2dOp. "));
          bool adaptive =
              attrs.at("adaptive").dyn_cast<pir::BoolAttribute>().data();

          PADDLE_ENFORCE_EQ(attrs.find("padding_algorithm") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'padding_algorithm' Attribute is expected "
                                "for Pool2dOp. "));
          std::string padding_algorithm = attrs.at("padding_algorithm")
                                              .dyn_cast<pir::StrAttribute>()
                                              .AsString();

          PADDLE_ENFORCE_EQ(attrs.find("mkldnn_data_type") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'mkldnn_data_type' Attribute is expected "
                                "for Pool2dOp. "));
          std::string mkldnn_data_type = attrs.at("mkldnn_data_type")
                                             .dyn_cast<pir::StrAttribute>()
                                             .AsString();

          PADDLE_ENFORCE_EQ(attrs.find("use_quantizer") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'use_quantizer' Attribute is expected "
                                "for Pool2dOp. "));
          bool use_quantizer =
              attrs.at("use_quantizer").dyn_cast<pir::BoolAttribute>().data();

          PADDLE_ENFORCE_EQ(
              attrs.find("is_test") != attrs.end(),
              true,
              common::errors::InvalidArgument("'is_test' Attribute is expected "
                                              "for Pool2dOp. "));
          bool is_test =
              attrs.at("is_test").dyn_cast<pir::BoolAttribute>().data();

          return rewriter.Build<paddle::onednn::dialect::Pool2dOp>(
              inputs[0],
              inputs[1],
              strides,
              paddings,
              ceil_mode,
              exclusive,
              data_format,
              pooling_type,
              global_pooling,
              adaptive,
              padding_algorithm,
              use_quantizer,
              mkldnn_data_type,
              is_test);
        }
        return rewriter.Build<paddle::onednn::dialect::Pool2dOp>(inputs[0],
                                                                 attrs);
      });

  RegisterOperationCreator(
      "onednn_op.sum",
      [](const std::vector<pir::Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        if (inputs.size() == 2) {
          // Add after mkldnn_data_type add in sum op
          PADDLE_ENFORCE_EQ(attrs.find("mkldnn_data_type") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'mkldnn_data_type' Attribute is expected "
                                "for SumOp. "));
          std::string mkldnn_data_type = attrs.at("mkldnn_data_type")
                                             .dyn_cast<pir::StrAttribute>()
                                             .AsString();
          PADDLE_ENFORCE_EQ(
              attrs.find("keepdim") != attrs.end(),
              true,
              common::errors::InvalidArgument("'keepdim' Attribute is expected "
                                              "for SumOp. "));
          bool keepdim =
              attrs.at("keepdim").dyn_cast<pir::BoolAttribute>().data();

          PADDLE_ENFORCE_EQ(
              attrs.find("dtype") != attrs.end(),
              true,
              common::errors::InvalidArgument("'dtype' Attribute is expected "
                                              "for SumOp. "));

          ::phi::DataType dtype =
              attrs.at("dtype")
                  .dyn_cast<paddle::dialect::DataTypeAttribute>()
                  .data();

          return rewriter.Build<paddle::onednn::dialect::SumOp>(
              inputs[0], inputs[1], dtype, keepdim, mkldnn_data_type);
        }
        return rewriter.Build<paddle::onednn::dialect::SumOp>(inputs[0], attrs);
      });

  RegisterOperationCreator(
      "onednn_op.slice",
      [](const std::vector<pir::Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        if (inputs.size() == 3) {
          PADDLE_ENFORCE_EQ(attrs.find("mkldnn_data_type") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'mkldnn_data_type' Attribute is expected "
                                "for SliceOp. "));
          std::string mkldnn_data_type = attrs.at("mkldnn_data_type")
                                             .dyn_cast<pir::StrAttribute>()
                                             .AsString();

          PADDLE_ENFORCE_EQ(attrs.find("decrease_axis") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'paddings' Attribute is expected for "
                                "SliceOp. "));
          std::vector<int64_t> decrease_axis;
          for (size_t i = 0;
               i <
               attrs.at("decrease_axis").dyn_cast<pir::ArrayAttribute>().size();
               i++) {
            decrease_axis.push_back(attrs.at("decrease_axis")
                                        .dyn_cast<pir::ArrayAttribute>()
                                        .at(i)
                                        .dyn_cast<pir::Int64Attribute>()
                                        .data());
          }

          PADDLE_ENFORCE_EQ(attrs.find("infer_flags") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'infer_flags' Attribute is expected for "
                                "SliceOp. "));
          std::vector<int64_t> infer_flags;
          for (size_t i = 0;
               i <
               attrs.at("infer_flags").dyn_cast<pir::ArrayAttribute>().size();
               i++) {
            infer_flags.push_back(attrs.at("infer_flags")
                                      .dyn_cast<pir::ArrayAttribute>()
                                      .at(i)
                                      .dyn_cast<pir::Int64Attribute>()
                                      .data());
          }

          PADDLE_ENFORCE_EQ(attrs.find("axes") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'axes' Attribute is expected for "
                                "SliceOp. "));
          std::vector<int64_t> axes;
          for (size_t i = 0;
               i < attrs.at("axes").dyn_cast<pir::ArrayAttribute>().size();
               i++) {
            axes.push_back(attrs.at("axes")
                               .dyn_cast<pir::ArrayAttribute>()
                               .at(i)
                               .dyn_cast<pir::Int64Attribute>()
                               .data());
          }

          return rewriter.Build<paddle::onednn::dialect::SliceOp>(
              inputs[0],
              inputs[1],
              inputs[2],
              axes,
              infer_flags,
              decrease_axis,
              mkldnn_data_type);
        }
        return rewriter.Build<paddle::onednn::dialect::SliceOp>(inputs[0],
                                                                attrs);
      });

  RegisterOperationCreator(
      "onednn_op.squeeze",
      [](const std::vector<pir::Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        if (inputs.size() == 2) {
          PADDLE_ENFORCE_EQ(attrs.find("mkldnn_data_type") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'mkldnn_data_type' Attribute is expected "
                                "for SqueezeOp. "));
          std::string mkldnn_data_type = attrs.at("mkldnn_data_type")
                                             .dyn_cast<pir::StrAttribute>()
                                             .AsString();

          return rewriter.Build<paddle::onednn::dialect::SqueezeOp>(
              inputs[0], inputs[1], mkldnn_data_type);
        }
        return rewriter.Build<paddle::onednn::dialect::SqueezeOp>(inputs[0],
                                                                  attrs);
      });

  RegisterOperationCreator(
      "onednn_op.squeeze_",
      [](const std::vector<pir::Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        if (inputs.size() == 2) {
          PADDLE_ENFORCE_EQ(attrs.find("mkldnn_data_type") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'mkldnn_data_type' Attribute is expected "
                                "for Squeeze_Op. "));
          std::string mkldnn_data_type = attrs.at("mkldnn_data_type")
                                             .dyn_cast<pir::StrAttribute>()
                                             .AsString();

          return rewriter.Build<paddle::onednn::dialect::Squeeze_Op>(
              inputs[0], inputs[1], mkldnn_data_type);
        }
        return rewriter.Build<paddle::onednn::dialect::Squeeze_Op>(inputs[0],
                                                                   attrs);
      });

  RegisterOperationCreator(
      "onednn_op.clip",
      [](const std::vector<pir::Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        if (inputs.size() == 3) {
          PADDLE_ENFORCE_EQ(attrs.find("mkldnn_data_type") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'mkldnn_data_type' Attribute is expected "
                                "for ClipOp. "));
          std::string mkldnn_data_type = attrs.at("mkldnn_data_type")
                                             .dyn_cast<pir::StrAttribute>()
                                             .AsString();

          return rewriter.Build<paddle::onednn::dialect::ClipOp>(
              inputs[0], inputs[1], inputs[2], mkldnn_data_type);
        }
        return rewriter.Build<paddle::onednn::dialect::ClipOp>(inputs[0],
                                                               attrs);
      });

  RegisterOperationCreator(
      "onednn_op.clip_",
      [](const std::vector<pir::Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        if (inputs.size() == 3) {
          PADDLE_ENFORCE_EQ(attrs.find("mkldnn_data_type") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'mkldnn_data_type' Attribute is expected "
                                "for Clip_Op. "));
          std::string mkldnn_data_type = attrs.at("mkldnn_data_type")
                                             .dyn_cast<pir::StrAttribute>()
                                             .AsString();

          return rewriter.Build<paddle::onednn::dialect::Clip_Op>(
              inputs[0], inputs[1], inputs[2], mkldnn_data_type);
        }
        return rewriter.Build<paddle::onednn::dialect::Clip_Op>(inputs[0],
                                                                attrs);
      });

  RegisterOperationCreator(
      "onednn_op.scale",
      [](const std::vector<pir::Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        if (inputs.size() == 2) {
          // Add after scale add this attr
          PADDLE_ENFORCE_EQ(attrs.find("mkldnn_data_type") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'mkldnn_data_type' Attribute is expected "
                                "for ScaleOp. "));
          std::string mkldnn_data_type = attrs.at("mkldnn_data_type")
                                             .dyn_cast<pir::StrAttribute>()
                                             .AsString();
          PADDLE_ENFORCE_EQ(attrs.find("bias_after_scale") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'bias_after_scale' Attribute is expected "
                                "for ScaleOp. "));
          bool bias_after_scale = attrs.at("bias_after_scale")
                                      .dyn_cast<pir::BoolAttribute>()
                                      .data();

          PADDLE_ENFORCE_EQ(
              attrs.find("bias") != attrs.end(),
              true,
              common::errors::InvalidArgument("'bias' Attribute is expected "
                                              "for ScaleOp. "));
          bool bias = attrs.at("bias").dyn_cast<pir::FloatAttribute>().data();

          return rewriter.Build<paddle::onednn::dialect::ScaleOp>(
              inputs[0], inputs[1], bias, bias_after_scale, mkldnn_data_type);
        }
        return rewriter.Build<paddle::onednn::dialect::ScaleOp>(inputs[0],
                                                                attrs);
      });

  RegisterOperationCreator(
      "onednn_op.conv2d_transpose",
      [](const std::vector<pir::Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        if (inputs.size() == 3) {
          PADDLE_ENFORCE_EQ(
              attrs.find("strides") != attrs.end(),
              true,
              common::errors::InvalidArgument("'strides' Attribute is expected "
                                              "for Conv2dTransposeOp. "));
          std::vector<int> strides;
          for (size_t i = 0;
               i < attrs.at("strides").dyn_cast<pir::ArrayAttribute>().size();
               i++) {
            strides.push_back(attrs.at("strides")
                                  .dyn_cast<pir::ArrayAttribute>()
                                  .at(i)
                                  .dyn_cast<pir::Int32Attribute>()
                                  .data());
          }

          PADDLE_ENFORCE_EQ(attrs.find("paddings") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'paddings' Attribute is expected "
                                "for Conv2dTransposeOp. "));
          std::vector<int> paddings;
          for (size_t i = 0;
               i < attrs.at("paddings").dyn_cast<pir::ArrayAttribute>().size();
               i++) {
            paddings.push_back(attrs.at("paddings")
                                   .dyn_cast<pir::ArrayAttribute>()
                                   .at(i)
                                   .dyn_cast<pir::Int32Attribute>()
                                   .data());
          }

          PADDLE_ENFORCE_EQ(attrs.find("output_padding") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'output_padding' Attribute is expected for "
                                "Conv2dTransposeOp. "));
          std::vector<int> output_padding;
          for (size_t i = 0; i < attrs.at("output_padding")
                                     .dyn_cast<pir::ArrayAttribute>()
                                     .size();
               i++) {
            output_padding.push_back(attrs.at("output_padding")
                                         .dyn_cast<pir::ArrayAttribute>()
                                         .at(i)
                                         .dyn_cast<pir::Int32Attribute>()
                                         .data());
          }

          PADDLE_ENFORCE_EQ(attrs.find("padding_algorithm") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'padding_algorithm' Attribute is expected for "
                                "Conv2dTransposeOp. "));
          std::string padding_algorithm = attrs.at("padding_algorithm")
                                              .dyn_cast<pir::StrAttribute>()
                                              .AsString();

          PADDLE_ENFORCE_EQ(attrs.find("groups") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'groups' Attribute is expected for "
                                "Conv2dTransposeOp. "));
          int groups =
              attrs.at("groups").dyn_cast<pir::Int32Attribute>().data();

          PADDLE_ENFORCE_EQ(attrs.find("dilations") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'dilations' Attribute is expected "
                                "for Conv2dTransposeOp. "));
          std::vector<int> dilations;
          for (size_t i = 0;
               i < attrs.at("dilations").dyn_cast<pir::ArrayAttribute>().size();
               i++) {
            dilations.push_back(attrs.at("dilations")
                                    .dyn_cast<pir::ArrayAttribute>()
                                    .at(i)
                                    .dyn_cast<pir::Int32Attribute>()
                                    .data());
          }

          PADDLE_ENFORCE_EQ(attrs.find("data_format") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'data_format' Attribute is expected for "
                                "Conv2dTransposeOp. "));
          std::string data_format =
              attrs.at("data_format").dyn_cast<pir::StrAttribute>().AsString();

          PADDLE_ENFORCE_EQ(attrs.find("mkldnn_data_type") != attrs.end(),
                            true,
                            common::errors::InvalidArgument(
                                "'mkldnn_data_type' Attribute is expected "
                                "for Conv2dTransposeOp. "));
          std::string mkldnn_data_type = attrs.at("mkldnn_data_type")
                                             .dyn_cast<pir::StrAttribute>()
                                             .AsString();

          PADDLE_ENFORCE_EQ(
              attrs.find("is_test") != attrs.end(),
              true,
              common::errors::InvalidArgument("'is_test' Attribute is expected "
                                              "for Conv2dTransposeOp. "));
          bool is_test =
              attrs.at("is_test").dyn_cast<pir::BoolAttribute>().data();

          return rewriter.Build<paddle::onednn::dialect::Conv2dTransposeOp>(
              inputs[0],
              inputs[1],
              inputs[2],
              strides,
              paddings,
              output_padding,
              padding_algorithm,
              groups,
              dilations,
              data_format,
              mkldnn_data_type,
              is_test);
        }

        return rewriter.Build<paddle::onednn::dialect::Conv2dTransposeOp>(
            inputs[0], inputs[1], attrs);
      });

  // below op since attr will replace to default after insert some bf16 op
  // We have to build here to solve
  RegisterOperationCreator(
      "onednn_op.multiply",
      [](const std::vector<pir::Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        return rewriter.Build<paddle::onednn::dialect::MultiplyOp>(
            inputs[0], inputs[1], attrs);
      });

  RegisterOperationCreator(
      "onednn_op.multiply_",
      [](const std::vector<pir::Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        return rewriter.Build<paddle::onednn::dialect::Multiply_Op>(
            inputs[0], inputs[1], attrs);
      });

  RegisterOperationCreator(
      "onednn_op.add",
      [](const std::vector<pir::Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        return rewriter.Build<paddle::onednn::dialect::AddOp>(
            inputs[0], inputs[1], attrs);
      });

  RegisterOperationCreator(
      "onednn_op.add_",
      [](const std::vector<pir::Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        return rewriter.Build<paddle::onednn::dialect::Add_Op>(
            inputs[0], inputs[1], attrs);
      });

  RegisterOperationCreator(
      "onednn_op.sigmoid",
      [](const std::vector<pir::Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        return rewriter.Build<paddle::onednn::dialect::SigmoidOp>(inputs[0],
                                                                  attrs);
      });

  RegisterOperationCreator(
      "onednn_op.sigmoid_",
      [](const std::vector<pir::Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        return rewriter.Build<paddle::onednn::dialect::Sigmoid_Op>(inputs[0],
                                                                   attrs);
      });
  RegisterOperationCreator(
      "onednn_op.relu",
      [](const std::vector<pir::Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        return rewriter.Build<paddle::onednn::dialect::ReluOp>(inputs[0],
                                                               attrs);
      });

  RegisterOperationCreator(
      "onednn_op.relu_",
      [](const std::vector<pir::Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        return rewriter.Build<paddle::onednn::dialect::Relu_Op>(inputs[0],
                                                                attrs);
      });

#endif

  RegisterOperationCreator(
      "pd_op.max",
      [](const std::vector<pir::Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {
        if (inputs.size() == 2) {
          PADDLE_ENFORCE_NE(attrs.find("keepdim"),
                            attrs.end(),
                            common::errors::InvalidArgument(
                                "'keepdim' Attribute is expected for MaxOp. "));
          bool keepdim =
              attrs.at("keepdim").dyn_cast<pir::BoolAttribute>().data();
          return rewriter.Build<paddle::dialect::MaxOp>(
              inputs[0], inputs[1], keepdim);
        }
        return rewriter.Build<paddle::dialect::MaxOp>(inputs[0], attrs);
      });
}

pir::Attribute CreateIrAttribute(const std::any& obj) {
  try {
    if (obj.type() == typeid(bool)) {
      return IrAttributeCreator<bool>()(std::any_cast<bool>(obj));
    } else if (obj.type() == typeid(int32_t)) {
      return IrAttributeCreator<int32_t>()(std::any_cast<int32_t>(obj));
    } else if (obj.type() == typeid(int64_t)) {
      return IrAttributeCreator<int64_t>()(std::any_cast<int64_t>(obj));
    } else if (obj.type() == typeid(float)) {
      return IrAttributeCreator<float>()(std::any_cast<float>(obj));
    } else if (obj.type() == typeid(double)) {
      return IrAttributeCreator<double>()(std::any_cast<double>(obj));
    } else if (obj.type() == typeid(std::string)) {
      return IrAttributeCreator<std::string>()(std::any_cast<std::string>(obj));
    } else if (obj.type() == typeid(const char*)) {
      return IrAttributeCreator<std::string>()(std::any_cast<const char*>(obj));
    } else if (obj.type() == typeid(phi::DataType)) {
      return IrAttributeCreator<phi::DataType>()(
          std::any_cast<phi::DataType>(obj));
    } else if (obj.type() == typeid(phi::Place)) {
      return IrAttributeCreator<phi::Place>()(std::any_cast<phi::Place>(obj));
    } else if (obj.type() == typeid(phi::DataLayout)) {
      return IrAttributeCreator<phi::DataLayout>()(
          std::any_cast<phi::DataLayout>(obj));
    } else if (obj.type() == typeid(std::vector<int32_t>)) {  // NOLINT
      return IrAttributeCreator<std::vector<int32_t>>()(
          std::any_cast<std::vector<int32_t>>(obj));
    } else if (obj.type() == typeid(std::vector<int64_t>)) {
      return IrAttributeCreator<std::vector<int64_t>>()(
          std::any_cast<std::vector<int64_t>>(obj));
    } else if (obj.type() == typeid(std::vector<float>)) {
      return IrAttributeCreator<std::vector<float>>()(
          std::any_cast<std::vector<float>>(obj));
    } else if (obj.type() == typeid(phi::IntArray)) {
      return IrAttributeCreator<phi::IntArray>()(
          std::any_cast<phi::IntArray>(obj));
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "Type error. CreateIrAttribute for type(%s) "
          "is unimplemented CreateInCurrently.",
          obj.type().name()));
    }
  } catch (const std::bad_any_cast& e) {
    PADDLE_THROW(common::errors::Fatal(
        "%s: CreateIrAttribute for type(%s) not successfully.",
        e.what(),
        obj.type().name()));
  }
}

pir::AttributeMap CreateAttributeMap(
    const std::unordered_map<std::string, Attribute>& attrs,
    const MatchContextImpl& src_match_ctx) {
  pir::AttributeMap attr_map;
  for (const auto& kv : attrs) {
    std::visit(
        [&](auto&& arg) {
          if constexpr (std::is_same_v<std::decay_t<decltype(arg)>,
                                       NormalAttribute>) {
            attr_map[kv.first] = src_match_ctx.GetIrAttr(arg.name());
          }
          if constexpr (std::is_same_v<std::decay_t<decltype(arg)>,
                                       ComputeAttribute>) {
            MatchContext ctx(std::make_shared<MatchContextImpl>(src_match_ctx));
            attr_map[kv.first] =
                CreateIrAttribute(arg.attr_compute_func()(ctx));
          }
        },
        kv.second);
  }
  return attr_map;
}

pir::Value GetIrValueByDrrTensor(const Tensor* tensor,
                                 const MatchContextImpl& res_match_ctx) {
  if (tensor->is_none()) {
    return pir::Value{};
  }
  return res_match_ctx.GetIrValue(tensor->name());
}

std::vector<pir::Value> GetIrValuesByDrrTensors(
    const std::vector<const Tensor*>& tensors,
    const MatchContextImpl& res_match_ctx) {
  std::vector<pir::Value> ir_values;
  ir_values.reserve(tensors.size());
  for (const auto* tensor : tensors) {
    ir_values.push_back(GetIrValueByDrrTensor(tensor, res_match_ctx));
  }
  return ir_values;
}

void BindIrOutputsWithDrrOutputs(const std::vector<const Tensor*>& tensors,
                                 pir::Operation* op,
                                 MatchContextImpl* match_ctx) {
  if (op->isa<paddle::dialect::ReshapeOp>()) {
    match_ctx->BindIrValue(tensors[0]->name(), op->result(0));
    return;
  }
  PADDLE_ENFORCE_LE(
      tensors.size(),
      op->num_results(),
      common::errors::InvalidArgument(
          "The size of drr outputs should less equal the size of pir outputs"));
  for (size_t i = 0; i < tensors.size(); ++i) {
    match_ctx->BindIrValue(tensors[i]->name(), op->result(i));
  }
}

pir::Operation* CreateOperation(const OpCall& op_call,
                                const MatchContextImpl& src_match_ctx,
                                pir::PatternRewriter& rewriter,  // NOLINT
                                MatchContextImpl* res_match_ctx) {
  VLOG(6) << "Drr create [" << op_call.name() << "] op...";
  pir::Operation* op = OperationFactory::Instance().CreateOperation(
      op_call.name(),
      GetIrValuesByDrrTensors(op_call.inputs(), *res_match_ctx),
      CreateAttributeMap(op_call.attributes(), src_match_ctx),
      rewriter);
  auto runtime_attr_map =
      CreateAttributeMap(op_call.runtime_attributes(), src_match_ctx);
  for (const auto& kv : runtime_attr_map) {
    op->set_attribute(kv.first, kv.second);
  }
  BindIrOutputsWithDrrOutputs(op_call.outputs(), op, res_match_ctx);
  VLOG(6) << "Drr create [" << op_call.name() << " @" << op << "] op done.";
  return op;
}

}  // namespace paddle::drr
