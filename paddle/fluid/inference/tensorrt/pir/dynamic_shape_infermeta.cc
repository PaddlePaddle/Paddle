// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/tensorrt/pir/dynamic_shape_infermeta_factory.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/builtin_attribute.h"

namespace paddle::inference::tensorrt::pir {

nvinfer1::DimsExprs UnchangedInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const ::pir::AttributeMap& op_attributes) {
  PADDLE_ENFORCE_EQ(nb_inputs,
                    1,
                    common::errors::InvalidArgument(
                        "inputs of UnchangedInferMeta should be equal to 1, "
                        "But received (%s)",
                        nb_inputs));
  return inputs[0];
}

inline const nvinfer1::IDimensionExpr* CalcOutputSize(
    const nvinfer1::IDimensionExpr* input_size,
    const nvinfer1::IDimensionExpr* filter_size,
    const nvinfer1::IDimensionExpr* dilation,
    const nvinfer1::IDimensionExpr* padding1,
    const nvinfer1::IDimensionExpr* padding2,
    const nvinfer1::IDimensionExpr* stride,
    nvinfer1::IExprBuilder& expr_builder  // NOLINT
) {
  // dkernel = dilation * (filter_size - 1) + 1;
  const nvinfer1::IDimensionExpr* dkernel = expr_builder.operation(
      nvinfer1::DimensionOperation::kSUM,
      *expr_builder.operation(
          nvinfer1::DimensionOperation::kPROD,
          *dilation,
          *expr_builder.operation(nvinfer1::DimensionOperation::kSUB,
                                  *filter_size,
                                  *expr_builder.constant(1))),
      *expr_builder.constant(1));

  // output_size = (input_size + padding1 + padding2 - dkernel) / stride + 1;
  const nvinfer1::IDimensionExpr* tmp = expr_builder.operation(
      nvinfer1::DimensionOperation::kSUB,
      *expr_builder.operation(
          nvinfer1::DimensionOperation::kSUM,
          *expr_builder.operation(
              nvinfer1::DimensionOperation::kSUM, *input_size, *padding1),
          *padding2),
      *dkernel);

  const nvinfer1::IDimensionExpr* output_size = expr_builder.operation(
      nvinfer1::DimensionOperation::kSUM,
      *expr_builder.operation(
          nvinfer1::DimensionOperation::kFLOOR_DIV, *tmp, *stride),
      *expr_builder.constant(1));
  return output_size;
}

nvinfer1::DimsExprs UnfoldInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const ::pir::AttributeMap& op_attributes) {
  PADDLE_ENFORCE_EQ(
      nb_inputs,
      1,
      common::errors::InvalidArgument("inputs of unfold should be equal to 1, "
                                      "But received (%s)",
                                      nb_inputs));

  const nvinfer1::DimsExprs in_dims = inputs[0];
  std::vector<const nvinfer1::IDimensionExpr*> out_dims;
  out_dims.push_back(in_dims.d[0]);

  std::vector<int> kernel_sizes;
  auto kernel_sizes_attrs = op_attributes.at("kernel_sizes")
                                .dyn_cast<::pir::ArrayAttribute>()
                                .AsVector();
  for (auto kernel_sizes_attr : kernel_sizes_attrs) {
    kernel_sizes.push_back(
        kernel_sizes_attr.dyn_cast<::pir::Int32Attribute>().data());
  }

  std::vector<int> dilations;
  auto dilations_attrs = op_attributes.at("dilations")
                             .dyn_cast<::pir::ArrayAttribute>()
                             .AsVector();
  for (auto dilations_attr : dilations_attrs) {
    dilations.push_back(
        dilations_attr.dyn_cast<::pir::Int32Attribute>().data());
  }

  std::vector<int> paddings;
  auto paddings_attrs =
      op_attributes.at("paddings").dyn_cast<::pir::ArrayAttribute>().AsVector();
  for (auto paddings_attr : paddings_attrs) {
    paddings.push_back(paddings_attr.dyn_cast<::pir::Int32Attribute>().data());
  }

  std::vector<int> strides;
  auto strides_attrs =
      op_attributes.at("strides").dyn_cast<::pir::ArrayAttribute>().AsVector();
  for (auto strides_attr : strides_attrs) {
    strides.push_back(strides_attr.dyn_cast<::pir::Int32Attribute>().data());
  }

  // output_channels = in_dims[1] * kernel_sizes[0] * kernel_sizes[1];
  const nvinfer1::IDimensionExpr* output_channels = expr_builder.operation(
      nvinfer1::DimensionOperation::kPROD,
      *in_dims.d[1],
      *expr_builder.operation(nvinfer1::DimensionOperation::kPROD,
                              *expr_builder.constant(kernel_sizes[0]),
                              *expr_builder.constant(kernel_sizes[1])));
  out_dims.push_back(output_channels);

  const nvinfer1::IDimensionExpr* output_height =
      CalcOutputSize(in_dims.d[2],
                     expr_builder.constant(kernel_sizes[0]),
                     expr_builder.constant(dilations[0]),
                     expr_builder.constant(paddings[0]),
                     expr_builder.constant(paddings[2]),
                     expr_builder.constant(strides[0]),
                     expr_builder);
  const nvinfer1::IDimensionExpr* output_width =
      CalcOutputSize(in_dims.d[3],
                     expr_builder.constant(kernel_sizes[1]),
                     expr_builder.constant(dilations[1]),
                     expr_builder.constant(paddings[1]),
                     expr_builder.constant(paddings[3]),
                     expr_builder.constant(strides[1]),
                     expr_builder);

  const nvinfer1::IDimensionExpr* output_col_length = expr_builder.operation(
      nvinfer1::DimensionOperation::kPROD, *output_height, *output_width);

  out_dims.push_back(output_col_length);
  nvinfer1::DimsExprs output = {};
  output.nbDims = out_dims.size();
  for (size_t i = 0; i < out_dims.size(); i++) output.d[i] = out_dims[i];
  return output;
}

nvinfer1::DimsExprs ArgsortInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const ::pir::AttributeMap& op_attributes) {
  const nvinfer1::DimsExprs input_dims = inputs[0];
  nvinfer1::DimsExprs output = {};
  output.nbDims = input_dims.nbDims;
  for (int i = 0; i < input_dims.nbDims; ++i) {
    output.d[i] = input_dims.d[i];
  }
  return output;
}

nvinfer1::DimsExprs Pad3dInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const ::pir::AttributeMap& op_attributes) {
  const nvinfer1::DimsExprs x_dim = inputs[0];

  nvinfer1::DimsExprs out_dims = {};
  out_dims.nbDims = x_dim.nbDims;
  out_dims.d[0] = x_dim.d[0];

  auto data_format = op_attributes.at("data_format")
                         .dyn_cast<::pir::StrAttribute>()
                         .AsString();

  auto paddings_iter = op_attributes.find("paddings");
  if (paddings_iter != op_attributes.end()) {
    auto paddings_attr =
        paddings_iter->second.dyn_cast<::pir::ArrayAttribute>();
    PADDLE_ENFORCE_NOT_NULL(paddings_attr,
                            common::errors::InvalidArgument(
                                "paddings_values must be an array attribute."));
    PADDLE_ENFORCE_EQ(paddings_attr.size(),
                      6,
                      common::errors::InvalidArgument(
                          "paddings must have 6 elements, but got %d.",
                          paddings_attr.size()));

    std::vector<int64_t> paddings(6);
    for (size_t i = 0; i < 6; ++i) {
      auto int_attr = paddings_attr[i].dyn_cast<::pir::Int64Attribute>();
      PADDLE_ENFORCE_NOT_NULL(
          int_attr,
          common::errors::InvalidArgument(
              "paddings_values must contain int64 attributes."));
      paddings[i] = int_attr.data();
    }

    if (data_format == "NCDHW") {
      out_dims.d[1] = x_dim.d[1];
      out_dims.d[2] = expr_builder.operation(
          nvinfer1::DimensionOperation::kSUM,
          *expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                  *x_dim.d[2],
                                  *expr_builder.constant(paddings[4])),
          *expr_builder.constant(paddings[5]));
      out_dims.d[3] = expr_builder.operation(
          nvinfer1::DimensionOperation::kSUM,
          *expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                  *x_dim.d[3],
                                  *expr_builder.constant(paddings[2])),
          *expr_builder.constant(paddings[3]));
      out_dims.d[4] = expr_builder.operation(
          nvinfer1::DimensionOperation::kSUM,
          *expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                  *x_dim.d[4],
                                  *expr_builder.constant(paddings[0])),
          *expr_builder.constant(paddings[1]));
    } else if (data_format == "NDHWC") {
      out_dims.d[4] = x_dim.d[4];
      out_dims.d[1] = expr_builder.operation(
          nvinfer1::DimensionOperation::kSUM,
          *expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                  *x_dim.d[1],
                                  *expr_builder.constant(paddings[4])),
          *expr_builder.constant(paddings[5]));
      out_dims.d[2] = expr_builder.operation(
          nvinfer1::DimensionOperation::kSUM,
          *expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                  *x_dim.d[2],
                                  *expr_builder.constant(paddings[2])),
          *expr_builder.constant(paddings[3]));
      out_dims.d[3] = expr_builder.operation(
          nvinfer1::DimensionOperation::kSUM,
          *expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                  *x_dim.d[3],
                                  *expr_builder.constant(paddings[0])),
          *expr_builder.constant(paddings[1]));
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Unsupported data_format: %s.", data_format));
    }
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "paddings_values attribute is missing, required for TensorRT."));
  }
  return out_dims;
}

PD_REGISTER_DYNAMIC_INFER_META_FN(inverse, UnchangedInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(unfold, UnfoldInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(argsort, ArgsortInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(pad3d, Pad3dInferMeta);
}  // namespace paddle::inference::tensorrt::pir
