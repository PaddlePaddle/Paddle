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

#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/binary_infer_sym.h"
#include "paddle/common/ddim.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_utils.h"

COMMON_DECLARE_bool(manually_trans_conv_filter);

namespace {

inline void UpdatePaddingAndDilation(
    std::vector<symbol::DimExpr> *paddings,
    std::vector<symbol::DimExpr> *dilation,
    const std::string padding_algorithm,
    const std::vector<symbol::DimExpr> data_dims,
    const std::vector<int> &strides,
    const std::vector<symbol::DimExpr> &ksize) {
  // set padding size == data_dims.size() * 2
  if (paddings->size() == data_dims.size()) {
    for (size_t i = 0; i < data_dims.size(); ++i) {
      symbol::DimExpr copy_pad = *(paddings->begin() + 2 * i);
      paddings->insert(paddings->begin() + 2 * i + 1, copy_pad);
    }
  }

  // when padding_algorithm is "VALID" or "SAME"
  symbol::DimExpr zero{0};
  symbol::DimExpr one{1};
  symbol::DimExpr two{2};
  if (padding_algorithm == "SAME") {
    symbol::DimExprBuilder builder;
    for (size_t i = 0; i < data_dims.size(); ++i) {
      symbol::DimExpr out_size = (data_dims[i] + strides[i] - 1) / strides[i];
      symbol::DimExpr pad_sum = builder.Max(
          (out_size - one) * strides[i] + ksize[i] - data_dims[i], zero);

      symbol::DimExpr pad_0 = pad_sum / two;
      symbol::DimExpr pad_1 = pad_sum - pad_0;

      *(paddings->begin() + i * 2) = pad_0;
      *(paddings->begin() + i * 2 + 1) = pad_1;

      // dilation
      *(dilation->begin() + i) = one;
    }

  } else if (padding_algorithm == "VALID") {
    for (auto it = paddings->begin(); it != paddings->end(); it++) {
      *it = zero;
    }
  }
}

}  // namespace
namespace paddle::dialect {

bool AllcloseOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  const auto y_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1)).shape();
  PADDLE_ENFORCE_EQ(x_shape.size(),
                    y_shape.size(),
                    common::errors::PreconditionNotMet(
                        "Input(X) and Input(Y) must have the same "
                        "dimension size. but got %d vs %d",
                        x_shape.size(),
                        y_shape.size()));
  for (size_t i = 0; i < x_shape.size(); ++i) {
    infer_context->AddEqualCstr(x_shape[i], y_shape[i]);
  }

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(std::vector<symbol::DimExpr>{})});
  return true;
}

bool ApplyPerChannelScaleOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &scales_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  std::vector<symbol::DimExpr> x_shape = x_shape_or_data.shape();
  std::vector<symbol::DimExpr> scales_shape = scales_shape_or_data.shape();

  PADDLE_ENFORCE_EQ(
      x_shape.size(),
      2UL,
      common::errors::InvalidArgument(
          "The rank of Input(x) must be 2, but received %d.", x_shape.size()));

  PADDLE_ENFORCE_EQ(scales_shape.size(),
                    1UL,
                    common::errors::InvalidArgument(
                        "The rank of Input(scales) must be 1, but received %d.",
                        scales_shape.size()));

  infer_context->AddEqualCstr(x_shape[1], scales_shape[0]);

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_shape)});

  return true;
}

bool BoxClipOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &input_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  const auto &im_info_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1)).shape();

  // Check rank and dimensions of input tensors
  const auto &three = symbol::DimExpr{3};
  const auto &four = symbol::DimExpr{4};
  infer_context->AddEqualCstr(input_shape[input_shape.size() - 1], four);
  PADDLE_ENFORCE_EQ(im_info_shape.size(),
                    2,
                    common::errors::InvalidArgument(
                        "The rank of Input(im_info) in BoxClipOp must be 2. "
                        "But received rank = %d",
                        im_info_shape.size()));
  infer_context->AddEqualCstr(im_info_shape[1], three);

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(input_shape)});

  return true;
}

bool Atan2OpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  const auto x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  const auto y_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1)).shape();
  PADDLE_ENFORCE_EQ(x_shape.size(),
                    y_shape.size(),
                    common::errors::PreconditionNotMet(
                        "Input(X) and Input(Y) must have the same "
                        "dimension size. but got %d vs %d",
                        x_shape.size(),
                        y_shape.size()));
  for (size_t i = 0; i < x_shape.size(); ++i) {
    infer_context->AddEqualCstr(x_shape[i], y_shape[i]);
  }

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_shape)});
  return true;
}

bool BceLossOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &input_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  const auto &label_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1)).shape();

  int rank = input_shape.size();
  PADDLE_ENFORCE_EQ(rank,
                    label_shape.size(),
                    common::errors::InvalidArgument(
                        "Input(X) and Input(Label) shall have the same rank."
                        "But received: the rank of Input(X) is [%d], "
                        "the rank of Input(Label) is [%d].",
                        rank,
                        label_shape.size()));

  for (int i = 0; i < rank; ++i) {
    infer_context->AddEqualCstr(input_shape[i], label_shape[i]);
  }

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(input_shape)});

  return true;
}

bool BceLoss_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return BceLossOpInferSymbolicShape(op, infer_context);
}

bool BinomialOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &count_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  const auto &prob_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1)).shape();
  PADDLE_ENFORCE_EQ(count_shape.size(),
                    prob_shape.size(),
                    common::errors::PreconditionNotMet(
                        "Input(count) and Input(prob) must have the same "
                        "dimension size. but got %d vs %d",
                        count_shape.size(),
                        prob_shape.size()));
  for (size_t i = 0; i < count_shape.size(); ++i) {
    infer_context->AddEqualCstr(count_shape[i], prob_shape[i]);
  }
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(count_shape)});
  return true;
}

bool Binomial_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return BinomialOpInferSymbolicShape(op, infer_context);
}

bool BincountOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &x_dims = x_shape_or_data.shape();

  PADDLE_ENFORCE_EQ(x_dims.size(),
                    1,
                    common::errors::InvalidArgument(
                        "The 'shape' of Input(X) must be 1-D tensor. But the "
                        "dimension of Input(X) is [%d]",
                        x_dims.size()));

  if (op->operand_source(1)) {
    const auto &weights_shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(1));
    const std::vector<symbol::DimExpr> &weights_dims =
        weights_shape_or_data.shape();

    PADDLE_ENFORCE_EQ(weights_dims.size(),
                      1,
                      common::errors::InvalidArgument(
                          "The 'shape' of Input(Weights) must be 1-D tensor. "
                          "But the dimension of Input(Weights) is [%d]",
                          weights_dims.size()));
    infer_context->AddEqualCstr(weights_dims[0], x_dims[0]);
  }

  symbol::DimExpr out_unknown = infer_context->GetNextSymName();
  const std::vector<symbol::DimExpr> out_dims = {out_unknown};
  symbol::ShapeOrDataDimExprs output_dims{
      symbol::TensorShapeOrDataDimExprs(out_dims)};
  infer_context->SetShapeOrDataForValue(op->result(0), output_dims);

  return true;
}

bool BmmOpInferSymbolicShape(pir::Operation *op,
                             pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &y_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  const std::vector<symbol::DimExpr> &x_shapes = x_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &y_shapes = y_shape_or_data.shape();
  std::size_t x_size = x_shapes.size();
  std::size_t y_size = y_shapes.size();

  PADDLE_ENFORCE_EQ(
      x_size,
      3,
      common::errors::InvalidArgument("Input(X) of BmmOp must be 3-dimensional "
                                      "in BmmOp, but received X's shape: [%d].",
                                      x_size));
  PADDLE_ENFORCE_EQ(
      y_size,
      3,
      common::errors::InvalidArgument("Input(Y) of BmmOp must be 3-dimensional "
                                      "in BmmOp, but received Y's shape: [%d].",
                                      y_size));

  infer_context->AddEqualCstr(x_shapes[2], y_shapes[1]);
  infer_context->AddEqualCstr(x_shapes[0], y_shapes[0]);

  const symbol::DimExpr batch_size = x_shapes[0];

  const symbol::DimExpr out_height = x_shapes[1];
  const symbol::DimExpr out_width = y_shapes[2];

  std::vector<symbol::DimExpr> out_dims = {batch_size, out_height, out_width};

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(out_dims)});

  return true;
}

bool CtcAlignOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &input_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  std::vector<symbol::DimExpr> out_shape = {input_shape[0], 1};
  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_shape)};
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(input_shape)});
  infer_context->SetShapeOrDataForValue(op->result(1), shape_data);
  return true;
}

bool Conv2dOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  const std::vector<int> strides =
      paddle::dialect::details::GetVectorAttr<int>(op, "strides");

  std::vector<int> paddings =
      paddle::dialect::details::GetVectorAttr<int>(op, "paddings");

  std::vector<int> dilations =
      paddle::dialect::details::GetVectorAttr<int>(op, "dilations");

  const auto &attributes = op->attributes();
  const std::string data_format =
      attributes.at("data_format").dyn_cast<pir::StrAttribute>().AsString();

  const std::string padding_algorithm = attributes.at("padding_algorithm")
                                            .dyn_cast<pir::StrAttribute>()
                                            .AsString();

  const auto in_s_or_d =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto filter_s_or_d =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

  const std::vector<symbol::DimExpr> in_data_dims =
      channel_last ? std::vector<symbol::DimExpr>(in_s_or_d.shape().begin() + 1,
                                                  in_s_or_d.shape().end() - 1)
                   : std::vector<symbol::DimExpr>(in_s_or_d.shape().begin() + 2,
                                                  in_s_or_d.shape().end());

  const std::vector<symbol::DimExpr> filter_data_dims = [&]() {
    if (channel_last && FLAGS_manually_trans_conv_filter) {  // NHWC
      return std::vector<symbol::DimExpr>(filter_s_or_d.shape().begin() + 1,
                                          filter_s_or_d.shape().end() - 1);
    } else {
      return std::vector<symbol::DimExpr>(filter_s_or_d.shape().begin() + 2,
                                          filter_s_or_d.shape().end());
    }
  }();

  std::vector<symbol::DimExpr> ksize = filter_data_dims;

  std::vector<symbol::DimExpr> new_paddings;
  for (const auto &i : paddings) {
    new_paddings.push_back(symbol::DimExpr{i});
  }
  std::vector<symbol::DimExpr> new_dilations;
  for (const auto &i : dilations) {
    new_dilations.push_back(symbol::DimExpr{i});
  }

  UpdatePaddingAndDilation(&new_paddings,
                           &new_dilations,
                           padding_algorithm,
                           in_data_dims,
                           strides,
                           ksize);

  const symbol::ShapeOrDataDimExprs &shape_data = [&] {
    std::vector<symbol::DimExpr> out_s_or_d({in_s_or_d.shape()[0]});
    if (!channel_last) {
      out_s_or_d.push_back(filter_s_or_d.shape()[0]);
    }

    for (size_t i = 0; i < in_data_dims.size(); ++i) {
      const symbol::DimExpr dkernel =
          new_dilations[i] * (filter_data_dims[i] - 1) + 1;
      symbol::DimExpr output_size = (in_data_dims[i] + new_paddings[2 * i] +
                                     new_paddings[2 * i + 1] - dkernel) /
                                        strides[i] +
                                    1;
      out_s_or_d.push_back(output_size);
    }
    if (channel_last) {
      out_s_or_d.push_back(filter_s_or_d.shape()[0]);
    }

    return symbol::ShapeOrDataDimExprs{
        symbol::TensorShapeOrDataDimExprs(out_s_or_d)};
  }();

  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);

  return true;
}

bool Conv3dOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  return Conv2dOpInferSymbolicShape(op, infer_context);
}

bool ConvTransposeFunction(pir::Operation *op,
                           pir::InferSymbolicShapeContext *infer_context,
                           std::vector<symbol::DimExpr> output_size) {
  auto x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  auto filter_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1)).shape();

  std::vector<int> strides =
      paddle::dialect::details::GetVectorAttr<int>(op, "strides");
  std::vector<int> paddings =
      paddle::dialect::details::GetVectorAttr<int>(op, "paddings");
  std::vector<int> output_padding =
      paddle::dialect::details::GetVectorAttr<int>(op, "output_padding");
  std::string padding_algorithm =
      op->attribute<pir::StrAttribute>("padding_algorithm").AsString();
  int groups = op->attribute<pir::Int32Attribute>("groups").data();
  std::vector<int> dilations =
      paddle::dialect::details::GetVectorAttr<int>(op, "dilations");
  std::string data_format =
      op->attribute<pir::StrAttribute>("data_format").AsString();

  std::vector<symbol::DimExpr> new_paddings;
  for (const auto &i : paddings) {
    new_paddings.push_back(symbol::DimExpr{i});
  }
  std::vector<symbol::DimExpr> new_dilations;
  for (const auto &i : dilations) {
    new_dilations.push_back(symbol::DimExpr{i});
  }

  PADDLE_ENFORCE_EQ(x_shape.size() == 4 || x_shape.size() == 5,
                    true,
                    phi::errors::InvalidArgument(
                        "Input of Op(conv_transpose) should be 4-D or "
                        "5-D Tensor. But received: %u-D Tensor",
                        x_shape.size()));
  PADDLE_ENFORCE_EQ(
      x_shape.size(),
      filter_shape.size(),
      phi::errors::InvalidArgument(
          "The input's dimension size and filter's dimension size of "
          "Op (conv_transpose) should be equal. But received: the shape of "
          "the dimension size of input is [%d],  the dimension size of filter "
          "is [%d]. ",
          x_shape.size(),
          filter_shape.size()));

  int stride_size = static_cast<int>(strides.size());
  for (int i = 0; i < stride_size; ++i) {
    PADDLE_ENFORCE_GT(
        strides[i],
        0,
        phi::errors::InvalidArgument(
            "The strides of Op(conv_transpose) should be greater than 0. "
            "But received: the strides of Op(conv_transpose) is [%d].",
            strides[i]));
  }

  int in_sub_stride_size = x_shape.size() - stride_size;

  PADDLE_ENFORCE_EQ(
      x_shape.size() - strides.size(),
      2U,
      phi::errors::InvalidArgument(
          "The input's dimension size minus Attr(stride)'s size must "
          "be equal to 2 for Op(conv_transpose). But received: [%d], the "
          "input's dimension size is [%d]"
          "the Attr(stride)'s size is [%d].",
          in_sub_stride_size,
          x_shape.size(),
          strides.size()));

  if (!output_size.empty())
    PADDLE_ENFORCE_EQ(
        output_size.size(),
        strides.size(),
        phi::errors::InvalidArgument(
            "The Attr(output_size) and Attr(stride) of Op(conv_transpose) "
            "should be the same."));

  if (output_padding.size() > 0)
    PADDLE_ENFORCE_EQ(
        output_padding.size(),
        strides.size(),
        phi::errors::InvalidArgument(
            "The Attr(output_padding) and Attr(stride) of Op(conv_transpose) "
            "should be the same."));

  const bool normal_data = (data_format != "NHWC");
  const symbol::DimExpr C =
      (normal_data ? x_shape[1] : x_shape[x_shape.size() - 1]);

  infer_context->AddEqualCstr(filter_shape[0], C);

  const std::vector<symbol::DimExpr> x_data_dims =
      normal_data
          ? std::vector<symbol::DimExpr>(x_shape.begin() + 2, x_shape.end())
          : std::vector<symbol::DimExpr>(x_shape.begin() + 1,
                                         x_shape.end() - 1);

  const std::vector<symbol::DimExpr> filter_data_dims = [&]() {
    return std::vector<symbol::DimExpr>(filter_shape.begin() + 2,
                                        filter_shape.end());
  }();
  std::vector<symbol::DimExpr> ksize = filter_data_dims;
  UpdatePaddingAndDilation(&new_paddings,
                           &new_dilations,
                           padding_algorithm,
                           x_data_dims,
                           strides,
                           ksize);
  std::vector<symbol::DimExpr> output_shape({x_shape[0]});

  if (normal_data) {
    output_shape.push_back(filter_shape[1] * groups);
  }
  const int offset = (normal_data ? 2 : 1);  // NHWC
  for (int i = 0; i < static_cast<int>(strides.size()); ++i) {
    symbol::DimExpr filter_extent =
        new_dilations[i] * (filter_shape[i + 2] - 1) + 1;
    symbol::DimExpr infer_shape;
    if (x_shape[i + offset].Has<std::int64_t>()) {
      if (x_shape[i + offset].Get<int64_t>() > 0) {
        infer_shape = (x_shape[i + offset] - 1) * symbol::DimExpr(strides[i]) -
                      new_paddings[2 * i] - new_paddings[2 * i + 1] +
                      filter_extent;
      }
    } else {
      infer_shape = infer_context->GetNextSymName();
    }
    if (!output_size.empty()) {
      output_shape.push_back(output_size[i]);
    } else if (!output_padding.empty()) {
      output_shape.push_back(infer_shape + symbol::DimExpr(output_padding[i]));
    } else {
      output_shape.push_back(infer_shape);
    }
  }

  if (!normal_data) {
    output_shape.push_back(filter_shape[1] * groups);
  }
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(output_shape)});
  return true;
}

bool CrossOpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  const auto &y_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1)).shape();

  size_t x_dim = x_shape.size();
  size_t y_dim = y_shape.size();

  PADDLE_ENFORCE_EQ(x_dim,
                    y_dim,
                    common::errors::InvalidArgument(
                        "The 'shape' of Input(X) should be equal to "
                        "the 'shape' of Input(Y). But received "
                        "Input(X).dimensions = [%d], "
                        "Input(Y).dimensions = [%d]",
                        x_dim,
                        y_dim));

  for (size_t i = 0; i < x_dim; i++) {
    infer_context->AddEqualCstr(x_shape[i], y_shape[i]);
  }

  const int axis = op->attribute<pir::Int32Attribute>("axis").data();
  if (axis != common::DDim::kMaxRank) {
    const int dim = axis < 0 ? axis + x_dim : axis;
    infer_context->AddEqualCstr(x_shape[dim], symbol::DimExpr{3});
    infer_context->AddEqualCstr(y_shape[dim], symbol::DimExpr{3});
  }

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_shape)});

  return true;
}

bool Conv3dTransposeOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  std::vector<int> out_size =
      paddle::dialect::details::GetVectorAttr<int>(op, "output_size");
  std::vector<symbol::DimExpr> output_size;
  for (const auto &i : out_size) {
    output_size.emplace_back(symbol::DimExpr{i});
  }
  return ConvTransposeFunction(op, infer_context, output_size);
}

bool Conv2dTransposeOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  //  if ( op->attributes().find("output_size") !=  op->attributes().end()) {
  std::vector<symbol::DimExpr> output_size =
      details::GetIntArrayFromAttrOrOperand(
          op, infer_context, "output_size", 2);
  return ConvTransposeFunction(op, infer_context, output_size);
}

bool Conv2dTransposeBiasOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  std::vector<int> out_size =
      paddle::dialect::details::GetVectorAttr<int>(op, "output_size");
  std::vector<symbol::DimExpr> output_size;
  for (const auto &i : out_size) {
    output_size.emplace_back(symbol::DimExpr{i});
  }
  return ConvTransposeFunction(op, infer_context, output_size);
}
// bool CorrelationOpInferSymbolicShape(pir::Operation *op,
//                                      pir::InferSymbolicShapeContext
//                                      *infer_context) {
//   // pass
//   return true;
// }

bool DepthwiseConv2dOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return Conv2dOpInferSymbolicShape(op, infer_context);
}

bool DepthwiseConv2dTransposeOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return Conv2dTransposeOpInferSymbolicShape(op, infer_context);
}

bool DotOpInferSymbolicShape(pir::Operation *op,
                             pir::InferSymbolicShapeContext *infer_context) {
  const auto x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  auto x_shape = x_shape_or_data.shape();
  const size_t x_rank = x_shape.size();
  PADDLE_ENFORCE_EQ(
      1 == x_rank || 2 == x_rank,
      true,
      common::errors::InvalidArgument(
          "ShapeError: The dimensions of input tensor X (%u) should be 1 or 2",
          x_rank));

  const auto y_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const auto y_shape = y_shape_or_data.shape();
  auto y_rank = y_shape.size();
  PADDLE_ENFORCE_EQ(x_rank == y_rank,
                    true,
                    common::errors::InvalidArgument(
                        "ShapeError: The rank of input tensor Y (%u) must "
                        "match that of input tensor X(%u).",
                        y_rank,
                        x_rank));
  for (size_t i = 0; i < x_rank; ++i) {
    infer_context->AddEqualCstr(x_shape[i], y_shape[i]);
  }
  // Dot OP require both inputs should have the same shape

  x_shape.erase(x_shape.end() - 1);
  auto output_shape =
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_shape)};
  infer_context->SetShapeOrDataForValue(op->result(0), output_shape);
  return true;
}

bool DistOpInferSymbolicShape(pir::Operation *op,
                              pir::InferSymbolicShapeContext *infer_context) {
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(std::vector<symbol::DimExpr>{})});
  return true;
}

bool DropoutOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(x_shape_or_data.shape())});

  if (op->result(1)) {
    symbol::TensorShapeOrDataDimExprs mask_shape(x_shape_or_data.shape());
    infer_context->SetShapeOrDataForValue(
        op->result(1), symbol::ShapeOrDataDimExprs{mask_shape});
  }

  return true;
}

bool EmbeddingOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const std::vector<symbol::DimExpr> &x_dims =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  const std::vector<symbol::DimExpr> &weight_dims =
      infer_context->GetShapeOrDataForValue(op->operand_source(1)).shape();

  const symbol::ShapeOrDataDimExprs &shape_data = [&] {
    std::vector<symbol::DimExpr> out_dims = x_dims;
    // no need to check validation of weight_dims index, since all checks have
    // been done at corresponding InferMeta
    out_dims.emplace_back(weight_dims[1]);
    return symbol::ShapeOrDataDimExprs{
        symbol::TensorShapeOrDataDimExprs(out_dims)};
  }();

  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);

  return true;
}

bool EqualAllOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_dims =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  const auto &y_dims =
      infer_context->GetShapeOrDataForValue(op->operand_source(1)).shape();

  PADDLE_ENFORCE_GE(
      x_dims.size(),
      y_dims.size(),
      common::errors::InvalidArgument(
          "The size of dim_y should not be greater than dim_x's."));

  std::vector<symbol::DimExpr> out_dims =
      {};  // Adjust the dimensions as necessary
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(out_dims)});

  return true;
}

bool SparseWeightEmbeddingOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  PADDLE_THROW(common::errors::Unimplemented(
      op->name() + " 's InferSymbolicShape interface is NOT implemented now."));
  return true;
}

bool ExpandAsOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  std::vector<int> target_shape =
      paddle::dialect::details::GetVectorAttr<int>(op, "target_shape");
  const std::vector<symbol::DimExpr> &output_dims = [&] {
    const auto &input_shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(1));
    if (!input_shape_or_data.isa<symbol::NullShapeOrDataDimExpr>()) {
      return input_shape_or_data.shape();
    }
    std::vector<symbol::DimExpr> output_dims;
    output_dims.reserve(target_shape.size());
    for (int shape : target_shape) {
      output_dims.push_back(shape);
    }
    return output_dims;
  }();

  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs(output_dims));

  return true;
}

bool FillDiagonalTensorOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  std::vector<symbol::DimExpr> x_dims = x_shape_or_data.shape();

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_dims)});

  return true;
}

bool FillDiagonalTensor_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return FillDiagonalTensorOpInferSymbolicShape(op, infer_context);
}

bool FusedSoftmaxMaskOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> x_dims = x_shape_or_data.shape();

  const auto &mask_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const std::vector<symbol::DimExpr> mask_dims = mask_shape_or_data.shape();

  PADDLE_ENFORCE_EQ(
      x_dims.size(),
      4,
      common::errors::InvalidArgument("Input x must be in 4D dimension but "
                                      "received the dimension of X is %d",
                                      x_dims.size()));
  PADDLE_ENFORCE_EQ(
      mask_dims.size(),
      4,
      common::errors::InvalidArgument("Input mask must be in 4D dimension but "
                                      "received the dimension of mask is %d",
                                      mask_dims.size()));

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_dims)});

  return true;
}

bool GridSampleOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  auto x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  auto grid_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1)).shape();

  PADDLE_ENFORCE_GE(x_shape.size(),
                    4,
                    common::errors::InvalidArgument(
                        "Input(X) of GridSampleOp should be 4-D Tensor, but "
                        "received X dimension size(%d)",
                        x_shape.size()));
  PADDLE_ENFORCE_LE(x_shape.size(),
                    5,
                    common::errors::InvalidArgument(
                        "Input(X) of GridSampleOp should be 4-D Tensor, but "
                        "received X dimension size(%d)",
                        x_shape.size()));
  PADDLE_ENFORCE_GE(grid_shape.size(),
                    4,
                    common::errors::InvalidArgument(
                        "Input(Grid) of GridSampleOp should be 4-D Tensor, "
                        "but received Grid dimension size(%d)",
                        grid_shape.size()));
  PADDLE_ENFORCE_LE(grid_shape.size(),
                    5,
                    common::errors::InvalidArgument(
                        "Input(Grid) of GridSampleOp should be 4-D Tensor, "
                        "but received Grid dimension size(%d)",
                        grid_shape.size()));

  if (grid_shape.size() == 4) {
    infer_context->AddEqualCstr(grid_shape[3], symbol::DimExpr(2));
  }
  if (grid_shape.size() == 5) {
    infer_context->AddEqualCstr(grid_shape[4], symbol::DimExpr(3));
  }

  infer_context->AddEqualCstr(grid_shape[0], x_shape[0]);

  std::vector<symbol::DimExpr> out_shape;
  if (grid_shape.size() == 4) {
    out_shape = {x_shape[0], x_shape[1], grid_shape[1], grid_shape[2]};
  } else {
    out_shape = {
        x_shape[0], x_shape[1], grid_shape[1], grid_shape[2], grid_shape[3]};
  }

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});

  return true;
}

bool GatherOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  const auto &input_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &index_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  const auto &numel = [&] {
    symbol::DimExpr numel{1};
    for (const auto &dim_expr : index_shape_or_data.shape()) {
      numel = numel * dim_expr;
    }
    return numel;
  }();

  int axis = 0;
  const auto &attributes = op->attributes();
  if (op->HasAttribute("axis")) {  // CINN Dialect
    axis = attributes.at("axis").dyn_cast<pir::Int32Attribute>().data();
  } else {
    PADDLE_ENFORCE_EQ(
        op->num_operands() == 3,
        true,
        common::errors::InvalidArgument(
            "in GatherOpInferSymbolicShape: The number of operands should be "
            "3 when the axis is not set."));
    const auto &axis_shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(2));
    axis =
        static_cast<int>(axis_shape_or_data.data().value()[0].Get<int64_t>());
  }

  const std::vector<symbol::DimExpr> &input_sym_shape =
      input_shape_or_data.data().has_value()
          ? input_shape_or_data.data().value()
          : input_shape_or_data.shape();

  const std::vector<symbol::DimExpr> &index_sym_shape =
      index_shape_or_data.data().has_value()
          ? index_shape_or_data.data().value()
          : index_shape_or_data.shape();

  if (axis < 0) axis += input_sym_shape.size();

  const auto &out_sym_shape = [&] {
    std::vector<symbol::DimExpr> out_sym_shape;

    if (index_sym_shape.size() == 0) {
      if (input_sym_shape.size() == 1) {
        out_sym_shape.push_back(symbol::DimExpr{0});
      } else {
        for (int i = 0; i < axis; ++i) {
          out_sym_shape.push_back(input_sym_shape[i]);
        }
        for (size_t i = axis + 1; i < input_sym_shape.size(); ++i) {
          out_sym_shape.push_back(input_sym_shape[i]);
        }
      }
    } else {
      for (int i = 0; i < axis; ++i) {
        out_sym_shape.push_back(input_sym_shape[i]);
      }
      out_sym_shape.push_back(numel);
      for (size_t i = axis + 1; i < input_sym_shape.size(); ++i) {
        out_sym_shape.push_back(input_sym_shape[i]);
      }
    }
    return out_sym_shape;
  }();

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_sym_shape)};

  pir::Value res = op->result(0);
  infer_context->SetShapeOrDataForValue(res, shape_data);

  return true;
}

bool GatherNdOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &index_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  const std::vector<symbol::DimExpr> &x_sym_shape = x_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &index_sym_shape =
      index_shape_or_data.shape();

  int x_dims_size = x_sym_shape.size();
  int index_dims_size = index_sym_shape.size();

  std::vector<symbol::DimExpr> result_sym_dims;
  // The result dims is
  //   Index.shape[:-1] + X.shape[Index.shape[-1]:]
  for (int i = 0; i < index_dims_size - 1; ++i) {
    result_sym_dims.emplace_back(index_sym_shape[i]);
  }

  PADDLE_ENFORCE_EQ(
      index_sym_shape[index_dims_size - 1].Has<std::int64_t>(),
      true,
      common::errors::InvalidArgument(
          "in GatherNdOpInferSymbolicShape: index[-1] should be unknown"));

  for (int i = static_cast<int>(
           index_sym_shape[index_dims_size - 1].Get<std::int64_t>());
       i < x_dims_size;
       ++i) {
    result_sym_dims.emplace_back(x_sym_shape[i]);
  }

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(result_sym_dims)};

  pir::Value res = op->result(0);
  infer_context->SetShapeOrDataForValue(res, shape_data);

  return true;
}

bool GatherTreeOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &ids_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const symbol::ShapeOrDataDimExprs &parents_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  const std::vector<symbol::DimExpr> &ids_shape = ids_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &parents_shape =
      parents_shape_or_data.shape();

  PADDLE_ENFORCE_EQ(ids_shape.size() == parents_shape.size(),
                    true,
                    common::errors::InvalidArgument(
                        "The shape of Input(Parents) must be same with the "
                        "shape of Input(Ids)."));
  size_t rank = ids_shape.size();
  for (size_t i = 0; i < rank; ++i) {
    infer_context->AddEqualCstr(ids_shape[i], parents_shape[i]);
  }

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(ids_shape)});

  return true;
}

bool HuberLossOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &input =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &label =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const std::vector<symbol::DimExpr> &input_dims = input.shape();
  const std::vector<symbol::DimExpr> &label_dims = label.shape();
  PADDLE_ENFORCE_EQ(input_dims.size(),
                    label_dims.size(),
                    common::errors::InvalidArgument(
                        "Input(input) rank and Input(label) rank should be "
                        "same, but received input rank(%d) != label rank(%d)",
                        input_dims.size(),
                        label_dims.size()));

  int rank = input_dims.size();
  for (int i = 0; i < rank; ++i) {
    infer_context->AddEqualCstr(input_dims[i], label_dims[i]);
  }

  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(label_dims)});
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(label_dims)});

  return true;
}

bool HistogramOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &input_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  int64_t bins = op->attribute<pir::Int64Attribute>("bins").data();
  float min = op->attribute<pir::FloatAttribute>("min").data();
  float max = op->attribute<pir::FloatAttribute>("max").data();
  PADDLE_ENFORCE_GE(bins,
                    1,
                    common::errors::InvalidArgument(
                        "The bins should be greater than or equal to 1."
                        "But received nbins is %d",
                        bins));
  PADDLE_ENFORCE_GE(
      max,
      min,
      common::errors::InvalidArgument("max must be larger or equal to min."
                                      "But received max is %f, min is %f",
                                      max,
                                      min));
  if (op->operand_source(1)) {
    const symbol::ShapeOrDataDimExprs &weight_shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(1));
    size_t ndims_input = input_shape_or_data.shape().size();
    for (size_t i = 0; i < ndims_input; ++i) {
      infer_context->AddEqualCstr(weight_shape_or_data.shape()[i],
                                  input_shape_or_data.shape()[i]);
    }
  }
  std::vector<symbol::DimExpr> dim_out = {bins};
  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(dim_out)};
  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}

bool IndexSampleOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(operand_shape_or_data.shape())});
  return true;
}

bool KldivLossOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &label_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const auto &x_shape = x_shape_or_data.shape();
  const auto &label_shape = label_shape_or_data.shape();

  PADDLE_ENFORCE_EQ(x_shape.size(),
                    label_shape.size(),
                    common::errors::InvalidArgument(
                        "Input(X) rank and Input(Target) rank should be same, "
                        "but received X rank(%d) != Target rank(%d)",
                        x_shape.size(),
                        label_shape.size()));

  for (size_t i = 0; i < x_shape.size(); ++i) {
    infer_context->AddEqualCstr(x_shape[i], label_shape[i]);
  }

  std::string reduction =
      op->attribute<pir::StrAttribute>("reduction").AsString();
  bool reduction_valid = (reduction == "mean" || reduction == "sum" ||
                          reduction == "batchmean" || reduction == "none");
  PADDLE_ENFORCE_EQ(
      reduction_valid,
      true,
      common::errors::InvalidArgument(
          "Attr(reduction) can only be 'none'|'batchmean'|'sum'|'mean'."));

  std::vector<symbol::DimExpr> out_shape;
  if (reduction == "none") {
    out_shape = x_shape;
  } else {
    out_shape = std::vector<symbol::DimExpr>{};
  }
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});

  return true;
}

bool KronOpInferSymbolicShape(pir::Operation *op,
                              pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  const auto &y_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1)).shape();
  const int rank_x = x_shape_or_data.size();
  const int rank_y = y_shape_or_data.size();
  const int rank = (rank_x > rank_y) ? rank_x : rank_y;

  std::vector<symbol::DimExpr> dim_out;
  dim_out.reserve(rank);
  const auto one = symbol::DimExpr{1};
  const auto minus_one = symbol::DimExpr{-1};
  for (int i = 0; i < rank; i++) {
    symbol::DimExpr dim_xi =
        (i < rank - rank_x) ? one : x_shape_or_data.at(i - (rank - rank_x));
    symbol::DimExpr dim_yi =
        (i < rank - rank_y) ? one : y_shape_or_data.at(i - (rank - rank_y));
    dim_out.push_back(dim_xi * dim_yi);
  }
  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(dim_out)};
  pir::Value res = op->result(0);
  infer_context->SetShapeOrDataForValue(res, shape_data);
  return true;
}

bool LstsqOpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  const auto &y_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1)).shape();
  size_t x_rank = x_shape.size();
  size_t y_rank = y_shape.size();

  const symbol::DimExpr &m = x_shape[x_rank - 2];
  const symbol::DimExpr &n = x_shape[x_rank - 1];
  const symbol::DimExpr &nrhs = y_shape[y_rank - 1];
  PADDLE_ENFORCE(
      (x_rank >= 2 && y_rank >= 2 && x_rank == y_rank),
      common::errors::InvalidArgument(
          "Expects input tensor x and y to have the same dimension, "
          "and the rank of input tensor x and y should be greater "
          "than or equal to 2, but received the rank of input tensor "
          "x is %d, the rank of input tensor y is %d",
          x_rank,
          y_rank));
  std::vector<symbol::DimExpr> batch_dims_vec;
  for (size_t i = 0; i < x_rank - 2; i++) {
    infer_context->AddEqualCstr(x_shape[i], y_shape[i]);
    batch_dims_vec.emplace_back(x_shape[i]);
  }
  infer_context->AddEqualCstr(m, y_shape[y_rank - 2]);
  infer_context->SetShapeOrDataForValue(
      op->result(2),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(batch_dims_vec)});

  std::vector<symbol::DimExpr> residuals_shape{};
  if (m.isa<int64_t>() && n.isa<int64_t>()) {
    int64_t m_value = m.Get<std::int64_t>();
    int64_t n_value = n.Get<std::int64_t>();
    if (m_value > n_value) {
      batch_dims_vec.emplace_back(nrhs);
      residuals_shape = batch_dims_vec;
      batch_dims_vec.pop_back();
    } else {
      residuals_shape.emplace_back(0);
    }
  }
  if (paddle::dialect::details::IsFakeValue(op->result(1)) ||
      residuals_shape.empty()) {
    infer_context->SetSymbolForValueByStaticShape(op->result(1));
  } else {
    infer_context->SetShapeOrDataForValue(
        op->result(1),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(residuals_shape)});
  }
  symbol::DimExprBuilder builder;
  batch_dims_vec.emplace_back(builder.Min(m, n));
  infer_context->SetShapeOrDataForValue(
      op->result(3),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(batch_dims_vec)});
  batch_dims_vec[x_rank - 2] = n;
  batch_dims_vec.emplace_back(nrhs);
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(batch_dims_vec)});
  return true;
}

bool LuUnpackOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();
  int x_rank = x_shape.size();

  bool unpack_ludata =
      op->attribute<pir::BoolAttribute>("unpack_ludata").data();
  bool unpack_pivots =
      op->attribute<pir::BoolAttribute>("unpack_pivots").data();

  PADDLE_ENFORCE_GE(
      x_rank,
      2,
      common::errors::InvalidArgument(
          "The rank of input must be greater than or equal to 2."));

  const auto &m = x_shape[x_rank - 1];
  const auto &n = x_shape[x_rank - 2];

  if (unpack_pivots) {
    std::vector<symbol::DimExpr> p_shape = x_shape;
    p_shape[x_rank - 1] = n;
    infer_context->SetShapeOrDataForValue(
        op->result(0),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(p_shape)});
  }

  if (unpack_ludata) {
    std::vector<symbol::DimExpr> l_shape = x_shape;
    std::vector<symbol::DimExpr> u_shape = x_shape;
    if (m.isa<int64_t>() && n.isa<int64_t>()) {
      int64_t m_value = static_cast<int64_t>(m.Get<std::int64_t>());
      int64_t n_value = static_cast<int64_t>(n.Get<std::int64_t>());
      if (m_value >= n_value) {
        l_shape[x_rank - 1] = n;
      } else {
        u_shape[x_rank - 2] = m;
      }
    }
    infer_context->SetShapeOrDataForValue(
        op->result(1),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(l_shape)});
    infer_context->SetShapeOrDataForValue(
        op->result(2),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(u_shape)});
  }

  return true;
}

bool MatrixRankTolOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  const auto &tol_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  std::vector<symbol::DimExpr> tol_shape = tol_shape_or_data.shape();
  size_t x_rank = x_shape.size();
  PADDLE_ENFORCE_GE(x_rank,
                    2,
                    common::errors::InvalidArgument(
                        "The dims of input must be greater than 2"));
  bool hermitian = GetBoolAttr(op, "hermitian");
  if (hermitian) {
    infer_context->AddEqualCstr(x_shape[x_rank - 2], x_shape[x_rank - 1]);
  }
  std::vector<symbol::DimExpr> x_shape_batch = [&] {
    std::vector<symbol::DimExpr> x_shape_batch;
    for (size_t i = 0; i < x_rank - 2; ++i) {
      x_shape_batch.push_back(x_shape[i]);
    }
    return x_shape_batch;
  }();

  int diff = x_shape_batch.size() - tol_shape.size();
  if (diff > 0) {
    for (int i = 0; i < diff; i++) {
      tol_shape.emplace(tol_shape.begin(), 1);
    }
  } else {
    for (int i = 0; i < -diff; i++) {
      x_shape_batch.emplace(x_shape_batch.begin(), 1);
    }
  }

  const std::vector<symbol::DimExpr> shapes = [&] {
    std::vector<symbol::DimExpr> shapes;
    symbol::DimExprBuilder builder;
    for (size_t i = 0; i < x_shape_batch.size(); i++) {
      if (x_shape_batch[i] == tol_shape[i]) {
        shapes.emplace_back(x_shape_batch[i]);
      } else if (x_shape_batch[i] == 1) {
        shapes.emplace_back(tol_shape[i]);
      } else if (tol_shape[i] == 1) {
        shapes.emplace_back(x_shape_batch[i]);
      } else {
        shapes.emplace_back(builder.Broadcast(x_shape_batch[i], tol_shape[i]));
        infer_context->AddBroadcastableCstr(x_shape_batch[i], tol_shape[i]);
      }
    }
    return shapes;
  }();

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(shapes)};
  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}

bool MaskedSelectOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const std::vector<symbol::DimExpr> &out_dims = [&] {
    std::vector<symbol::DimExpr> out_dims;
    symbol::DimExpr out_shape =
        infer_context->GetNextSymName();  // unknown until runtime
    out_dims.push_back(out_shape);
    return out_dims;
  }();
  // Add constrains between the shapes of x and mask
  const std::vector<symbol::DimExpr> &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  const std::vector<symbol::DimExpr> &mask_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1)).shape();
  size_t ndims_x = x_shape.size();
  size_t ndims_mask = mask_shape.size();
  if (ndims_x >= ndims_mask) {
    size_t diff = ndims_x - ndims_mask;
    for (size_t i = 0; i < ndims_mask; i++) {
      infer_context->AddBroadcastableCstr(x_shape[i + diff], mask_shape[i]);
    }
  } else {
    size_t diff = ndims_mask - ndims_x;
    for (size_t i = 0; i < ndims_x; i++) {
      infer_context->AddBroadcastableCstr(x_shape[i], mask_shape[i + diff]);
    }
  }

  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs{out_dims});
  return true;
}

bool MatmulOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  // x_dims can't be const or ref here, in case to be broadcasted
  std::vector<symbol::DimExpr> x_dims = [&] {
    std::vector<symbol::DimExpr> dims;
    const auto &x_shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(0));
    dims = x_shape_or_data.shape();
    return dims;
  }();

  // y_dims can't be const or ref here, in case to be broadcasted
  std::vector<symbol::DimExpr> y_dims = [&] {
    std::vector<symbol::DimExpr> dims;
    const auto y_shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(1));
    dims = y_shape_or_data.shape();
    return dims;
  }();

  size_t ndims_x = x_dims.size();
  size_t ndims_y = y_dims.size();

  const bool x_broadcasted = [&] {
    bool broadcasted = false;
    if (ndims_x == 1) {
      x_dims.insert(x_dims.begin(), 1);
      ndims_x = 2;
      broadcasted = true;
    }
    return broadcasted;
  }();

  const bool y_broadcasted = [&] {
    bool broadcasted = false;
    if (ndims_y == 1) {
      y_dims.emplace_back(1);
      ndims_y = 2;
      broadcasted = true;
    }
    return broadcasted;
  }();

  const auto &Padding = [&]() {
    int diff = ndims_x - ndims_y;
    if (diff > 0) {
      for (int i = 0; i < diff; i++) {
        y_dims.emplace(y_dims.begin(), 1);
      }
    } else {
      for (int i = 0; i < -diff; i++) {
        x_dims.emplace(x_dims.begin(), 1);
      }
    }
    ndims_x = x_dims.size();
    ndims_y = y_dims.size();
  };

  const auto &out_dims = [&]() -> std::vector<symbol::DimExpr> {
    Padding();
    std::vector<symbol::DimExpr> res;
    symbol::DimExprBuilder builder;
    for (size_t i = 0; i < ndims_x - 2; ++i) {
      res.emplace_back(builder.Broadcast(x_dims[i], y_dims[i]));
      infer_context->AddBroadcastableCstr(x_dims[i], y_dims[i]);
    }
    bool transpose_x_attr = GetBoolAttr(op, "transpose_x");
    bool transpose_y_attr = GetBoolAttr(op, "transpose_y");
    symbol::DimExpr out_M =
        transpose_x_attr ? x_dims[ndims_x - 1] : x_dims[ndims_x - 2];
    symbol::DimExpr out_N =
        transpose_y_attr ? y_dims[ndims_y - 2] : y_dims[ndims_y - 1];
    if (!x_broadcasted) {
      res.emplace_back(out_M);
    }
    if (!y_broadcasted) {
      res.emplace_back(out_N);
    }

    symbol::DimExpr x_K =
        transpose_x_attr ? x_dims[ndims_x - 2] : x_dims[ndims_x - 1];
    symbol::DimExpr y_K =
        transpose_y_attr ? y_dims[ndims_y - 1] : y_dims[ndims_y - 2];
    infer_context->AddEqualCstr(x_K, y_K);

    return res;
  }();

  infer_context->SetShapeOrDataForValue(op->result(0),
                                        ShapeOrData{TensorExprs(out_dims)});
  return true;
}

bool MatrixNmsOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &bboxes_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &scores_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const std::vector<symbol::DimExpr> &bboxes_shape =
      bboxes_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &scores_shape =
      scores_shape_or_data.shape();

  infer_context->AddEqualCstr(bboxes_shape[2], symbol::DimExpr(4));
  infer_context->AddEqualCstr(bboxes_shape[1], scores_shape[2]);

  symbol::DimExpr num_kept = infer_context->GetNextSymName();

  std::vector<symbol::DimExpr> out_shape = {
      num_kept, bboxes_shape[2] + symbol::DimExpr(2)};
  std::vector<symbol::DimExpr> index_shape = {num_kept, symbol::DimExpr(1)};

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});

  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(index_shape)});

  if (op->result(2)) {
    std::vector<symbol::DimExpr> roisnum_shape = {scores_shape[0]};
    infer_context->SetShapeOrDataForValue(
        op->result(2),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(roisnum_shape)});
  }

  return true;
}

bool MarginCrossEntropyOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &logits_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &labels_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  std::vector<symbol::DimExpr> logits_dims = logits_shape_or_data.shape();
  std::vector<symbol::DimExpr> labels_dims = labels_shape_or_data.shape();

  size_t logits_rank = logits_dims.size();
  auto axis = logits_rank - 1;

  for (size_t i = 0; i < logits_rank; i++) {
    if (i != axis) {
      infer_context->AddEqualCstr(logits_dims[i], labels_dims[i]);
    }
  }

  const auto &one = symbol::DimExpr{1};

  if (labels_dims.size() > 1) {
    infer_context->AddEqualCstr(labels_dims[axis], one);
  }

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(logits_dims)});

  logits_dims[axis] = symbol::DimExpr(1);

  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(logits_dims)});

  return true;
}

bool MatmulWithFlattenOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &y_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  std::vector<symbol::DimExpr> x_dims = x_shape_or_data.shape();
  std::vector<symbol::DimExpr> y_dims = y_shape_or_data.shape();

  int x_num_col_dims =
      op->attribute<pir::Int32Attribute>("x_num_col_dims").data();
  int y_num_col_dims =
      op->attribute<pir::Int32Attribute>("y_num_col_dims").data();

  PADDLE_ENFORCE_GT(
      x_dims.size(),
      x_num_col_dims,
      common::errors::InvalidArgument(
          "The input tensor X's dimensions of MulOp "
          "should be larger than x_num_col_dims. But received X's "
          "dimensions = %d, X's shape = [%s], x_num_col_dims = %d.",
          x_dims.size(),
          x_dims,
          x_num_col_dims));
  PADDLE_ENFORCE_GT(
      y_dims.size(),
      y_num_col_dims,
      common::errors::InvalidArgument(
          "The input tensor Y's dimensions of MulOp "
          "should be larger than y_num_col_dims. But received Y's "
          "dimensions = %d, Y's shape = [%s], y_num_col_dims = %d.",
          y_dims.size(),
          y_dims,
          y_num_col_dims));

  auto slice =
      [](const std::vector<symbol::DimExpr> &dims, int begin, int end) {
        std::vector<symbol::DimExpr> slice_dims;
        slice_dims.reserve(end - begin);
        for (int i = begin; i < end; ++i) {
          slice_dims.push_back(dims[i]);
        }
        return slice_dims;
      };
  auto x_mat_dims = slice(x_dims, x_num_col_dims, x_dims.size());
  auto y_mat_dims = slice(y_dims, 0, y_num_col_dims);

  PADDLE_ENFORCE_EQ(x_mat_dims.size(),
                    y_mat_dims.size(),
                    common::errors::InvalidArgument(
                        "The second dimension of input x_mat_dims should be "
                        "equal to the first dimension of input y_mat_dims. But "
                        "received X's shape = [%s], Y's shape = [%s].",
                        x_mat_dims.size(),
                        y_mat_dims.size()));

  for (size_t i = 0; i < x_mat_dims.size(); ++i) {
    infer_context->AddEqualCstr(x_mat_dims[i], y_mat_dims[i]);
  }

  std::vector<symbol::DimExpr> output_dims;
  output_dims.reserve(
      static_cast<size_t>(x_num_col_dims + y_dims.size() - y_num_col_dims));

  for (size_t i = 0; i < static_cast<size_t>(x_num_col_dims); ++i) {
    output_dims.push_back(x_dims[i]);
  }
  for (size_t i = static_cast<size_t>(y_num_col_dims); i < y_dims.size(); ++i) {
    output_dims.push_back(y_dims[i]);
  }

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(output_dims)});

  return true;
}

bool MvOpInferSymbolicShape(pir::Operation *op,
                            pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &vec_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  PADDLE_ENFORCE_EQ(x_shape_or_data.shape().size(),
                    2,
                    common::errors::InvalidArgument(
                        "The rank of input X should be 2, but is %d",
                        x_shape_or_data.shape().size()));
  PADDLE_ENFORCE_EQ(vec_shape_or_data.shape().size(),
                    1,
                    common::errors::InvalidArgument(
                        "The rank of input Vec should be 1, but is %d",
                        vec_shape_or_data.shape().size()));
  infer_context->AddEqualCstr(x_shape_or_data.shape()[1],
                              vec_shape_or_data.shape()[0]);

  std::vector<symbol::DimExpr> out_shape = {x_shape_or_data.shape()[0]};
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});
  return true;
}

bool PriorBoxOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();
  const auto &image_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const std::vector<symbol::DimExpr> &image_shape = image_shape_or_data.shape();

  PADDLE_ENFORCE_EQ(
      x_shape.size(),
      4,
      common::errors::InvalidArgument(
          "The Input(Input) of Op(PriorBoxOp) should be a 4-D Tensor "
          "and data format is NCHW. But received Input's dimensions = %d, "
          "shape = [%s].",
          x_shape.size(),
          x_shape));
  PADDLE_ENFORCE_EQ(
      image_shape.size(),
      4,
      common::errors::InvalidArgument(
          "The Input(Image) of Op(PriorBoxOp) should be a 4-D Tensor "
          "and data format is NCHW. But received Image's dimensions = %d, "
          "shape = [%s].",
          image_shape.size(),
          image_shape));

  const std::vector<float> &aspect_ratios =
      paddle::dialect::details::GetVectorAttr<float>(op, "aspect_ratios");
  bool flip = op->attribute<pir::BoolAttribute>("flip").data();
  std::vector<float> aspect_ratios_vec;
  constexpr float epsilon = 1e-6;
  aspect_ratios_vec.push_back(1.0f);
  for (auto ar : aspect_ratios) {
    bool already_exist = false;
    for (auto item : aspect_ratios_vec) {
      if (fabs(ar - item) < epsilon) {
        already_exist = true;
        break;
      }
    }
    if (!already_exist) {
      aspect_ratios_vec.push_back(ar);
      if (flip) {
        aspect_ratios_vec.push_back(1.0f / ar);
      }
    }
  }
  const std::vector<float> &min_sizes =
      paddle::dialect::details::GetVectorAttr<float>(op, "min_sizes");
  const std::vector<float> &max_sizes =
      paddle::dialect::details::GetVectorAttr<float>(op, "max_sizes");
  int num_priors = aspect_ratios_vec.size() * min_sizes.size();
  if (!max_sizes.empty()) {
    num_priors += max_sizes.size();
  }

  std::vector<symbol::DimExpr> out_shape = {
      x_shape[2], x_shape[3], symbol::DimExpr(num_priors), symbol::DimExpr(4)};
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});

  return true;
}

bool PruneGateByCapacityOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &expert_count_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const auto &expert_count_shape = expert_count_shape_or_data.shape();
  int64_t n_expert = op->attribute<pir::Int64Attribute>("n_expert").data();
  int64_t n_worker = op->attribute<pir::Int64Attribute>("n_worker").data();
  symbol::DimExpr expert_count_num_ele = 1;
  for (const auto &i : expert_count_shape) {
    expert_count_num_ele = expert_count_num_ele * i;
  }

  infer_context->AddEqualCstr(expert_count_num_ele,
                              symbol::DimExpr(n_expert * n_worker));
  const auto &gate_idx_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(gate_idx_shape_or_data.shape())});
  return true;
}
// bool PullBoxSparseOpInferSymbolicShape(pir::Operation *op,
//                                        pir::InferSymbolicShapeContext
//                                        *infer_context) {
//   // pass
//   return true;
// }

// bool PullGpuPsSparseOpInferSymbolicShape(pir::Operation *op,
//                                          pir::InferSymbolicShapeContext
//                                          *infer_context) {
//   // pass
//   return true;
// }

// bool PullSparseV2OpInferSymbolicShape(pir::Operation *op,
//                                       pir::InferSymbolicShapeContext
//                                       *infer_context) {
//   // pass
//   return true;
// }

bool RepeatInterleaveWithTensorIndexOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const symbol::ShapeOrDataDimExprs &repeats_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  const auto &attributes = op->attributes();
  int axis = attributes.at("axis").dyn_cast<pir::Int32Attribute>().data();

  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();

  PADDLE_ENFORCE_EQ(
      repeats_shape_or_data.shape().size(),
      1,
      common::errors::InvalidArgument(
          "The shape of Input(Repeats) should be 1-D Tensor, but received "
          "shape is %d-D.",
          repeats_shape_or_data.shape().size()));

  ExprVec repeat_times_shape;
  if (repeats_shape_or_data.data().has_value()) {
    repeat_times_shape.assign(repeats_shape_or_data.data()->begin(),
                              repeats_shape_or_data.data()->end());
  } else {
    symbol::DimExpr out_unknown = infer_context->GetNextSymName();
    repeat_times_shape.push_back(out_unknown);
  }

  const auto &GetSum = [](const auto &dim_exprs) {
    return std::accumulate(
        dim_exprs.begin(), dim_exprs.end(), symbol::DimExpr{0}, std::plus<>());
  };

  int x_rank = x_shape.size();
  if (axis < 0) axis += x_rank;

  const auto &out_sym_shape = [&] {
    std::vector<symbol::DimExpr> out_sym_shape;
    for (int i = 0; i < x_rank; i++) {
      if (i == axis) {
        out_sym_shape.push_back(GetSum(repeat_times_shape));
      } else {
        out_sym_shape.push_back(x_shape.at(i));
      }
    }
    return out_sym_shape;
  }();

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_sym_shape)});

  return true;
}

bool SearchsortedOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  // The shape of output is the same as input `values` (op->operand_source(1))
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  // TODO(fty1777): Add constrains between the shapes of `sorted_sequence` and
  // `values`
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(operand_shape_or_data.shape())});
  return true;
}

bool SegmentPoolOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &input_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &input_shape = input_shape_or_data.shape();
  const auto &ids_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const std::vector<symbol::DimExpr> &ids_shape = ids_shape_or_data.shape();
  const std::string pool_type =
      op->attribute<pir::StrAttribute>("pooltype").AsString();

  std::vector<symbol::DimExpr> out_shape;
  if (ids_shape_or_data.data().has_value()) {
    const auto &ids_data = ids_shape_or_data.data();
    out_shape.push_back(ids_data.value()[ids_shape.size() - 1] +
                        symbol::DimExpr{1});
  } else {
    symbol::DimExpr out_unknown =
        infer_context->GetNextSymName();  // unknown until runtime
    out_shape.push_back(out_unknown);
  }
  int axis = input_shape.size();
  for (int i = 1; i < axis; ++i) {
    out_shape.push_back(input_shape[i]);
  }
  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_shape)};
  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  if (pool_type == "MEAN") {
    std::vector<symbol::DimExpr> summed_shape;
    summed_shape.push_back(out_shape[0]);  // same as before
    summed_shape.push_back(symbol::DimExpr{1});
    infer_context->SetShapeOrDataForValue(
        op->result(1),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(summed_shape)});
  }
  return true;
}

bool SequenceMaskOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();

  std::vector<symbol::DimExpr> y_dims = x_shape;
  if (op->HasAttribute("maxlen")) {
    int maxlen = op->attribute<pir::Int64Attribute>("maxlen").data();
    y_dims.push_back(maxlen > 0 ? symbol::DimExpr(maxlen)
                                : infer_context->GetNextSymName());
  } else if (op->operand_source(1)) {
    const auto &maxlen_shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(1));
    if (maxlen_shape_or_data.data().has_value()) {
      y_dims.push_back(maxlen_shape_or_data.data().value()[0]);
    } else {
      y_dims.push_back(infer_context->GetNextSymName());
    }
  } else {
    PADDLE_THROW(::common::errors::InvalidArgument(
        "Find maxlen or max_len_tensor Failed"));
  }

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(y_dims)});

  return true;
}

bool ShuffleBatchOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_shape)});
  infer_context->SetShapeOrDataForValue(
      op->result(2),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs({1})});
  const symbol::DimExpr shuffleidx = [&] {
    symbol::DimExpr shuffleidx{1};
    for (const auto &i : x_shape) {
      shuffleidx = shuffleidx * i;
    }
    return shuffleidx;
  }();

  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs({shuffleidx})});
  return true;
}

bool StftOpInferSymbolicShape(pir::Operation *op,
                              pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &window_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &window_shape =
      window_shape_or_data.shape();

  int n_fft = op->attribute<pir::Int32Attribute>("n_fft").data();
  int hop_length = op->attribute<pir::Int32Attribute>("hop_length").data();
  bool onesided = op->attribute<pir::BoolAttribute>("onesided").data();

  const int x_rank = x_shape.size();

  PADDLE_ENFORCE_EQ(
      x_rank,
      2,
      common::errors::InvalidArgument(
          "Input(X) of StftOp should be a tensor with shape [N, T], "
          "but got rank %s.",
          x_rank));

  PADDLE_ENFORCE_GT(
      hop_length,
      0,
      common::errors::InvalidArgument(
          "Attribute(hop_length) should be greater than 0, but got %s.",
          hop_length));

  infer_context->AddEqualCstr(window_shape[0], symbol::DimExpr{n_fft});
  const symbol::DimExpr seq_length = x_shape[x_rank - 1];
  const symbol::DimExpr n_frames =
      (symbol::DimExpr{1}) +
      (seq_length - symbol::DimExpr{n_fft}) / symbol::DimExpr{hop_length};

  if (seq_length.isa<int64_t>()) {
    PADDLE_ENFORCE_LE(n_fft,
                      seq_length.Get<std::int64_t>(),
                      common::errors::InvalidArgument(
                          "Attribute(frame_length) should be less equal than "
                          "sequence length, but got (%s) > (%s).",
                          n_fft,
                          seq_length.Get<std::int64_t>()));
  }

  std::vector<symbol::DimExpr> output_shape;
  output_shape.push_back(x_shape[0]);
  if (onesided) {
    output_shape.push_back(symbol::DimExpr{n_fft / 2 + 1});
  } else {
    output_shape.push_back(symbol::DimExpr{n_fft});
  }
  output_shape.push_back(n_frames);

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(output_shape)});
  return true;
}

bool SwigluOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  size_t rank = x_shape_or_data.shape().size();
  if (op->operand_source(1)) {
    const auto &y_shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(1));
    for (size_t i = 0; i < rank; ++i) {
      infer_context->AddEqualCstr(x_shape_or_data.shape()[i],
                                  y_shape_or_data.shape()[i]);
    }
    infer_context->SetShapeOrDataForValue(op->result(0), x_shape_or_data);
  } else {
    std::vector<symbol::DimExpr> x_shape = x_shape_or_data.shape();
    // TODO(CINN): Add distribute constraint
    x_shape[rank - 1] = x_shape[rank - 1] / symbol::DimExpr{2};
    infer_context->SetShapeOrDataForValue(
        op->result(0),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(x_shape)});
  }
  return true;
}

bool IscloseOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  // The shape of output is the same as input `values` (op->operand_source(1))
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(operand_shape_or_data.shape())});
  return true;
}

bool IndexSelectStridedOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  std::vector<symbol::DimExpr> input_dims = x_shape_or_data.shape();

  int dim = op->attribute<pir::Int32Attribute>("dim").data();

  if (dim < 0) {
    dim += input_dims.size();
  }

  std::vector<symbol::DimExpr> output_dims(input_dims.begin(),
                                           input_dims.end());
  output_dims.erase(output_dims.begin() + dim);
  // No need to add any constraints here as we are simply removing a dimension.

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(output_dims)});

  return true;
}

bool AccuracyCheckOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  // The shape of output is the same as input `values` (op->operand_source(1))
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(operand_shape_or_data.shape())});
  return true;
}

bool ReduceAsOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &target_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(target_shape.shape())});
  return true;
}

bool TakeAlongAxisOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  // input
  const auto &arr_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &indices_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const auto &attributes = op->attributes();
  int axis = attributes.at("axis").dyn_cast<pir::Int32Attribute>().data();

  const std::vector<symbol::DimExpr> &arr_sym_shape =
      arr_shape_or_data.data().has_value() ? arr_shape_or_data.data().value()
                                           : arr_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &indices_sym_shape =
      indices_shape_or_data.data().has_value()
          ? indices_shape_or_data.data().value()
          : indices_shape_or_data.shape();

  if (axis < 0) axis += arr_sym_shape.size();

  const auto &out_sym_shape = [&] {
    std::vector<symbol::DimExpr> out_sym_shape;
    for (int i = 0; i < axis; ++i) {
      out_sym_shape.push_back(arr_sym_shape[i]);
    }
    out_sym_shape.push_back(indices_sym_shape[axis]);
    for (size_t i = axis + 1; i < arr_sym_shape.size(); ++i) {
      out_sym_shape.push_back(arr_sym_shape[i]);
    }
    return out_sym_shape;
  }();

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_sym_shape)};

  pir::Value res = op->result(0);
  infer_context->SetShapeOrDataForValue(res, shape_data);

  return true;
}

bool TopPSamplingOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_dims = [op, infer_context] {
    const auto &shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(0));
    if (shape_or_data.data().has_value()) {
      return shape_or_data.data().value();
    } else {
      return shape_or_data.shape();
    }
  }();

  // all the result have the same shape
  for (uint32_t rst_idx = 0; rst_idx < op->num_results(); rst_idx++) {
    const std::vector<symbol::DimExpr> out_dims{x_dims[0], 1};
    infer_context->SetShapeOrDataForValue(
        op->result(rst_idx),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(out_dims)});
  }

  return true;
}

bool TdmChildOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &input_shape = x_shape_or_data.shape();
  int child_nums = op->attribute<pir::Int32Attribute>("child_nums").data();

  std::vector<symbol::DimExpr> output_shape = input_shape;
  output_shape.push_back(symbol::DimExpr(child_nums));

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(output_shape)});
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(output_shape)});

  return true;
}

static inline std::vector<symbol::DimExpr> MatrixGetBroadcastBatchPortion(
    const std::vector<symbol::DimExpr> &x,
    const std::vector<symbol::DimExpr> &y,
    pir::InferSymbolicShapeContext *infer_context) {
  // use int to avoid underflow for minus
  int size_x = x.size();
  int size_y = y.size();

  if (size_y == 0) {
    return {};
  }

  int max_size = std::max(size_x, size_y);
  std::vector<symbol::DimExpr> batchPortion(max_size);

  int size_diff = size_x - size_y;
  if (size_diff > 0) {
    for (int i = 0; i < size_diff; i++) {
      batchPortion[i] = x[i];
    }
  } else {
    size_diff = -size_diff;
    for (int i = 0; i < size_diff; i++) {
      batchPortion[i] = y[i];
    }
  }

  symbol::DimExprBuilder builder;
  for (int i = size_diff; i < max_size; i++) {
    int offset = max_size - i;
    int dim_x = size_x - offset;
    int dim_y = size_y - offset;
    infer_context->AddBroadcastableCstr(x[dim_x], y[dim_y]);
    batchPortion[i] = builder.Broadcast(x[dim_x], y[dim_y]);
  }
  return batchPortion;
}

bool TriangularSolveOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();
  const auto &y_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const std::vector<symbol::DimExpr> &y_shape = y_shape_or_data.shape();

  const auto &x_rank = x_shape.size();
  const auto &y_rank = y_shape.size();

  infer_context->AddEqualCstr(x_shape[x_rank - 2], x_shape[x_rank - 1]);

  std::vector<symbol::DimExpr> x_shape_cut(x_shape.begin(), x_shape.end() - 2);
  std::vector<symbol::DimExpr> y_shape_cut(y_shape.begin(), y_shape.end() - 2);

  std::vector<symbol::DimExpr> expand_batch_portion =
      MatrixGetBroadcastBatchPortion(x_shape_cut, y_shape_cut, infer_context);

  std::vector<symbol::DimExpr> output_shape({expand_batch_portion});
  output_shape.insert(output_shape.end(),
                      {y_shape[y_rank - 2], y_shape[y_rank - 1]});

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(output_shape)});

  return true;
}

bool CholeskySolveOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &b_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &a_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  const std::vector<symbol::DimExpr> &a_shape = a_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &b_shape = b_shape_or_data.shape();
  const auto &a_rank = a_shape.size();
  const auto &b_rank = b_shape.size();

  infer_context->AddEqualCstr(a_shape[a_rank - 2], a_shape[a_rank - 1]);

  std::vector<symbol::DimExpr> a_shape_cut(a_shape.begin(), a_shape.end() - 2);
  std::vector<symbol::DimExpr> b_shape_cut(b_shape.begin(), b_shape.end() - 2);

  std::vector<symbol::DimExpr> expand_batch_portion =
      MatrixGetBroadcastBatchPortion(a_shape_cut, b_shape_cut, infer_context);

  std::vector<symbol::DimExpr> output_shape({expand_batch_portion});
  output_shape.insert(output_shape.end(),
                      {b_shape[b_rank - 2], b_shape[b_rank - 1]});

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(output_shape)});

  return true;
}

bool SolveOpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  return TriangularSolveOpInferSymbolicShape(op, infer_context);
}

bool Unpool3dOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &indices_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &indices_shape =
      indices_shape_or_data.shape();

  for (size_t i = 0; i < x_shape.size(); ++i) {
    infer_context->AddEqualCstr(x_shape[i], indices_shape[i]);
  }

  const std::vector<int> &ksize = details::GetVectorAttr<int>(op, "ksize");
  const std::vector<int> &output_size =
      details::GetVectorAttr<int>(op, "output_size");

  std::vector<symbol::DimExpr> output_shape = {x_shape[0], x_shape[1]};
  for (size_t i = 0; i < ksize.size(); ++i) {
    output_shape.emplace_back(symbol::DimExpr(output_size[i]));
  }

  if (op->result(0)) {
    infer_context->SetShapeOrDataForValue(
        op->result(0),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(output_shape)});
  }

  return true;
}

bool UnpoolOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &indices_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &indices_shape =
      indices_shape_or_data.shape();

  for (size_t i = 0; i < x_shape.size(); ++i) {
    infer_context->AddEqualCstr(x_shape[i], indices_shape[i]);
  }

  const std::vector<int> &ksize = details::GetVectorAttr<int>(op, "ksize");

  const auto &attributes = op->attributes();
  std::vector<symbol::DimExpr> output_size;
  if (attributes.find("output_size") != attributes.end()) {
    std::vector<int64_t> output_size_int =
        details::GetVectorAttr<int64_t>(op, "output_size");
    for (size_t i = 0; i < output_size_int.size(); i++) {
      output_size.emplace_back(symbol::DimExpr(output_size_int[i]));
    }
  } else if (op->operand_source(2)) {
    const auto &shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(2));
    output_size =
        details::GetOrCreateExprVecFromData(shape_or_data, infer_context);
  }

  std::vector<symbol::DimExpr> output_shape = {x_shape[0], x_shape[1]};
  for (size_t i = 0; i < ksize.size(); ++i) {
    output_shape.emplace_back(output_size[i]);
  }

  if (op->result(0)) {
    infer_context->SetShapeOrDataForValue(
        op->result(0),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(output_shape)});
  }

  return true;
}

bool WeightDequantizeOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  const auto &scale_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1)).shape();
  PADDLE_ENFORCE_EQ(x_shape.size(),
                    2UL,
                    common::errors::InvalidArgument(
                        "The x tensor of dequantize op must be 2D, but got[%d]",
                        x_shape.size()));
  const int group_size =
      op->attribute<pir::Int32Attribute>("group_size").data();
  const std::string algo = op->attribute<pir::StrAttribute>("algo").AsString();
  PADDLE_ENFORCE_EQ(
      (group_size == -1 || group_size == 64 || group_size == 128),
      true,
      common::errors::InvalidArgument("group_size must be -1, 64 or 128."));
  symbol::DimExpr real_channel_shape;
  if (algo == "weight_only_int8") {
    real_channel_shape = x_shape[0];
  } else if (algo == "weight_only_int4") {
    real_channel_shape = x_shape[0] * 2;
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The algo must be in ['weight_only_int8', 'weight_only_int4'], "
        "but got[%s]",
        algo));
  }
  if (group_size == -1) {
    PADDLE_ENFORCE_EQ(scale_shape.size(),
                      1UL,
                      common::errors::InvalidArgument(
                          "The scale tensor of dequantize op must "
                          "be 1D in per-channel mode, but got[%d]",
                          scale_shape.size()));
    infer_context->AddEqualCstr(scale_shape[0], real_channel_shape);
  } else {
    PADDLE_ENFORCE_EQ(scale_shape.size(),
                      2UL,
                      common::errors::InvalidArgument(
                          "The scale tensor of dequantize op must "
                          "be 2D in group-wise mode, but got[%d]",
                          scale_shape.size()));
    infer_context->AddEqualCstr(scale_shape[0],
                                (x_shape[1] + (group_size - 1)) / group_size);
    infer_context->AddEqualCstr(scale_shape[1], real_channel_shape);
  }
  std::vector<symbol::DimExpr> out_shape{x_shape[1], real_channel_shape};
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});
  return true;
}

bool YoloBoxOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto anchors =
      paddle::dialect::details::GetVectorAttr<int>(op, "anchors");
  int class_num = op->attribute<pir::Int32Attribute>("class_num").data();

  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();
  int anchor_num = static_cast<int>(anchors.size() / 2);

  bool iou_aware = op->attribute<pir::BoolAttribute>("iou_aware").data();
  if (iou_aware) {
    infer_context->AddEqualCstr(x_shape[1],
                                symbol::DimExpr(anchor_num * (6 + class_num)));
  } else {
    infer_context->AddEqualCstr(x_shape[1],
                                symbol::DimExpr(anchor_num * (5 + class_num)));
  }

  symbol::DimExpr box_num = symbol::DimExpr(0);
  box_num = x_shape[2] * x_shape[3] * symbol::DimExpr(anchor_num);

  std::vector<symbol::DimExpr> boxes_shape = {
      x_shape[0], box_num, symbol::DimExpr(4)};
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(boxes_shape)});

  std::vector<symbol::DimExpr> scores_shape = {
      x_shape[0], box_num, symbol::DimExpr(class_num)};
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(scores_shape)});

  return true;
}

bool IndexSelectOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &index_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  std::vector<symbol::DimExpr> x_shape = x_shape_or_data.shape();
  std::vector<symbol::DimExpr> index_shape = index_shape_or_data.shape();

  int64_t axis = op->attribute<pir::Int32Attribute>("axis").data();

  auto input_rank = x_shape.size();
  auto index_rank = index_shape.size();
  PADDLE_ENFORCE_EQ(
      axis < static_cast<int64_t>(input_rank) &&
          axis >= (0 - static_cast<int64_t>(input_rank)),
      true,
      common::errors::OutOfRange(
          "Attr(axis) is out of range, It's expected "
          "to be in range of [-%d, %d]. But received Attr(axis) = %d.",
          input_rank,
          input_rank - 1,
          axis));

  PADDLE_ENFORCE_EQ(index_rank == 1 || index_rank == 2,
                    true,
                    common::errors::InvalidArgument(
                        "The 'shape' of Input(Index) must be 1-D tensor or 2-D "
                        "tensor where second dimension is 1. "
                        "But received: the 'shape' of Input(Index) is [%s], "
                        "the dimension of Input(Index) is [%d].",
                        index_shape,
                        index_shape.size()));

  if (index_rank == 2)
    infer_context->AddEqualCstr(index_shape[1], symbol::DimExpr{1});

  if (axis < 0) {
    axis += input_rank;
  }

  std::vector<symbol::DimExpr> output_shape = x_shape;
  output_shape[axis] = index_shape[0];

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(output_shape)});

  return true;
}

bool IndexSelect_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return IndexSelectOpInferSymbolicShape(op, infer_context);
}

bool IndexAddOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &index_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const auto &add_value_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));
  std::vector<symbol::DimExpr> x_shape = x_shape_or_data.shape();
  std::vector<symbol::DimExpr> index_shape = index_shape_or_data.shape();
  std::vector<symbol::DimExpr> add_value_shape =
      add_value_shape_or_data.shape();
  int axis = op->attribute<pir::Int32Attribute>("axis").data();
  int ndims_x = x_shape.size();

  // Real axis calculation
  int real_axis = axis >= 0 ? axis : axis + ndims_x;

  // Check dimensions
  PADDLE_ENFORCE_EQ(
      index_shape.size(),
      1,
      common::errors::InvalidArgument("Index tensor must be 1-dimensional."));

  PADDLE_ENFORCE_EQ(
      x_shape.size(),
      add_value_shape.size(),
      common::errors::InvalidArgument(
          "Input and addition value must have the same dimension."));

  for (int i = 0; i < ndims_x; i++) {
    if (i != real_axis) {
      infer_context->AddEqualCstr(x_shape[i], add_value_shape[i]);
    }
  }

  // Set the shape for the output
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_shape)});

  return true;
}

bool IndexAdd_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return IndexAddOpInferSymbolicShape(op, infer_context);
}

bool IndexPutOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  std::vector<symbol::DimExpr> x_shape = x_shape_or_data.shape();

  PADDLE_ENFORCE_LT(
      x_shape.size(),
      7,
      common::errors::InvalidArgument(
          "The rank of input should be less than 7, but received %d.",
          x_shape.size()));

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_shape)});

  return true;
}

bool IndexPut_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return IndexPutOpInferSymbolicShape(op, infer_context);
}

bool LogLossOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &input_shape_or_data =
                                        infer_context->GetShapeOrDataForValue(
                                            op->operand_source(0)),

                                    &label_shape_or_data =
                                        infer_context->GetShapeOrDataForValue(
                                            op->operand_source(1));

  const auto &input_shape = input_shape_or_data.shape();
  const auto &label_shape = label_shape_or_data.shape();

  PADDLE_ENFORCE_EQ(
      input_shape.size() == 2 && label_shape.size() == 2,
      true,
      common::errors::InvalidArgument(
          "The rank of input and label should both be 2, but received: "
          "input: %d, label: %d\n",
          input_shape.size(),
          label_shape.size()));

  for (int i = 0; i < 2; i++) {
    infer_context->AddEqualCstr(input_shape[i], label_shape[i]);
  }

  symbol::DimExpr one_dim = symbol::DimExpr{1};

  infer_context->AddEqualCstr(input_shape[1], one_dim);
  infer_context->AddEqualCstr(label_shape[1], one_dim);

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(input_shape)});

  return true;
}

}  // namespace paddle::dialect

namespace cinn::dialect {
using paddle::dialect::IscloseOpInferSymbolicShape;
}
