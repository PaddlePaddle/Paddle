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

#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/multiary_infer_sym.h"
#include "paddle/common/ddim.h"
#include "paddle/common/layout.h"
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_slice_utils.h"
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"

namespace paddle::dialect {

bool AccuracyOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &out_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const symbol::ShapeOrDataDimExprs &label_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));

  // Assume indices has same shape as inference, because
  // it's the output of topk.
  PADDLE_ENFORCE_EQ(
      label_shape.shape().size(),
      2UL,
      common::errors::InvalidArgument(
          "ShapeError: label's dimensions of AccuracyOp must be 2. "
          "But received label's dimensions = %d",
          label_shape.shape().size()));

  infer_context->AddEqualCstr(label_shape.shape()[1], symbol::DimExpr{1});
  infer_context->AddEqualCstr(out_shape.shape()[0], label_shape.shape()[0]);

  std::vector<symbol::DimExpr> accuracy_shape = {};
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(accuracy_shape)});

  std::vector<symbol::DimExpr> correct_shape = {};
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(correct_shape)});

  std::vector<symbol::DimExpr> total_shape = {};
  infer_context->SetShapeOrDataForValue(
      op->result(2),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(total_shape)});

  return true;
}

bool AddNOpInferSymbolicShape(pir::Operation *op,
                              pir::InferSymbolicShapeContext *infer_context) {
  const auto &input_list_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  PADDLE_ENFORCE_EQ(
      input_list_shape.isa<symbol::TensorListShapeOrDataDimExprs>(),
      true,
      common::errors::InvalidArgument(
          "The type of inputs shape should be TensorListShapeOrDataDimExprs"));
  const auto &inputs_shape =
      input_list_shape.dyn_cast<symbol::TensorListShapeOrDataDimExprs>();
  PADDLE_ENFORCE_GT(
      inputs_shape.size(),
      0,
      common::errors::InvalidArgument(
          "The input tensor X's dimensions of AddNOp "
          "should be larger than 0. But received X's dimensions %d.",
          inputs_shape.size()));
  symbol::TensorShapeOrDataDimExprs candidate_shape = inputs_shape.front();
  for (size_t i = 1; i < inputs_shape.size(); ++i) {
    // 0D tensor
    if (inputs_shape[i].shape().size() == 0) {
      continue;
    }
    if (candidate_shape.shape().size() == 0) {
      candidate_shape = inputs_shape[i];
      continue;
    }
    for (size_t j = 0; j < candidate_shape.shape().size(); ++j) {
      infer_context->AddEqualCstr(candidate_shape.shape()[j],
                                  inputs_shape[i].shape()[j]);
    }
  }
  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::ShapeOrDataDimExprs{candidate_shape});

  return true;
}

bool AddmmOpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  const auto &input_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const auto &y_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));

  auto ndim_input = input_shape.shape().size();
  auto ndim_x = x_shape.shape().size();
  auto ndim_y = y_shape.shape().size();

  PADDLE_ENFORCE_EQ(ndim_input == 2 || ndim_input == 1,
                    true,
                    common::errors::InvalidArgument(
                        "The input tensor input's dimension must be 2 or 1. "
                        "But received input's dimension = [%d].",
                        ndim_input));
  PADDLE_ENFORCE_EQ(ndim_x,
                    2,
                    common::errors::InvalidArgument(
                        "The input tensor x's dimension must be 2. "
                        "But received x's dimension = [%d].",
                        ndim_x));
  PADDLE_ENFORCE_EQ(ndim_y,
                    2,
                    common::errors::InvalidArgument(
                        "The input tensor y's dimension must be 2. "
                        "But received y's dimension = [%d].",
                        ndim_y));

  std::vector<symbol::DimExpr> output_shape;
  output_shape.push_back(x_shape.shape()[0]);
  output_shape.push_back(y_shape.shape()[1]);

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(output_shape)});

  infer_context->AddEqualCstr(x_shape.shape()[1], y_shape.shape()[0]);

  if (ndim_input == 2) {
    infer_context->AddBroadcastableCstr(input_shape.shape()[0],
                                        x_shape.shape()[0]);
    infer_context->AddBroadcastableCstr(input_shape.shape()[1],
                                        y_shape.shape()[1]);
  } else if (ndim_input == 1) {
    infer_context->AddBroadcastableCstr(input_shape.shape()[0],
                                        y_shape.shape()[1]);
  }

  return true;
}

bool Addmm_OpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  return AddmmOpInferSymbolicShape(op, infer_context);
}

bool AucOpInferSymbolicShape(pir::Operation *op,
                             pir::InferSymbolicShapeContext *infer_context) {
  const auto &predict_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &label_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  PADDLE_ENFORCE_GE(
      predict_shape.shape().size(),
      2,
      common::errors::InvalidArgument(
          "The Input(Predict) has not been initialized properly. The "
          "shape of Input(Predict) = [%s], the shape size must be "
          "greater_equal 2.",
          predict_shape.shape()));

  const auto &predict_height = predict_shape.shape()[0];
  const auto &label_height = label_shape.shape()[0];

  infer_context->AddEqualCstr(predict_height, label_height);

  int num_thresholds =
      op->attribute<pir::Int32Attribute>("num_thresholds").data();
  int slide_steps = op->attribute<pir::Int32Attribute>("slide_steps").data();

  int num_pred_buckets = num_thresholds + 1;

  PADDLE_ENFORCE_GE(
      num_pred_buckets,
      1,
      common::errors::InvalidArgument("num_thresholds must larger than 1"));
  PADDLE_ENFORCE_GE(
      slide_steps,
      0,
      common::errors::InvalidArgument("slide_steps must be natural number"));

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(std::vector<symbol::DimExpr>{})});

  if (slide_steps) {
    infer_context->SetShapeOrDataForValue(
        op->result(1),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(std::vector<symbol::DimExpr>{
                (1 + slide_steps) * num_pred_buckets + 1})});
    infer_context->SetShapeOrDataForValue(
        op->result(2),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(std::vector<symbol::DimExpr>{
                (1 + slide_steps) * num_pred_buckets + 1})});
  } else {
    infer_context->SetShapeOrDataForValue(
        op->result(1),
        symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(
            std::vector<symbol::DimExpr>{1, num_pred_buckets})});
    infer_context->SetShapeOrDataForValue(
        op->result(2),
        symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(
            std::vector<symbol::DimExpr>{1, num_pred_buckets})});
  }

  return true;
}

bool BatchNormOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &scale_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(3));
  const auto &bias_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(4));

  std::vector<symbol::DimExpr> x_dims = x_shape_or_data.shape();

  std::string data_layout_str =
      op->attribute<pir::StrAttribute>("data_format").AsString();
  const DataLayout data_layout = common::StringToDataLayout(data_layout_str);

  PADDLE_ENFORCE_GE(
      x_dims.size(),
      2,
      phi::errors::InvalidArgument(
          "ShapeError: the dimension of input "
          "X must greater than or equal to 2. But received: the shape of input "
          "X = [%s], the dimension of input X =[%d]",
          x_dims,
          x_dims.size()));
  PADDLE_ENFORCE_LE(
      x_dims.size(),
      5,
      phi::errors::InvalidArgument(
          "ShapeError: the dimension of input X "
          "must smaller than or equal to 5. But received: the shape of input X "
          "= [%s], the dimension of input X = [%d]",
          x_dims,
          x_dims.size()));

  symbol::DimExpr C = (data_layout == DataLayout::kNCHW)
                          ? x_dims[1]
                          : x_dims[x_dims.size() - 1];

  if (!scale_shape_or_data.isa<symbol::NullShapeOrDataDimExpr>()) {
    std::vector<symbol::DimExpr> scale_dims = scale_shape_or_data.shape();
    PADDLE_ENFORCE_EQ(scale_dims.size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "ShapeError: the dimension of scale must equal to 1."
                          "But received: the dimension of scale is [%d]",
                          scale_dims.size()));
    infer_context->AddEqualCstr(scale_dims[0], C);
  }

  if (!bias_shape_or_data.isa<symbol::NullShapeOrDataDimExpr>()) {
    std::vector<symbol::DimExpr> bias_dims = bias_shape_or_data.shape();
    PADDLE_ENFORCE_EQ(bias_dims.size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "ShapeError: the dimension of bias must equal to 1."
                          "But received: the dimension of bias is [%d]",
                          bias_dims.size()));
    infer_context->AddEqualCstr(bias_dims[0], C);
  }

  // Set output shapes
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_dims)});

  std::vector<symbol::DimExpr> param_dims = {C};
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(param_dims)});
  infer_context->SetShapeOrDataForValue(
      op->result(2),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(param_dims)});

  if (op->result(3) && op->result(3).type()) {
    infer_context->SetShapeOrDataForValue(
        op->result(3),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(param_dims)});
  }
  if (op->result(4) && op->result(4).type()) {
    infer_context->SetShapeOrDataForValue(
        op->result(4),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(param_dims)});
  }
  if (op->result(5) && op->result(5).type()) {
    std::vector<symbol::DimExpr> reserve_space_dims = {symbol::DimExpr{-1}};
    infer_context->SetShapeOrDataForValue(
        op->result(5),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(reserve_space_dims)});
  }

  return true;
}

bool BatchNorm_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return BatchNormOpInferSymbolicShape(op, infer_context);
}

bool BicubicInterpOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &x =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));

  const auto &attributes = op->attributes();

  const std::string data_format =
      attributes.at("data_format").dyn_cast<pir::StrAttribute>().AsString();
  int out_d = attributes.at("out_d").dyn_cast<pir::Int32Attribute>().data();
  int out_h = attributes.at("out_h").dyn_cast<pir::Int32Attribute>().data();
  int out_w = attributes.at("out_w").dyn_cast<pir::Int32Attribute>().data();
  const std::vector<float> &scale = details::GetVectorAttr<float>(op, "scale");

  const bool has_size_tensor = [&] {
    pir::Value size_tensor = op->operand_source(2);
    if (!size_tensor || !size_tensor.type()) {
      return false;
    }
    const auto &list_size_tensor =
        size_tensor.type().dyn_cast<pir::VectorType>();
    return list_size_tensor && !list_size_tensor.empty();
  }();
  auto GetSizeTensorDataExpr =
      [&](pir::Value value) -> std::vector<symbol::DimExpr> {
    const symbol::ShapeOrDataDimExprs &size_tensor_shape =
        infer_context->GetShapeOrDataForValue(value);
    PADDLE_ENFORCE_EQ(
        size_tensor_shape.isa<symbol::TensorListShapeOrDataDimExprs>(),
        true,
        common::errors::InvalidArgument(
            "The size_tensor of Interpolation should be type of "
            "TensorListShapeOrDataDimExprs"));
    return details::GetOrCreateExprVecFromData(size_tensor_shape,
                                               infer_context);
  };
  auto GetOutSizeDataExpr =
      [&](pir::Value value) -> std::vector<symbol::DimExpr> {
    const symbol::ShapeOrDataDimExprs &out_size_tensor_shape =
        infer_context->GetShapeOrDataForValue(value);
    return details::GetOrCreateExprVecFromData(out_size_tensor_shape,
                                               infer_context);
  };
  auto GetOutDimByScale = [&](const symbol::DimExpr &in_dim,
                              float scale) -> symbol::DimExpr {
    PADDLE_ENFORCE_GT(scale,
                      0,
                      common::errors::InvalidArgument(
                          "The scale in Attr(scale) of Operator(interpolate) "
                          "should be greater than 0, but received value is %d.",
                          scale));
    if (in_dim.isa<int64_t>()) {
      return symbol::DimExpr{
          static_cast<int64_t>(in_dim.dyn_cast<int64_t>() * scale)};
    }
    return symbol::DimExpr{infer_context->GetNextSymName()};
  };

  std::vector<int> size_tensor;
  if (out_d != -1) size_tensor.push_back(out_d);
  if (out_h != -1) size_tensor.push_back(out_h);
  if (out_w != -1) size_tensor.push_back(out_w);

  const DataLayout data_layout = common::StringToDataLayout(data_format);

  if (x.shape().size() == 3) {
    // shape check for 1D interpolate for input tensor shape NCHW
    if (!size_tensor.empty()) {
      // top priority size
      std::vector<symbol::DimExpr> dim_out;
      if (data_layout == DataLayout::kNCHW) {
        dim_out = {x.shape()[0], x.shape()[1], symbol::DimExpr{out_w}};
      } else {
        dim_out = {x.shape()[0], symbol::DimExpr{out_w}, x.shape()[2]};
      }

      symbol::ShapeOrDataDimExprs shape_data{
          symbol::TensorShapeOrDataDimExprs(dim_out)};

      pir::Value res = op->result(0);
      infer_context->SetShapeOrDataForValue(res, shape_data);
      return true;
    }

    symbol::DimExpr out_w_tmp{0};
    const auto &next_sym = infer_context->GetNextSymName();
    out_w_tmp = symbol::DimExpr(next_sym);

    std::vector<symbol::DimExpr> dim_out;
    if (data_layout == DataLayout::kNCHW) {
      dim_out = {x.shape()[0], x.shape()[1], out_w_tmp};
    } else {
      dim_out = {x.shape()[0], out_w_tmp, x.shape()[2]};
    }

    symbol::ShapeOrDataDimExprs shape_data{
        symbol::TensorShapeOrDataDimExprs(dim_out)};

    pir::Value res = op->result(0);
    infer_context->SetShapeOrDataForValue(res, shape_data);
    return true;
  } else if (x.shape().size() == 4) {
    // shape check for 2D interpolate for input tensor shape NCHW
    auto GetOutHW = [&]() -> std::tuple<symbol::DimExpr, symbol::DimExpr> {
      // top priority size
      if (has_size_tensor) {
        const auto &size_tensor_list_shape =
            GetSizeTensorDataExpr(op->operand_source(2));
        PADDLE_ENFORCE_EQ(size_tensor_list_shape.size(),
                          2,
                          common::errors::InvalidArgument(
                              "The size of size_tensor list should be 2."));
        return std::make_tuple(size_tensor_list_shape.at(0),
                               size_tensor_list_shape.at(1));
      }
      // has out_size tensor
      if (op->operand_source(1)) {
        const auto &out_size_data = GetOutSizeDataExpr(op->operand_source(1));
        return std::make_tuple(symbol::DimExpr{out_size_data[0]},
                               symbol::DimExpr{out_size_data[1]});
      }
      // has scale
      if (scale.size() == 2) {
        float scale_h = scale[0];
        float scale_w = scale[1];
        const auto &in_h =
            data_layout == DataLayout::kNCHW ? x.shape()[2] : x.shape()[1];
        const auto &in_w =
            data_layout == DataLayout::kNCHW ? x.shape()[3] : x.shape()[2];
        return std::make_tuple(GetOutDimByScale(in_h, scale_h),
                               GetOutDimByScale(in_w, scale_w));
      }

      return std::make_tuple(symbol::DimExpr{out_h}, symbol::DimExpr{out_w});
    };

    const std::vector<symbol::DimExpr> dim_out = [&] {
      const auto &[out_h_sym, out_w_sym] = GetOutHW();
      if (data_layout == DataLayout::kNCHW) {
        return std::vector<symbol::DimExpr>{
            x.shape()[0], x.shape()[1], out_h_sym, out_w_sym};
      } else {
        return std::vector<symbol::DimExpr>{
            x.shape()[0], out_h_sym, out_w_sym, x.shape()[3]};
      }
    }();

    symbol::ShapeOrDataDimExprs shape_data{
        symbol::TensorShapeOrDataDimExprs(dim_out)};
    infer_context->SetShapeOrDataForValue(op->result(0), shape_data);

    return true;
  } else if (x.shape().size() == 5) {
    auto GetOutDHW =
        [&]() -> std::tuple<symbol::DimExpr, symbol::DimExpr, symbol::DimExpr> {
      // top priority size
      if (has_size_tensor) {
        const auto &size_tensor_list_shape =
            GetSizeTensorDataExpr(op->operand_source(2));
        PADDLE_ENFORCE_EQ(size_tensor_list_shape.size(),
                          3,
                          common::errors::InvalidArgument(
                              "The size of size_tensor list should be 3."));
        return std::make_tuple(size_tensor_list_shape.at(0),
                               size_tensor_list_shape.at(1),
                               size_tensor_list_shape.at(2));
      }
      // has out_size tensor
      if (op->operand_source(1)) {
        const auto &out_size_data = GetOutSizeDataExpr(op->operand_source(1));
        return std::make_tuple(symbol::DimExpr{out_size_data[0]},
                               symbol::DimExpr{out_size_data[1]},
                               symbol::DimExpr{out_size_data[2]});
      }
      // has scale
      if (scale.size() == 3) {
        float scale_d = scale[0];
        float scale_h = scale[1];
        float scale_w = scale[2];
        const auto &in_d =
            data_layout == DataLayout::kNCHW ? x.shape()[2] : x.shape()[1];
        const auto &in_h =
            data_layout == DataLayout::kNCHW ? x.shape()[3] : x.shape()[2];
        const auto &in_w =
            data_layout == DataLayout::kNCHW ? x.shape()[4] : x.shape()[3];
        return std::make_tuple(GetOutDimByScale(in_d, scale_d),
                               GetOutDimByScale(in_h, scale_h),
                               GetOutDimByScale(in_w, scale_w));
      }

      return std::make_tuple(symbol::DimExpr{out_d},
                             symbol::DimExpr{out_h},
                             symbol::DimExpr{out_w});
    };

    const std::vector<symbol::DimExpr> dim_out = [&] {
      const auto &[out_d_sym, out_h_sym, out_w_sym] = GetOutDHW();
      if (data_layout == DataLayout::kNCHW) {
        return std::vector<symbol::DimExpr>{
            x.shape()[0], x.shape()[1], out_d_sym, out_h_sym, out_w_sym};
      } else {
        return std::vector<symbol::DimExpr>{
            x.shape()[0], out_d_sym, out_h_sym, out_w_sym, x.shape()[4]};
      }
    }();

    symbol::ShapeOrDataDimExprs shape_data{
        symbol::TensorShapeOrDataDimExprs(dim_out)};
    infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
    return true;
  } else {
    PADDLE_THROW(
        common::errors::Fatal("Input(X) dimension must be 3, 4 or 5!"));
  }

  return true;
}

bool BilinearOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &y_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const auto &weight_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));

  PADDLE_ENFORCE_EQ(
      x_shape.shape().size(),
      2UL,
      common::errors::InvalidArgument("The input(X) must be a 2D Tensor."));
  PADDLE_ENFORCE_EQ(
      y_shape.shape().size(),
      2UL,
      common::errors::InvalidArgument("The input(Y) must be a 2D Tensor."));
  PADDLE_ENFORCE_EQ(
      weight_shape.shape().size(),
      3UL,
      common::errors::InvalidArgument(
          "Expected the input(Weight) is a 3D tensor. But received %dD tensor.",
          weight_shape.shape().size()));

  infer_context->AddEqualCstr(x_shape.shape()[0], y_shape.shape()[0]);

  infer_context->AddEqualCstr(x_shape.shape()[1], weight_shape.shape()[1]);
  infer_context->AddEqualCstr(y_shape.shape()[1], weight_shape.shape()[2]);

  if (op->operand_source(3)) {  // has bias
    const auto &bias_shape =
        infer_context->GetShapeOrDataForValue(op->operand_source(3));
    PADDLE_ENFORCE_EQ(bias_shape.shape().size(),
                      2UL,
                      common::errors::InvalidArgument(
                          "The Input(Bias) must be a 2-D tensor with "
                          "the 2nd dimension fixed to 1 (a row vector)."));
    infer_context->AddEqualCstr(bias_shape.shape()[0], symbol::DimExpr{1});
    infer_context->AddEqualCstr(bias_shape.shape()[1], weight_shape.shape()[0]);
  }

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(
          {x_shape.shape()[0], weight_shape.shape()[0]})});

  return true;
}

bool BilinearInterpOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return BicubicInterpOpInferSymbolicShape(op, infer_context);
}

bool CrossEntropyWithSoftmaxOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &input_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const symbol::ShapeOrDataDimExprs &index_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  const auto &input_dim = input_shape.shape();
  const auto &index_dim = index_shape.shape();
  const auto &attributes = op->attributes();
  int axis = attributes.at("axis").dyn_cast<pir::Int32Attribute>().data();
  if (axis < 0) axis += input_shape.shape().size();
  bool soft_label =
      attributes.at("soft_label").dyn_cast<pir::BoolAttribute>().data();
  PADDLE_ENFORCE(!soft_label || input_dim.size() == index_dim.size(),
                 common::errors::InvalidArgument(
                     "The input and index should have the same rank when "
                     "soft_label is true. But received input rank(%d) and "
                     "index rank(%d)",
                     input_dim.size(),
                     index_dim.size()));

  auto softmax_dim = index_dim;
  auto out_dim = index_dim;

  if (index_dim.size() == input_dim.size()) {
    if (soft_label) {
      out_dim[axis] = 1;
    }
    softmax_dim[axis] = input_dim[axis];
  } else {
    softmax_dim.insert(softmax_dim.begin() + axis, input_dim[axis]);
    if (soft_label) {
      out_dim.insert(out_dim.begin() + axis, 1);
    }
  }

  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs(softmax_dim));
  infer_context->SetShapeOrDataForValue(
      op->result(1), symbol::TensorShapeOrDataDimExprs(out_dim));

  return true;
}

bool CrossEntropyWithSoftmax_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return CrossEntropyWithSoftmaxOpInferSymbolicShape(op, infer_context);
}

bool ConcatOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  const auto &axis_expr =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  if (!axis_expr.data() || !axis_expr.data()->at(0).isa<int64_t>()) {
    pir::Value res = op->result(0);
    infer_context->SetSymbolForValueByStaticShape(res);
    return true;
  }

  pir::Value operand_source = op->operand_source(0);
  const auto &shape_data_list =
      infer_context->GetShapeOrDataForValue(operand_source)
          .dyn_cast<symbol::TensorListShapeOrDataDimExprs>();

  size_t rank = shape_data_list.at(0).shape().size();
  const int64_t axis = [&] {
    int64_t axis = axis_expr.data()->at(0).dyn_cast<int64_t>();
    return axis >= 0 ? axis : std::max(int64_t(0), int64_t(axis + rank));
  }();

  if (shape_data_list.at(0).data().has_value()) {
    if (rank == 1) {
      const auto &s_or_d =
          infer_context->GetShapeOrDataForValue(operand_source);
      ExprVec data = details::GetExprVecFromData(s_or_d);

      const std::vector<symbol::DimExpr> shape{std::int64_t(data.size())};
      symbol::ShapeOrDataDimExprs shape_data{
          symbol::TensorShapeOrDataDimExprs(shape, data)};
      pir::Value res = op->result(0);
      infer_context->SetShapeOrDataForValue(res, shape_data);

      return true;
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          op->name() +
          " 's InferSymbolicShape can NOT deal with rank > 1 now."));
    }
    std::vector<symbol::DimExpr> data;
    data.reserve(shape_data_list.size());
    for (auto &data_elem : shape_data_list) {
      data.push_back(data_elem.data().value().at(0));
    }
    const std::vector<symbol::DimExpr> shape{std::int64_t(data.size())};
    symbol::ShapeOrDataDimExprs shape_data{
        symbol::TensorShapeOrDataDimExprs(shape, data)};
    pir::Value res = op->result(0);
    infer_context->SetShapeOrDataForValue(res, shape_data);

    return true;
  }

  const std::vector<symbol::DimExpr> &out_dims = [&] {
    std::vector<symbol::DimExpr> out_dims = shape_data_list.at(0).shape();
    for (size_t i = 0; i < rank; ++i) {
      if (i != static_cast<size_t>(axis)) {
        details::BuildCstrEqForTensorListAlongAxis(
            infer_context, shape_data_list, i);
        continue;
      }
      for (size_t j = 1; j < shape_data_list.size(); ++j) {
        out_dims.at(axis) =
            out_dims.at(axis) + shape_data_list.at(j).shape().at(axis);
      }
    }
    return out_dims;
  }();

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_dims)};

  pir::Value res = op->result(0);
  infer_context->SetShapeOrDataForValue(res, shape_data);

  return true;
}

bool FullWithTensorOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(1);
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(operand_source);

  const auto &out_shape = operand_shape_or_data.data().has_value()
                              ? operand_shape_or_data.data().value()
                              : operand_shape_or_data.shape();

  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs(out_shape));
  return true;
}

bool FlashAttnOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(0);
  const symbol::ShapeOrDataDimExprs &q =
      infer_context->GetShapeOrDataForValue(operand_source);

  const symbol::ShapeOrDataDimExprs &k =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  const symbol::ShapeOrDataDimExprs &v =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));

  PADDLE_ENFORCE_EQ(q.shape().size(),
                    4,
                    common::errors::InvalidArgument(
                        "flash_attn receive input with dim "
                        "[batch_size, seq_len, num_heads, head_dim]"));

  infer_context->AddEqualCstr(q.shape()[0], k.shape()[0]);
  infer_context->AddEqualCstr(q.shape()[0], v.shape()[0]);
  infer_context->AddEqualCstr(k.shape()[1], v.shape()[1]);

  if (op->operand_source(4)) {
    const symbol::ShapeOrDataDimExprs &attn_mask =
        infer_context->GetShapeOrDataForValue(op->operand_source(4));
    infer_context->AddEqualCstr(attn_mask.shape()[0], q.shape()[0]);
    infer_context->AddEqualCstr(attn_mask.shape()[2], q.shape()[1]);
    infer_context->AddEqualCstr(attn_mask.shape()[3], k.shape()[1]);
  }

  std::vector<symbol::DimExpr> out_shape = q.shape();

  out_shape.back() = v.shape().back();

  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs(out_shape));

  // GPU has round for seqlen, but XPU has not. Here we align with the GPU
  // version.
  auto round_multiple = [](symbol::DimExpr x) {
    auto m = symbol::DimExpr{128};
    auto m_minus_one = symbol::DimExpr{127};
    return (x + m_minus_one) / m * m;
  };
  auto batch_size_expr = q.shape()[0];
  auto num_heads_expr = q.shape()[2];
  auto seqlen_q_rounded_expr = round_multiple(q.shape()[1]);
  auto seqlen_k_rounded_expr = round_multiple(k.shape()[1]);
  if (op->result(1)) {
    std::vector<symbol::DimExpr> softmax_shape{batch_size_expr,
                                               num_heads_expr,
                                               seqlen_q_rounded_expr,
                                               seqlen_k_rounded_expr};
    infer_context->SetShapeOrDataForValue(
        op->result(1), symbol::TensorShapeOrDataDimExprs(softmax_shape));
  }
  if (op->result(2)) {
    std::vector<symbol::DimExpr> softmax_lse_shape{
        batch_size_expr, num_heads_expr, seqlen_q_rounded_expr};
    infer_context->SetShapeOrDataForValue(
        op->result(2), symbol::TensorShapeOrDataDimExprs(softmax_lse_shape));
  }
  if (op->result(3)) {
    std::vector<symbol::DimExpr> seed_offset_shape{symbol::DimExpr{2}};
    infer_context->SetShapeOrDataForValue(
        op->result(3), symbol::TensorShapeOrDataDimExprs(out_shape));
  }
  return true;
}

bool GroupNormOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));

  infer_context->SetShapeOrDataForValue(op->result(0), x_shape);

  const symbol::DimExpr &batch_size = x_shape.shape()[0];
  int groups = op->attribute<pir::Int32Attribute>("groups").data();
  symbol::TensorShapeOrDataDimExprs mean_shape(
      std::vector<symbol::DimExpr>{batch_size, groups});
  if (op->result(1)) {
    infer_context->SetShapeOrDataForValue(op->result(1), mean_shape);
  }
  if (op->result(2)) {
    infer_context->SetShapeOrDataForValue(op->result(2), mean_shape);
  }
  return true;
}

bool LayerNormOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  // Get the shapes of input tensors
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &scale_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const auto &bias_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));

  std::vector<symbol::DimExpr> x_dims = x_shape_or_data.shape();
  int begin_norm_axis =
      op->attribute<pir::Int32Attribute>("begin_norm_axis").data();

  // Flatten x_dims to 2D and get dim[1]
  symbol::DimExpr matrix_dim_1 = x_dims[begin_norm_axis];
  for (std::size_t i = begin_norm_axis + 1; i < x_dims.size(); ++i) {
    matrix_dim_1 = matrix_dim_1 * x_dims[i];
  }

  if (!scale_shape_or_data.isa<symbol::NullShapeOrDataDimExpr>()) {
    std::vector<symbol::DimExpr> scale_dims = scale_shape_or_data.shape();
    infer_context->AddEqualCstr(scale_dims[0], matrix_dim_1);
  }
  if (!bias_shape_or_data.isa<symbol::NullShapeOrDataDimExpr>()) {
    std::vector<symbol::DimExpr> bias_dims = bias_shape_or_data.shape();
    infer_context->AddEqualCstr(bias_dims[0], matrix_dim_1);
  }

  // Set output shapes
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_dims)});

  // Set mean and variance shapes
  std::vector<symbol::DimExpr> before_norm_dims(
      x_dims.begin(), x_dims.begin() + begin_norm_axis);
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(before_norm_dims)});
  infer_context->SetShapeOrDataForValue(
      op->result(2),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(before_norm_dims)});

  return true;
}

bool LinspaceOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &num_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));
  const auto step = [&] {
    symbol::DimExpr expr;
    if (num_shape_or_data.data().has_value()) {
      expr = num_shape_or_data.data().value()[0];
    } else {
      expr = num_shape_or_data.shape()[0];
    }
    return expr;
  }();
  const symbol::ShapeOrDataDimExprs &shape_data = [&] {
    std::vector<symbol::DimExpr> out_dims{step};
    return symbol::ShapeOrDataDimExprs{
        symbol::TensorShapeOrDataDimExprs(out_dims)};
  }();
  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}

bool LinearInterpOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return BicubicInterpOpInferSymbolicShape(op, infer_context);
}

bool LogspaceOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return LinspaceOpInferSymbolicShape(op, infer_context);
}

bool NearestInterpOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return BicubicInterpOpInferSymbolicShape(op, infer_context);
}

bool MemoryEfficientAttentionOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &q_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  const auto &k_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1)).shape();
  const auto &v_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(2)).shape();
  PADDLE_ENFORCE_EQ(
      q_shape.size(),
      4,
      common::errors::InvalidArgument("Query should be a 4-D tensor"
                                      "But received Query dimension(%d)",
                                      q_shape.size()));
  PADDLE_ENFORCE_EQ(
      k_shape.size(),
      4,
      common::errors::InvalidArgument("Key should be a 4-D tensor"
                                      "But received Key dimension(%d)",
                                      k_shape.size()));
  PADDLE_ENFORCE_EQ(
      v_shape.size(),
      4,
      common::errors::InvalidArgument("Value should be a 4-D tensor"
                                      "But received Value dimension(%d)",
                                      v_shape.size()));

  const auto &query_batch_size = q_shape[0];
  const auto &query_seq_length = q_shape[1];
  const auto &query_num_head = q_shape[2];
  const auto &query_head_size = q_shape[3];

  const auto &key_batch_size = k_shape[0];
  const auto &key_seq_length = k_shape[1];
  const auto &key_num_head = k_shape[2];
  const auto &key_head_size = k_shape[3];

  const auto &value_batch_size = v_shape[0];
  const auto &value_seq_length = v_shape[1];
  const auto &value_num_head = v_shape[2];
  const auto &value_head_size = v_shape[3];

  infer_context->AddEqualCstr(query_batch_size, key_batch_size);
  infer_context->AddEqualCstr(key_batch_size, value_batch_size);

  infer_context->AddEqualCstr(query_num_head, key_num_head);
  infer_context->AddEqualCstr(key_num_head, value_num_head);

  infer_context->AddEqualCstr(query_head_size, key_head_size);

  infer_context->AddEqualCstr(key_seq_length, value_seq_length);

  const std::vector<symbol::DimExpr> out_dims{
      query_batch_size, query_seq_length, query_num_head, value_head_size};
  const std::vector<symbol::DimExpr> logsumexp_dims{query_num_head,
                                                    query_batch_size};
  const std::vector<symbol::DimExpr> seed_and_offset_dims{2};

  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs(out_dims));
  infer_context->SetShapeOrDataForValue(
      op->result(1), symbol::TensorShapeOrDataDimExprs(logsumexp_dims));
  infer_context->SetShapeOrDataForValue(
      op->result(2), symbol::TensorShapeOrDataDimExprs(seed_and_offset_dims));

  return true;
}

bool RoiAlignOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x = op->operand_source(0);
  const auto &boxes = op->operand_source(1);

  const auto &num_boxes =
      infer_context->GetShapeOrDataForValue(boxes).shape()[0];
  symbol::DimExpr channel_num =
      infer_context->GetShapeOrDataForValue(x).shape()[1];

  int32_t out_h = op->attribute<pir::Int32Attribute>("pooled_height").data();
  int32_t out_w = op->attribute<pir::Int32Attribute>("pooled_width").data();

  std::vector<symbol::DimExpr> out_dim = {num_boxes, channel_num, out_h, out_w};
  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs(out_dim));
  return true;
}

bool MeshgridOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::TensorListShapeOrDataDimExprs &shape_data_list =
      infer_context->GetShapeOrDataForValue(op->operand_source(0))
          .dyn_cast<symbol::TensorListShapeOrDataDimExprs>();

  const symbol::ShapeOrDataDimExprs sym_shape_dim_exprs = [&] {
    symbol::TensorListShapeOrDataDimExprs shape_dim_exprs_list;
    std::vector<symbol::DimExpr> vec;

    for (auto &shape_data : shape_data_list) {
      if (shape_data.shape().size() == 0) {
        vec.emplace_back(1);
      } else {
        vec.emplace_back(shape_data.shape()[0]);
      }
    }

    auto shape_dim_exprs = symbol::TensorShapeOrDataDimExprs(vec);

    for (size_t i = 0; i < shape_data_list.size(); i++) {
      shape_dim_exprs_list.emplace_back(shape_dim_exprs);
    }

    return symbol::ShapeOrDataDimExprs(shape_dim_exprs_list);
  }();

  pir::Value res = op->result(0);
  infer_context->SetShapeOrDataForValue(res, sym_shape_dim_exprs);
  return true;
}

bool StackOpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(0);

  const auto &attributes = op->attributes();
  int axis = attributes.at("axis").dyn_cast<pir::Int32Attribute>().data();
  const symbol::TensorListShapeOrDataDimExprs &shape_data_list =
      infer_context->GetShapeOrDataForValue(operand_source)
          .dyn_cast<symbol::TensorListShapeOrDataDimExprs>();

  size_t rank = shape_data_list.at(0).shape().size();
  if (axis < 0) axis += rank + 1;
  const symbol::ShapeOrDataDimExprs shape_data = [&] {
    std::vector<symbol::DimExpr> result_shape = {};
    std::vector<symbol::DimExpr> result_data = {};
    const symbol::TensorShapeOrDataDimExprs &x_shape_data =
        shape_data_list.at(0);

    const bool data_flag = [&] {
      for (const auto &shape_data : shape_data_list) {
        if (!shape_data.data().has_value()) {
          return false;
        }
      }
      return true;
    }();

    if (data_flag) {
      // case 1: data is not empty, eg: shape_data_list =
      // [[shape:{3},data:{S0,6,7}],...]
      if (axis == 0 && x_shape_data.data().value().size() <= 1) {
        for (const auto &shape_data : shape_data_list) {
          result_data.emplace_back(shape_data.data().value().at(0));
        }
      } else {
        PADDLE_THROW(common::errors::Unimplemented(
            op->name() +
            " 's InferSymbolicShape can NOT deal with data size > 1 now."));
      }
      result_shape.emplace_back(
          static_cast<std::int64_t>(shape_data_list.size()));
    } else {
      // case 2: data is empty, eg: shape_data_list =
      // [[shape:{5,6,7},data:{}],...]
      for (size_t i = 0; i < rank; ++i) {
        details::BuildCstrEqForTensorListAlongAxis(
            infer_context, shape_data_list, i);
      }
      for (const symbol::DimExpr &dim : x_shape_data.shape()) {
        result_shape.emplace_back(dim);
      }
      result_shape.insert(result_shape.begin() + axis,
                          static_cast<std::int64_t>(shape_data_list.size()));
    }

    if (result_data.empty()) {
      return symbol::ShapeOrDataDimExprs(
          symbol::TensorShapeOrDataDimExprs(result_shape));
    }
    return symbol::ShapeOrDataDimExprs(
        symbol::TensorShapeOrDataDimExprs(result_shape, result_data));
  }();

  pir::Value res = op->result(0);
  infer_context->SetShapeOrDataForValue(res, shape_data);
  return true;
}

bool TrilinearInterpOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return BicubicInterpOpInferSymbolicShape(op, infer_context);
}

bool WhereOpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      infer_context->GetShapeOrDataForValue(op->operand_source(0)));

  const std::vector<pir::Value> &operands = {op->operand_source(0),
                                             op->operand_source(1)};

  size_t rank = infer_context->GetShapeOrDataForValue(op->operand_source(0))
                    .shape()
                    .size();

  for (size_t i = 0; i < rank; ++i) {
    paddle::dialect::details::BuildCstrEqForTensorListAlongAxis(
        infer_context, operands, i);
  }

  return true;
}

bool Where_OpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  return WhereOpInferSymbolicShape(op, infer_context);
}

bool FakeChannelWiseDequantizeMaxAbsOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));

  int quant_axis = op->attribute<pir::Int32Attribute>("quant_axis").data();
  int x_num_col_dims =
      op->attribute<pir::Int32Attribute>("x_num_col_dims").data();

  PADDLE_ENFORCE_EQ(
      quant_axis == 0 || quant_axis == 1,
      true,
      common::errors::InvalidArgument("'quant_axis' should be 0 or 1, but "
                                      "the received is %d",
                                      quant_axis));
  PADDLE_ENFORCE_EQ(x_num_col_dims == 0,
                    false,
                    common::errors::InvalidArgument(
                        "'x_num_col_dims' should be larger than 0, but "
                        "the received is %d",
                        x_num_col_dims));

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(x_shape_or_data.shape())});

  return true;
}

}  // namespace paddle::dialect
