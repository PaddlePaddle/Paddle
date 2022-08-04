// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/tensorrt/dynamic_shape_infermeta_factory.h"
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/kernels/funcs/unfold_functor.h"

namespace paddle {
namespace inference {
namespace tensorrt {

nvinfer1::DimsExprs UnchangedInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc) {
  CHECK_EQ(nb_inputs, 1);
  return inputs[0];
}

nvinfer1::DimsExprs ScatterNaAddInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc) {
  return inputs[0];
}

nvinfer1::DimsExprs YoloBoxInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc) {
  auto anchors =
      PADDLE_GET_CONST(std::vector<int32_t>, op_desc.GetAttr("anchors"));
  const int anchor_num = anchors.size() / 2;

  const int class_num = PADDLE_GET_CONST(int32_t, op_desc.GetAttr("class_num"));

  // inputs[0].d[2] * inputs[0].d[3] * anchor_num;
  const int box_num = expr_builder
                          .operation(nvinfer1::DimensionOperation::kPROD,
                                     *expr_builder.constant(anchor_num),
                                     *expr_builder.operation(
                                         nvinfer1::DimensionOperation::kPROD,
                                         *inputs[0].d[2],
                                         *inputs[0].d[3]))
                          ->getConstantValue();

  assert(output_index <= 1);

  nvinfer1::DimsExprs output;
  output.nbDims = 3;
  if (output_index == 0) {
    output.d[0] = inputs[0].d[0];
    output.d[1] = expr_builder.constant(box_num);
    output.d[2] = expr_builder.constant(4);
  } else {
    output.d[0] = inputs[0].d[0];
    output.d[1] = expr_builder.constant(box_num);
    output.d[2] = expr_builder.constant(class_num);
  }
  return output;
}

inline void ExpandAspectRatios(const std::vector<float>& input_aspect_ratior,
                               bool flip,
                               std::vector<float>* output_aspect_ratior) {
  constexpr float epsilon = 1e-6;
  output_aspect_ratior->clear();
  output_aspect_ratior->push_back(1.0f);
  for (size_t i = 0; i < input_aspect_ratior.size(); ++i) {
    float ar = input_aspect_ratior[i];
    bool already_exist = false;
    for (size_t j = 0; j < output_aspect_ratior->size(); ++j) {
      if (fabs(ar - output_aspect_ratior->at(j)) < epsilon) {
        already_exist = true;
        break;
      }
    }
    if (!already_exist) {
      output_aspect_ratior->push_back(ar);
      if (flip) {
        output_aspect_ratior->push_back(1.0f / ar);
      }
    }
  }
}

nvinfer1::DimsExprs PriorBoxInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc) {
  nvinfer1::DimsExprs input_dims = inputs[0];
  nvinfer1::DimsExprs output;
  output.nbDims = 4;
  output.d[0] = input_dims.d[2];
  output.d[1] = input_dims.d[3];

  auto aspect_ratios =
      PADDLE_GET_CONST(std::vector<float>, op_desc.GetAttr("aspect_ratios"));
  auto max_sizes =
      PADDLE_GET_CONST(std::vector<float>, op_desc.GetAttr("max_sizes"));
  auto min_sizes =
      PADDLE_GET_CONST(std::vector<float>, op_desc.GetAttr("min_sizes"));
  auto flip = PADDLE_GET_CONST(bool, op_desc.GetAttr("flip"));
  std::vector<float> aspect_ratios_vec;
  ExpandAspectRatios(aspect_ratios, flip, &aspect_ratios_vec);

  size_t num_priors = aspect_ratios_vec.size() * min_sizes.size();
  if (max_sizes.size() > 0) {
    num_priors += max_sizes.size();
  }

  output.d[2] = expr_builder.constant(num_priors);
  output.d[3] = expr_builder.constant(4);
  return output;
}

inline int ConvOutputSize(
    int input_size, int filter_size, int dilation, int padding, int stride) {
  const int dkernel = dilation * (filter_size - 1) + 1;
  int output_size = (input_size + 2 * padding - dkernel) / stride + 1;
  return output_size;
}

nvinfer1::DimsExprs DeformableConvInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc) {
  nvinfer1::DimsExprs output;
  output.nbDims = inputs[0].nbDims;

  nvinfer1::DimsExprs kernel_dims = inputs[2];
  output.d[0] = inputs[0].d[0];
  output.d[1] = kernel_dims.d[0];

  auto paddings =
      PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("paddings"));
  auto strides = PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("strides"));
  auto dilations =
      PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("dilations"));

  output.d[2] =
      expr_builder.constant(ConvOutputSize(inputs[0].d[2]->getConstantValue(),
                                           kernel_dims.d[2]->getConstantValue(),
                                           dilations[0],
                                           paddings[0],
                                           strides[0]));

  output.d[3] =
      expr_builder.constant(ConvOutputSize(inputs[0].d[3]->getConstantValue(),
                                           kernel_dims.d[3]->getConstantValue(),
                                           dilations[1],
                                           paddings[1],
                                           strides[1]));
  return output;
}

nvinfer1::DimsExprs GatherNdInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc) {
  const nvinfer1::DimsExprs x_dims = inputs[0];
  const int x_dims_size = inputs[0].nbDims;
  const nvinfer1::DimsExprs index_dims = inputs[1];
  const int index_dims_size = inputs[1].nbDims;

  std::vector<const nvinfer1::IDimensionExpr*> result_dims;
  // The result dims is
  //   Index.shape[:-1] + X.shape[Index.shape[-1]:]
  for (int i = 0; i < index_dims_size - 1; ++i) {
    result_dims.emplace_back(index_dims.d[i]);
  }
  for (int i = index_dims.d[index_dims_size - 1]->getConstantValue();
       i < x_dims_size;
       ++i) {
    result_dims.emplace_back(x_dims.d[i]);
  }

  nvinfer1::DimsExprs output;
  output.nbDims = result_dims.size();
  for (int i = 0; i < output.nbDims; i++) {
    output.d[i] = result_dims[i];
  }
  return output;
}

nvinfer1::DimsExprs UnfoldInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc) {
  auto kernel_sizes =
      PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("kernel_sizes"));
  auto dilations =
      PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("dilations"));
  auto paddings =
      PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("paddings"));
  auto strides = PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("strides"));

  const nvinfer1::DimsExprs in_dims = inputs[0];
  std::vector<const nvinfer1::IDimensionExpr*> out_dims;
  out_dims.push_back(in_dims.d[0]);
  int output_channels =
      in_dims.d[1]->getConstantValue() * kernel_sizes[0] * kernel_sizes[1];
  out_dims.push_back(expr_builder.constant(output_channels));

  int output_height =
      phi::funcs::CalcOutputSize(in_dims.d[2]->getConstantValue(),
                                 kernel_sizes[0],
                                 dilations[0],
                                 paddings[0],
                                 paddings[2],
                                 strides[0]);
  int output_width =
      phi::funcs::CalcOutputSize(in_dims.d[3]->getConstantValue(),
                                 kernel_sizes[1],
                                 dilations[1],
                                 paddings[1],
                                 paddings[3],
                                 strides[1]);
  int output_col_length = output_height * output_width;
  out_dims.push_back(expr_builder.constant(output_col_length));

  nvinfer1::DimsExprs output;
  output.nbDims = out_dims.size();
  for (int i = 0; i < output.nbDims; i++) {
    output.d[i] = out_dims[i];
  }
  return output;
}

nvinfer1::DimsExprs MatMulInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc) {
  const nvinfer1::DimsExprs x = inputs[0];
  const nvinfer1::DimsExprs y = inputs[1];
  std::vector<int64_t> dims_x;
  std::vector<int64_t> dims_y;

  for (int i = 0; i < x.nbDims; i++) {
    dims_x.push_back(x.d[i]->getConstantValue());
  }
  for (int i = 0; i < y.nbDims; i++) {
    dims_y.push_back(y.d[i]->getConstantValue());
  }

  auto ndims_x = dims_x.size();
  auto ndims_y = dims_y.size();

  PADDLE_ENFORCE_GT(ndims_x,
                    0UL,
                    phi::errors::InvalidArgument(
                        "The Input(x) dims size must be greater than 0,"
                        " but reviced dims size is 0. "));
  PADDLE_ENFORCE_GT(ndims_y,
                    0UL,
                    phi::errors::InvalidArgument(
                        "The Input(y) dims size must be greater than 0,"
                        " but reviced dims size is 0. "));

  bool x_broadcasted = false, y_broadcasted = false;
  if (ndims_x == 1) {
    dims_x.insert(dims_x.begin(), 1);
    ndims_x = 2;
    x_broadcasted = true;
  }

  if (ndims_y == 1) {
    dims_y.push_back(1);
    ndims_y = 2;
    y_broadcasted = true;
  }

  bool trans_x = PADDLE_GET_CONST(bool, op_desc.GetAttr("trans_x"));
  bool trans_y = PADDLE_GET_CONST(bool, op_desc.GetAttr("trans_x"));

  size_t M, N;
  if (trans_x) {
    M = dims_x[ndims_x - 1];
  } else {
    M = dims_x[ndims_x - 2];
  }
  if (trans_y) {
    N = dims_y[ndims_y - 2];
  } else {
    N = dims_y[ndims_y - 1];
  }

  std::vector<int64_t> new_dims;
  if (ndims_x > ndims_y) {
    new_dims.assign(dims_x.begin(), dims_x.end() - 2);
  } else if (ndims_x < ndims_y) {
    new_dims.assign(dims_y.begin(), dims_y.end() - 2);
  } else {
    new_dims.reserve(ndims_x);
    for (size_t i = 0; i < ndims_x - 2; ++i) {
      new_dims.push_back(std::max(dims_x[i], dims_y[i]));
    }
  }
  if (!x_broadcasted) {
    new_dims.push_back(M);
  }
  if (!y_broadcasted) {
    new_dims.push_back(N);
  }
  if (x_broadcasted && y_broadcasted) {
    new_dims.push_back(1);
  }

  nvinfer1::DimsExprs output;
  output.nbDims = new_dims.size();
  for (int i = 0; i < output.nbDims; i++) {
    output.d[i] = expr_builder.constant(new_dims[i]);
  }
  return output;
}

nvinfer1::DimsExprs Pad3dInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc) {
  const nvinfer1::DimsExprs x_dim = inputs[0];
  PADDLE_ENFORCE_EQ(x_dim.nbDims,
                    5,
                    phi::errors::InvalidArgument(
                        "The size of Input(X)'s dimension should be equal to "
                        "5, but received %d. ",
                        x_dim.nbDims));

  std::vector<int64_t> out_dims(x_dim.nbDims, -1);
  out_dims[0] = x_dim.d[0]->getConstantValue();

  auto data_format =
      PADDLE_GET_CONST(std::string, op_desc.GetAttr("data_format"));

  auto paddings =
      PADDLE_GET_CONST(std::vector<int32_t>, op_desc.GetAttr("paddings"));

  if (data_format == "NCDHW") {
    out_dims[1] = x_dim.d[1]->getConstantValue();
  } else {
    out_dims[4] = x_dim.d[4]->getConstantValue();
  }

  PADDLE_ENFORCE_EQ(paddings.size(),
                    6,
                    phi::errors::InvalidArgument(
                        "Shape of Input(Paddings) should be equal to "
                        "[6], but received [%d].",
                        paddings.size()));
  if (data_format == "NCDHW") {
    out_dims[2] = x_dim.d[2]->getConstantValue() + paddings[4] + paddings[5];
    out_dims[3] = x_dim.d[3]->getConstantValue() + paddings[2] + paddings[3];
    out_dims[4] = x_dim.d[4]->getConstantValue() + paddings[0] + paddings[1];
  } else {
    out_dims[1] = x_dim.d[1]->getConstantValue() + paddings[4] + paddings[5];
    out_dims[2] = x_dim.d[2]->getConstantValue() + paddings[2] + paddings[3];
    out_dims[3] = x_dim.d[3]->getConstantValue() + paddings[0] + paddings[1];
  }

  nvinfer1::DimsExprs output;
  output.nbDims = out_dims.size();
  for (int i = 0; i < output.nbDims; i++) {
    output.d[i] = expr_builder.constant(out_dims[i]);
  }
  return output;
}

nvinfer1::DimsExprs TileInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc) {
#define MAX_RANK_SUPPORTED 6

  std::vector<int> x_dims(inputs[0].nbDims);

  for (size_t i = 0; i < x_dims.size(); i++) {
    x_dims[i] = inputs[0].d[i]->getConstantValue();
  }

  auto repeat_times_data =
      PADDLE_GET_CONST(std::vector<int32_t>, op_desc.GetAttr("repeat_times"));

  if (repeat_times_data.size() == 0) {
    repeat_times_data = std::vector<int32_t>(x_dims.size(), -1);
  }

  PADDLE_ENFORCE_LE(
      x_dims.size(),
      MAX_RANK_SUPPORTED,
      phi::errors::InvalidArgument(
          "The rank of the input 'x' for tile op "
          "must not be greater than %d, but the value received is %d.",
          MAX_RANK_SUPPORTED,
          x_dims.size()));
  PADDLE_ENFORCE_LE(
      repeat_times_data.size(),
      MAX_RANK_SUPPORTED,
      phi::errors::InvalidArgument(
          "The size of the shape of input 'repeat_times' for tile op "
          "must not be greater than %d, but the value received is %d.",
          MAX_RANK_SUPPORTED,
          repeat_times_data.size()));
  PADDLE_ENFORCE_GE(
      repeat_times_data.size(),
      1,
      phi::errors::InvalidArgument(
          "The size of the shape of input 'repeat_times' for tile op "
          "must be positive integers, but the value received is %d.",
          repeat_times_data.size()));

  auto out_rank =
      std::max(static_cast<size_t>(x_dims.size()), repeat_times_data.size());

  std::vector<int64_t> out_shape(out_rank);
  if (x_dims.size() > repeat_times_data.size()) {
    auto diff = x_dims.size() - repeat_times_data.size();
    repeat_times_data.insert(repeat_times_data.begin(), diff, -1);
  } else {
    auto diff = repeat_times_data.size() - x_dims.size();
    x_dims.insert(x_dims.begin(), diff, -1);
  }
  for (size_t i = 0; i < repeat_times_data.size(); ++i) {
    if (x_dims[i] == -1 || repeat_times_data[i] == -1) {
      out_shape[i] = -1;
    } else {
      PADDLE_ENFORCE_GT(
          repeat_times_data[i],
          0,
          phi::errors::InvalidArgument(
              "Every element of the input 'repeat_times' for tile op must be "
              "greater than 0, but the value given is %d.",
              repeat_times_data[i]));
      out_shape[i] = x_dims[i] * repeat_times_data[i];
    }
  }

  nvinfer1::DimsExprs output;
  output.nbDims = out_shape.size();
  for (int i = 0; i < output.nbDims; i++) {
    output.d[i] = expr_builder.constant(out_shape[i]);
  }
  return output;
}

PD_REGISTER_DYNAMIC_INFER_META_FN(assign, UnchangedInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(scatter_nd_add, ScatterNaAddInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(gather_nd, GatherNdInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(yolo_box, YoloBoxInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(prior_box, PriorBoxInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(deformable_conv, DeformableConvInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(unfold, UnfoldInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(matmul_v2, MatMulInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(silu, UnchangedInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(reciprocal, UnchangedInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(pad3d, Pad3dInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(tile, TileInferMeta);
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
