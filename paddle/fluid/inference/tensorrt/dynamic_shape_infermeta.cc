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

  if (index_dims.d[index_dims_size - 1]->isConstant()) {
    for (int i = index_dims.d[index_dims_size - 1]->getConstantValue();
         i < x_dims_size;
         ++i) {
      result_dims.emplace_back(x_dims.d[i]);
    }
  }

  nvinfer1::DimsExprs output;
  output.nbDims = result_dims.size();
  for (int i = 0; i < output.nbDims; i++) {
    output.d[i] = result_dims[i];
  }
  return output;
}

nvinfer1::DimsExprs YoloBoxInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc) {
  PADDLE_ENFORCE_EQ(
      nb_inputs,
      2,
      phi::errors::InvalidArgument("inputs of yolo_box should be equal to 2, "
                                   "But received (%s)",
                                   nb_inputs));

  const nvinfer1::DimsExprs dim_x = inputs[0];

  auto anchors = PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("anchors"));
  int anchor_num = anchors.size() / 2;

  // box_num = dim_x[2] * dim_x[3] * anchor_num;
  const nvinfer1::IDimensionExpr* box_num = expr_builder.operation(
      nvinfer1::DimensionOperation::kPROD,
      *expr_builder.operation(
          nvinfer1::DimensionOperation::kPROD, *dim_x.d[2], *dim_x.d[3]),
      *expr_builder.constant(anchor_num));

  nvinfer1::DimsExprs output;
  output.nbDims = 3;
  if (output_index == 0) {
    output.d[0] = dim_x.d[0];
    output.d[1] = box_num;
    output.d[2] = expr_builder.constant(4);
  } else {
    auto class_num = PADDLE_GET_CONST(int, op_desc.GetAttr("class_num"));
    output.d[0] = dim_x.d[0];
    output.d[1] = box_num;
    output.d[2] = expr_builder.constant(class_num);
  }
  return output;
}

nvinfer1::DimsExprs InstanceNormInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc) {
  nvinfer1::DimsExprs x_dims = inputs[0];
  return x_dims;
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

nvinfer1::DimsExprs UnflodInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc) {
  PADDLE_ENFORCE_EQ(
      nb_inputs,
      1,
      phi::errors::InvalidArgument("inputs of unfold should be equal to 1, "
                                   "But received (%s)",
                                   nb_inputs));

  const nvinfer1::DimsExprs in_dims = inputs[0];
  std::vector<const nvinfer1::IDimensionExpr*> out_dims;
  out_dims.push_back(in_dims.d[0]);

  auto kernel_sizes =
      PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("kernel_sizes"));
  auto dilations =
      PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("dilations"));
  auto paddings =
      PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("paddings"));
  auto strides = PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("strides"));

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
  nvinfer1::DimsExprs output;
  output.nbDims = out_dims.size();
  for (size_t i = 0; i < out_dims.size(); i++) output.d[i] = out_dims[i];
  return output;
}

nvinfer1::DimsExprs ScatterNdAddInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc) {
  PADDLE_ENFORCE_EQ(nb_inputs,
                    3,
                    phi::errors::InvalidArgument(
                        "inputs of scatter_nd_add should be equal to 3, "
                        "But received (%s)",
                        nb_inputs));
  const nvinfer1::DimsExprs ref_dims = inputs[0];
  return ref_dims;
}

nvinfer1::DimsExprs UnchangedInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc) {
  PADDLE_ENFORCE_EQ(nb_inputs,
                    1,
                    phi::errors::InvalidArgument(
                        "inputs of UnchangedInferMeta should be equal to 1, "
                        "But received (%s)",
                        nb_inputs));
  return inputs[0];
}

nvinfer1::DimsExprs Pad3dInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc) {
  const nvinfer1::DimsExprs x_dim = inputs[0];

  nvinfer1::DimsExprs out_dims;
  out_dims.nbDims = x_dim.nbDims;

  out_dims.d[0] = x_dim.d[0];

  auto paddings =
      PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("paddings"));
  auto data_format =
      PADDLE_GET_CONST(std::string, op_desc.GetAttr("data_format"));

  if (data_format == "NCDHW") {
    out_dims.d[1] = x_dim.d[1];
  } else {
    out_dims.d[4] = x_dim.d[4];
  }

  if (data_format == "NCDHW") {
    // depth
    out_dims.d[2] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                *x_dim.d[2],
                                *expr_builder.constant(paddings[4])),
        *expr_builder.constant(paddings[5]));
    // height
    out_dims.d[3] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                *x_dim.d[3],
                                *expr_builder.constant(paddings[2])),
        *expr_builder.constant(paddings[3]));
    // width
    out_dims.d[4] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                *x_dim.d[4],
                                *expr_builder.constant(paddings[0])),
        *expr_builder.constant(paddings[1]));
  } else {  // NDHWC
    // depth
    out_dims.d[1] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                *x_dim.d[1],
                                *expr_builder.constant(paddings[4])),
        *expr_builder.constant(paddings[5]));
    // height
    out_dims.d[2] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                *x_dim.d[2],
                                *expr_builder.constant(paddings[2])),
        *expr_builder.constant(paddings[3]));
    // width
    out_dims.d[3] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                *x_dim.d[3],
                                *expr_builder.constant(paddings[0])),
        *expr_builder.constant(paddings[1]));
  }
  return out_dims;
}

nvinfer1::DimsExprs PNormInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc) {
  const nvinfer1::DimsExprs x_dim = inputs[0];
  std::vector<const nvinfer1::IDimensionExpr*> reduce_dims;
  std::vector<const nvinfer1::IDimensionExpr*> keep_dims;

  bool asvector = PADDLE_GET_CONST(bool, op_desc.GetAttr("asvector"));
  bool keepdim = PADDLE_GET_CONST(bool, op_desc.GetAttr("keepdim"));
  int axis = PADDLE_GET_CONST(int, op_desc.GetAttr("axis"));

  if (asvector) {
    reduce_dims.emplace_back(expr_builder.constant(1));
    keep_dims.emplace_back(expr_builder.constant(1));
    if (keepdim) {
      for (int i = 1; i < x_dim.nbDims; ++i) {
        keep_dims.emplace_back(expr_builder.constant(1));
      }
    }
  } else {
    if (axis < 0) axis = x_dim.nbDims + axis;
    for (int i = 0; i < x_dim.nbDims; ++i) {
      if (i != axis) reduce_dims.emplace_back(x_dim.d[i]);
    }
    if (reduce_dims.size() == 0) {
      reduce_dims.emplace_back(expr_builder.constant(1));
    }
  }
  keep_dims[axis] = expr_builder.constant(1);

  nvinfer1::DimsExprs output;
  if (keepdim) {
    output.nbDims = keep_dims.size();
    for (int i = 0; i < output.nbDims; i++) output.d[i] = keep_dims[i];
  } else {
    output.nbDims = reduce_dims.size();
    for (int i = 0; i < output.nbDims; i++) output.d[i] = reduce_dims[i];
  }
  return output;
}

nvinfer1::DimsExprs GridSamplerInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc) {
  const nvinfer1::DimsExprs x_dims = inputs[0];
  const nvinfer1::DimsExprs grid_dims = inputs[1];

  nvinfer1::DimsExprs output;
  if (grid_dims.nbDims == 4) {
    output.nbDims = 4;
    output.d[0] = x_dims.d[0];
    output.d[1] = x_dims.d[1];
    output.d[2] = grid_dims.d[1];
    output.d[3] = grid_dims.d[2];
  } else {
    output.nbDims = 4;
    output.d[0] = x_dims.d[0];
    output.d[1] = x_dims.d[1];
    output.d[2] = grid_dims.d[1];
    output.d[3] = grid_dims.d[2];
    output.d[4] = grid_dims.d[3];
  }
  return output;
}

PD_REGISTER_DYNAMIC_INFER_META_FN(gather_nd, GatherNdInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(yolo_box, YoloBoxInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(instance_norm, InstanceNormInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(unfold, UnflodInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(scatter_nd_add, ScatterNdAddInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(inverse, UnchangedInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(pad3d, Pad3dInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(grid_sampler, GridSamplerInferMeta);
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
