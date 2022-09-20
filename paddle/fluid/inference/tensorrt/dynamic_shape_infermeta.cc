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

PD_REGISTER_DYNAMIC_INFER_META_FN(gather_nd, GatherNdInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(yolo_box, YoloBoxInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(instance_norm, InstanceNormInferMeta);

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
