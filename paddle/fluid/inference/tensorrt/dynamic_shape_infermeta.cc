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

#include "paddle/fluid/inference/tensorrt/dynamic_shape_infermeta.h"
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace inference {
namespace tensorrt {

nvinfer1::DimsExprs IndexSelectInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc_) {
  nvinfer1::DimsExprs output(inputs[0]);
  int dim = PADDLE_GET_CONST(int, op_desc_.GetAttr("dim"));
  output.d[dim] = expr_builder.constant(inputs[1].d[0]->getConstantValue());
  return output;
}

nvinfer1::DimsExprs RollInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc_) {
  return inputs[0];
}

nvinfer1::DimsExprs SliceInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc_) {
  std::vector<int> axes_ =
      PADDLE_GET_CONST(std::vector<int>, op_desc_.GetAttr("axes"));
  std::vector<int> starts_ =
      PADDLE_GET_CONST(std::vector<int>, op_desc_.GetAttr("starts"));
  std::vector<int> ends_ =
      PADDLE_GET_CONST(std::vector<int>, op_desc_.GetAttr("ends"));
  std::vector<int> decrease_axises_ =
      PADDLE_GET_CONST(std::vector<int>, op_desc_.GetAttr("decrease_axis"));
  int decrease_axis_ = decrease_axises_.size() == 0 ? -1 : decrease_axises_[0];
  auto in_dims = inputs[0];
  nvinfer1::DimsExprs ret = in_dims;
  // start, ends should greater 0
  for (size_t i = 0; i < axes_.size(); i++) {
    int start = starts_[i];
#if IS_TRT_VERSION_GE(7200)
    ret.d[axes_[i]] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUB,
        *expr_builder.operation(nvinfer1::DimensionOperation::kMIN,
                                *expr_builder.constant(ends_[i]),
                                *in_dims.d[axes_[i]]),
        *expr_builder.constant(start));
#else
    int end = ends_[i];
    ret.d[axes_[i]] = expr_builder.constant(end - start);
#endif
  }
  if (decrease_axis_ != -1) {
    nvinfer1::DimsExprs res;
    res.nbDims = ret.nbDims - 1;
    int j = 0;
    for (int i = 0; i < in_dims.nbDims; i++) {
      if (decrease_axis_ == i) continue;
      res.d[j++] = expr_builder.operation(nvinfer1::DimensionOperation::kMAX,
                                          *expr_builder.constant(0),
                                          *ret.d[i]);
    }
    return res;
  }
  return ret;
}

nvinfer1::DimsExprs Pool2dInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc_) {
  bool is_global_ = PADDLE_GET_CONST(bool, op_desc_.GetAttr("global_pooling"));
  bool adaptive_ = PADDLE_GET_CONST(bool, op_desc_.GetAttr("adaptive"));

  std::vector<int> ksize_ =
      PADDLE_GET_CONST(std::vector<int>, op_desc_.GetAttr("ksize"));
  std::vector<int> strides_ =
      PADDLE_GET_CONST(std::vector<int>, op_desc_.GetAttr("strides"));
  std::vector<int> paddings_ =
      PADDLE_GET_CONST(std::vector<int>, op_desc_.GetAttr("paddings"));
  bool ceil_mode_ = PADDLE_GET_CONST(bool, op_desc_.GetAttr("ceil_mode"));

  nvinfer1::DimsExprs output(inputs[0]);
  if (is_global_ && !adaptive_) {
    output.d[2] = expr_builder.constant(1);
    output.d[3] = expr_builder.constant(1);
    return output;
  }
  if (is_global_ && adaptive_) {
    return inputs[0];
  }
  if (adaptive_) {
    output.d[2] = expr_builder.constant(ksize_[0]);
    output.d[3] = expr_builder.constant(ksize_[1]);
    return output;
  }

  auto stri_0 = expr_builder.constant(strides_[0]);
  auto stri_1 = expr_builder.constant(strides_[1]);
  auto one_value = expr_builder.constant(1);

  auto v0_tmp = expr_builder.constant(-ksize_[0] + 2 * paddings_[0]);
  auto v1_tmp = expr_builder.constant(-ksize_[1] + 2 * paddings_[1]);

  auto ceil_tmp =
      expr_builder.constant(-ksize_[0] + 2 * paddings_[0] + strides_[0] - 1);
  auto ceil1_tmp =
      expr_builder.constant(-ksize_[1] + 2 * paddings_[1] + strides_[1] - 1);

  if (!ceil_mode_) {
    output.d[2] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(
            nvinfer1::DimensionOperation::kFLOOR_DIV,
            *expr_builder.operation(
                nvinfer1::DimensionOperation::kSUM, *inputs[0].d[2], *v0_tmp),
            *stri_0),
        *one_value);
    output.d[3] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(
            nvinfer1::DimensionOperation::kFLOOR_DIV,
            *expr_builder.operation(
                nvinfer1::DimensionOperation::kSUM, *inputs[0].d[3], *v1_tmp),
            *stri_1),
        *one_value);

  } else {
    output.d[2] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(
            nvinfer1::DimensionOperation::kFLOOR_DIV,
            *expr_builder.operation(
                nvinfer1::DimensionOperation::kSUM, *inputs[0].d[2], *ceil_tmp),
            *stri_0),
        *one_value);
    output.d[3] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(
            nvinfer1::DimensionOperation::kFLOOR_DIV,
            *expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                    *inputs[0].d[3],
                                    *ceil1_tmp),
            *stri_1),
        *one_value);
  }
  return output;
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
