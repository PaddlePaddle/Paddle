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

nvinfer1::DimsExprs AssignInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc_) {
  return inputs[0];
}

PD_REGISTER_DYNAMIC_INFER_META_FN(index_select, IndexSelectInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(assign, AssignInferMeta);
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
