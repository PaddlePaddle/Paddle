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
PD_REGISTER_DYNAMIC_INFER_META_FN(gather_nd, GatherNdInferMeta);
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
