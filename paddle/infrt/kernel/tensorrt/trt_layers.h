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

#pragma once

#include <NvInfer.h>
#include <mlir/IR/Operation.h>

#include <string>

#include "paddle/infrt/dialect/tensorrt/trt_ops.h"
#include "paddle/infrt/kernel/tensorrt/trt_helper.h"

#include "paddle/phi/core/dense_tensor.h"

namespace infrt {
namespace kernel {
namespace tensorrt {

using ValueToTensorMap = llvm::DenseMap<mlir::Value, phi::DenseTensor*>;
using ValueToITensorMap = llvm::DenseMap<mlir::Value, nvinfer1::ITensor*>;

inline void ActivationFunc(
    trt::ActivationOp& act_op,  // NOLINT
    nvinfer1::INetworkDefinition* network,
    ValueToITensorMap& value_to_trt_tensor_map,  // NOLINT
    ValueToTensorMap& value_to_tensor_map) {     // NOLINT
  auto in_arg = act_op.getOperand();
  CHECK(value_to_trt_tensor_map.count(in_arg))
      << "value_to_trt_tensor_map not has in_arg.";

  nvinfer1::ActivationType act_type =
      static_cast<nvinfer1::ActivationType>(act_op.activation_type());
  auto* act_layer =
      network->addActivation(*value_to_trt_tensor_map[in_arg], act_type);
  act_layer->setAlpha(act_op.alpha().convertToFloat());
  act_layer->setBeta(act_op.beta().convertToFloat());
  for (size_t i = 0; i < act_op->getNumResults(); ++i) {
    nvinfer1::ITensor* act_out_tensor = act_layer->getOutput(i);
    mlir::Value act_out = act_op->getResult(i);
    value_to_trt_tensor_map[act_out] = act_out_tensor;
  }
}

inline void ConvFunc(trt::ConvolutionOp& op,  // NOLINT
                     nvinfer1::INetworkDefinition* network,
                     ValueToITensorMap& value_to_trt_tensor_map,  // NOLINT
                     ValueToTensorMap& value_to_tensor_map) {     // NOLINT
  mlir::Value input_tensor_repr = op.input_tensor();
  int out_channel_num = op.out_channel_num();
  auto size_attrs = op.kernel_size();
  nvinfer1::Dims dims = ArrayAttrToNvDims(size_attrs);
  auto kernel_weights =
      TensorToWeights(value_to_tensor_map[op.kernel_weights()]);
  auto bias_weights = TensorToWeights(value_to_tensor_map[op.bias_weights()]);

  auto* layer =
      network->addConvolutionNd(*value_to_trt_tensor_map[input_tensor_repr],
                                out_channel_num,
                                dims,
                                kernel_weights,
                                bias_weights);
  CHECK_NOTNULL(layer);
  mlir::Value out_repr = op.output_tensor();
  nvinfer1::ITensor* out_tensor = layer->getOutput(0);
  value_to_trt_tensor_map[out_repr] = out_tensor;
}

inline void FcFunc(trt::FullyConnectedOp& op,  // NOLINT
                   nvinfer1::INetworkDefinition* network,
                   ValueToITensorMap& value_to_trt_tensor_map,  // NOLINT
                   ValueToTensorMap& value_to_tensor_map) {     // NOLINT
  mlir::Value input_tensor_repr = op.input_tensor();
  CHECK(value_to_trt_tensor_map.count(input_tensor_repr));

  auto kernel_weights =
      TensorToWeights(value_to_tensor_map[op.kernel_weights()]);
  auto bias_weights = TensorToWeights(value_to_tensor_map[op.bias_weights()]);

  int out_channel_num = op.out_channel_num();
  auto* layer =
      network->addFullyConnected(*value_to_trt_tensor_map[input_tensor_repr],
                                 out_channel_num,
                                 kernel_weights,
                                 bias_weights);

  mlir::Value out_repr = op.output_tensor();
  nvinfer1::ITensor* out_tensor = layer->getOutput(0);
  value_to_trt_tensor_map[out_repr] = out_tensor;
}
}  // namespace tensorrt
}  // namespace kernel
}  // namespace infrt
