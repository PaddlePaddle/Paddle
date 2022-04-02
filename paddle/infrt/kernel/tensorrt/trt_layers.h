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
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

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
  nvinfer1::Weights bias_weights;
  if (op.bias_weights() == mlir::Value()) {
    bias_weights = nvinfer1::Weights{};
  } else {
    bias_weights = TensorToWeights(value_to_tensor_map[op.bias_weights()]);
  }

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

inline void PoolFunc(trt::PoolingOp& op,  // NOLINT
                     nvinfer1::INetworkDefinition* network,
                     ValueToITensorMap& value_to_trt_tensor_map,  // NOLINT
                     ValueToTensorMap& value_to_tensor_map) {     // NOLINT
  mlir::Value input_tensor_repr = op.input_tensor();
  nvinfer1::ITensor* input_itensor = value_to_trt_tensor_map[input_tensor_repr];
  // nvinfer1::Dims input_shape = input_itensor->getDimensions();
  // int input_dims = input_shape.nbDims;

  auto padding_mode = op.padding_mode();
  auto pool_type = op.pool_type();
  mlir::ArrayAttr paddings = op.paddings();
  mlir::ArrayAttr strides = op.strides();
  mlir::ArrayAttr ksize = op.window_size();
  bool exclusive = op.exclusive();
  bool adaptive = op.adaptive();
  auto padding_algorithm = op.padding_algorithm().str();

  if (padding_algorithm == "SAME") {
    // TODO(wilber)
    CHECK(false) << "Not supported `same` padding algorithm";
  }

  if (adaptive) {
    // TODO(Inference)
    CHECK(false) << "Not supported adaptive pool";
  }

  nvinfer1::Dims window_size = ArrayAttrToNvDims(ksize);

  auto* layer =
      network->addPoolingNd(*input_itensor,
                            static_cast<nvinfer1::PoolingType>(pool_type),
                            window_size);
  CHECK_NOTNULL(layer);
  layer->setPaddingMode(static_cast<nvinfer1::PaddingMode>(padding_mode));
  layer->setPaddingNd(ArrayAttrToNvDims(paddings));
  layer->setStrideNd(ArrayAttrToNvDims(strides));
  layer->setAverageCountExcludesPadding(exclusive);

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

inline void ShuffleFunc(trt::ShuffleOp& op,  // NOLINT
                     nvinfer1::INetworkDefinition* network,
                     ValueToITensorMap& value_to_trt_tensor_map,  // NOLINT
                     ValueToTensorMap& value_to_tensor_map) {     // NOLINT
  mlir::Value input_tensor_repr = op.input_tensor();
  nvinfer1::ITensor* input = value_to_trt_tensor_map[input_tensor_repr];
  int dims = input->getDimensions().nbDims;

  int start_axis = op.start_axisAttr().getInt();
  int stop_axis = op.start_axisAttr().getInt();

  nvinfer1::IShuffleLayer* layer = nullptr;
  if (start_axis < 0) start_axis += dims + 1;
  if (stop_axis < 0) stop_axis += dims + 1;
  int dim_prod = 1;
  nvinfer1::Dims flatten_dim;
  flatten_dim.nbDims = dims - (stop_axis - start_axis);
  for (int i = 0, j = 0; i < dims; ++i) {
    if (start_axis <= i + 1 && i + 1 <= stop_axis) {
      int dim_i = input->getDimensions().d[i];
      CHECK_GT(dim_i, 0);
      dim_prod *= dim_i;
      if (i + 1 == stop_axis) {
        flatten_dim.d[j++] = dim_prod;
      }
    } else {
      flatten_dim.d[j++] = input->getDimensions().d[i];
    }
  }
  layer = network->addShuffle(*value_to_trt_tensor_map[input_tensor_repr]);
  CHECK_NOTNULL(layer);
  layer->setReshapeDimensions(flatten_dim);
  for (size_t i = 0; i < op->getNumResults(); ++i) {
    nvinfer1::ITensor* out_tensor = layer->getOutput(i);
    mlir::Value out_value = op->getResult(i);
    value_to_trt_tensor_map[out_value] = out_tensor;
  }
}

inline void ScaleNdFunc(trt::ScaleNdOp& op,  // NOLINT
                     nvinfer1::INetworkDefinition* network,
                     ValueToITensorMap& value_to_trt_tensor_map,  // NOLINT
                     ValueToTensorMap& value_to_tensor_map) {     // NOLINT
  mlir::Value input_tensor_repr = op.input_tensor();
  nvinfer1::ITensor* input = value_to_trt_tensor_map[input_tensor_repr];

  mlir::Value shift_tensor_repr = op.shift();
  nvinfer1::Weights shift = TensorToWeights(value_to_tensor_map[shift_tensor_repr]);
  
  mlir::Value scale_tensor_repr = op.scale();

  nvinfer1::Weights scale = TensorToWeights(value_to_tensor_map[scale_tensor_repr]);

  nvinfer1::Weights power_weights{nvinfer1::DataType::kFLOAT, nullptr,0};

  nvinfer1::IScaleLayer* layer = nullptr;
  layer = network->addScaleNd(*input, nvinfer1::ScaleMode::kCHANNEL, shift, scale, power_weights, 0);
  CHECK_NOTNULL(layer);

  for (size_t i = 0; i < op->getNumResults(); ++i) {
    nvinfer1::ITensor* out_tensor = layer->getOutput(i);
    mlir::Value out_value = op->getResult(i);
    value_to_trt_tensor_map[out_value] = out_tensor;
  }
}
}  // namespace tensorrt
}  // namespace kernel
}  // namespace infrt
