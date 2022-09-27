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

#include "paddle/infrt/backends/tensorrt/plugin/pool_op_plugin.h"
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

  layer->setPaddingNd(ArrayAttrToNvDims(op.paddings()));
  layer->setStrideNd(ArrayAttrToNvDims(op.strides()));
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
  nvinfer1::Dims input_shape = input_itensor->getDimensions();
  int input_dims = input_shape.nbDims;

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
    // CHECK(false) << "Not supported adaptive pool";

    // TODO(wilber): Reformat.
    // global average pooling.
    auto ksize_vec = ArrayAttrToVec<int>(ksize);
    if (static_cast<nvinfer1::PoolingType>(pool_type) ==
            nvinfer1::PoolingType::kAVERAGE &&
        ksize_vec.size() == 2 && ksize_vec[0] == 1 && ksize_vec[1] == 1) {
      nvinfer1::Dims dims;
      dims.nbDims = 2;
      dims.d[0] = input_shape.d[1];
      dims.d[1] = input_shape.d[2];
      auto* layer = network->addPoolingNd(
          *input_itensor, static_cast<nvinfer1::PoolingType>(pool_type), dims);
      CHECK_NOTNULL(layer);

      mlir::Value out_repr = op.output_tensor();
      nvinfer1::ITensor* out_tensor = layer->getOutput(0);
      value_to_trt_tensor_map[out_repr] = out_tensor;
      return;
    }

    // plugin...
    std::vector<int> input_shape_v;
    for (int i = 0; i < input_dims; i++) {
      input_shape_v.push_back(input_shape.d[i]);
    }
    auto paddings_val = ArrayAttrToVec<int>(paddings);
    std::vector<int> real_paddings = paddings_val;
    for (int i = 0; i < 2; ++i) {
      int copy_pad = *(paddings_val.begin() + i);
      real_paddings.insert(real_paddings.begin() + 2 * i + 1, copy_pad);
    }

    auto* plugin = new backends::tensorrt::plugin::PoolPlugin(
        false,
        backends::tensorrt::plugin::PoolPlugin::PoolType::avg,
        adaptive,
        exclusive,
        ArrayAttrToVec<int>(ksize),
        ArrayAttrToVec<int>(strides),
        paddings_val,
        input_shape_v,
        real_paddings);
    auto* layer = network->addPluginV2(&input_itensor, 1, *plugin);

    mlir::Value out_repr = op.output_tensor();
    nvinfer1::ITensor* out_tensor = layer->getOutput(0);
    value_to_trt_tensor_map[out_repr] = out_tensor;
    return;
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

  nvinfer1::ITensor* input_itensor = value_to_trt_tensor_map[input_tensor_repr];
  nvinfer1::Dims input_shape = input_itensor->getDimensions();
  int input_dims = input_shape.nbDims;
  CHECK_EQ(input_dims, 1) << "Now we only support 2-d input.";
  // TODO(wilber): We should place the logic to ir.  Now only support 2-d input
  // and we reshape to 4-d.
  nvinfer1::Dims reshape_before_fc_dim;
  reshape_before_fc_dim.nbDims = input_dims + 2;
  // padding shape "* x q x 1 x 1"
  for (int i = 0; i < reshape_before_fc_dim.nbDims; i++) {
    reshape_before_fc_dim.d[i] = 1;
  }
  reshape_before_fc_dim.d[0] = input_shape.d[0];
  auto* reshape_before_fc_layer = network->addShuffle(*input_itensor);
  reshape_before_fc_layer->setReshapeDimensions(reshape_before_fc_dim);

  auto* reshape_itensor = reshape_before_fc_layer->getOutput(0);

  auto kernel_weights =
      TensorToWeights(value_to_tensor_map[op.kernel_weights()]);
  auto bias_weights = TensorToWeights(value_to_tensor_map[op.bias_weights()]);

  int out_channel_num = op.out_channel_num();
  auto* layer = network->addFullyConnected(
      *reshape_itensor, out_channel_num, kernel_weights, bias_weights);

  // TODO(wilber): fix.
  nvinfer1::Dims reshape_after_fc_dim;
  reshape_after_fc_dim.nbDims = 1;
  reshape_after_fc_dim.d[0] = layer->getOutput(0)->getDimensions().d[0];
  auto* reshape_after_fc_layer = network->addShuffle(*layer->getOutput(0));
  reshape_after_fc_layer->setReshapeDimensions(reshape_after_fc_dim);

  mlir::Value out_repr = op.output_tensor();
  nvinfer1::ITensor* out_tensor = reshape_after_fc_layer->getOutput(0);
  value_to_trt_tensor_map[out_repr] = out_tensor;
}

inline void ShuffleFunc(trt::ShuffleOp& op,  // NOLINT
                        nvinfer1::INetworkDefinition* network,
                        ValueToITensorMap& value_to_trt_tensor_map,  // NOLINT
                        ValueToTensorMap& value_to_tensor_map) {     // NOLINT
  mlir::Value input_tensor_repr = op.input_tensor();
  nvinfer1::ITensor* input = value_to_trt_tensor_map[input_tensor_repr];
  int dims = input->getDimensions().nbDims;
  int start_axis = op.start_axis();
  int stop_axis = op.stop_axis();

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
  nvinfer1::Weights shift =
      TensorToWeights(value_to_tensor_map[shift_tensor_repr]);

  mlir::Value scale_tensor_repr = op.scale();

  nvinfer1::Weights scale =
      TensorToWeights(value_to_tensor_map[scale_tensor_repr]);

  nvinfer1::Weights power_weights{nvinfer1::DataType::kFLOAT, nullptr, 0};

  nvinfer1::IScaleLayer* layer = nullptr;
  layer = network->addScaleNd(
      *input, nvinfer1::ScaleMode::kCHANNEL, shift, scale, power_weights, 0);
  CHECK_NOTNULL(layer);

  for (size_t i = 0; i < op->getNumResults(); ++i) {
    nvinfer1::ITensor* out_tensor = layer->getOutput(i);
    mlir::Value out_value = op->getResult(i);
    value_to_trt_tensor_map[out_value] = out_tensor;
  }
}

inline void EltwiseFunc(trt::ElementWiseOp& op,  // NOLINT
                        nvinfer1::INetworkDefinition* network,
                        ValueToITensorMap& value_to_trt_tensor_map,  // NOLINT
                        ValueToTensorMap& value_to_tensor_map) {     // NOLINT
  mlir::Value input1_tensor_repr = op.input1();
  mlir::Value input2_tensor_repr = op.input2();
  nvinfer1::ITensor* input1 = value_to_trt_tensor_map[input1_tensor_repr];
  nvinfer1::ITensor* input2 = value_to_trt_tensor_map[input2_tensor_repr];

  auto eltwise_operation = op.elementwise_operation();

  auto* layer = network->addElementWise(
      *input1,
      *input2,
      static_cast<nvinfer1::ElementWiseOperation>(eltwise_operation));
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
