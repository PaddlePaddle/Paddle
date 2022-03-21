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

#include <math.h>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "paddle/fluid/inference/tensorrt/plugin/split_op_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"
#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/infrt/backends/tensorrt/trt_engine.h"
#include "paddle/infrt/backends/tensorrt/trt_options.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/meta_tensor.h"

namespace infrt {
namespace backends {
namespace tensorrt {

const char* model_input = "input_0";
const char* model_output = "output_0";
const char* model_output2 = "output_1";

TrtUniquePtr<nvinfer1::INetworkDefinition> ConstructNetwork(
    nvinfer1::IBuilder* builder, nvinfer1::Dims dims, bool is_static_shape) {
  TrtUniquePtr<nvinfer1::INetworkDefinition> network;
  if (is_static_shape) {
    network.reset(builder->createNetworkV2(0U));
  } else {
    auto networkFlags =
        1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    network.reset(builder->createNetworkV2(networkFlags));
  }

  ITensor* data =
      network->addInput(model_input, nvinfer1::DataType::kFLOAT, dims);
  CHECK_NOTNULL(data);
  IActivationLayer* act =
      network->addActivation(*data, ActivationType::kSIGMOID);
  CHECK_NOTNULL(act);
  auto* act_out = act->getOutput(0);
  std::vector<int> output_length{1, 2};
  int axis;
  nvinfer1::IPluginV2Layer* split_layer;
  if (is_static_shape) {
    axis = 0;
    paddle::inference::tensorrt::plugin::SplitPlugin plugin(
        axis, output_length, false);
    split_layer = network->addPluginV2(&act_out, 1, plugin);
  } else {
    axis = 1;
    paddle::inference::tensorrt::plugin::SplitPluginDynamic plugin(
        axis, output_length, false);
    split_layer = network->addPluginV2(&act_out, 1, plugin);
  }

  split_layer->getOutput(0)->setName(model_output);
  split_layer->getOutput(1)->setName(model_output2);
  network->markOutput(*split_layer->getOutput(0));
  network->markOutput(*split_layer->getOutput(1));
  return network;
}

TrtUniquePtr<nvinfer1::INetworkDefinition> ConstructFCNetwork(
    nvinfer1::IBuilder* builder, nvinfer1::Dims dims, bool is_static_shape) {
  TrtUniquePtr<nvinfer1::INetworkDefinition> network;
  if (is_static_shape) {
    network.reset(builder->createNetworkV2(0U));
  } else {
    auto networkFlags =
        1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    network.reset(builder->createNetworkV2(networkFlags));
  }

  ITensor* data =
      network->addInput(model_input, nvinfer1::DataType::kFLOAT, dims);
  CHECK_NOTNULL(data);
  nvinfer1::Weights kernel_weights;
  kernel_weights.type = nvinfer1::DataType::kFLOAT;
  kernel_weights.count = 7840;
  std::vector<float> weight_data(kernel_weights.count);
  for (size_t i = 0; i < weight_data.size(); ++i) {
    weight_data[i] = i % 255 * 0.02f;
  }
  kernel_weights.values = weight_data.data();
  auto* layer = network->addFullyConnected(
      *data, 10, kernel_weights, nvinfer1::Weights{});
  CHECK_NOTNULL(layer);
  auto* out = layer->getOutput(0);
  out->setName(model_output);
  network->markOutput(*out);
  return network;
}

TrtUniquePtr<nvinfer1::INetworkDefinition> ConstructConvNetwork(
    nvinfer1::IBuilder* builder, nvinfer1::Dims dims, bool is_static_shape) {
  TrtUniquePtr<nvinfer1::INetworkDefinition> network;
  if (is_static_shape) {
    network.reset(builder->createNetworkV2(0U));
  } else {
    auto networkFlags =
        1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    network.reset(builder->createNetworkV2(networkFlags));
  }

  ITensor* data =
      network->addInput(model_input, nvinfer1::DataType::kFLOAT, dims);
  CHECK_NOTNULL(data);
  nvinfer1::Weights kernel_weights, bias_weights;
  kernel_weights.type = nvinfer1::DataType::kFLOAT;
  bias_weights.type = nvinfer1::DataType::kFLOAT;
  kernel_weights.count = 81;
  bias_weights.count = 3;
  std::vector<float> weight_data(kernel_weights.count);
  for (size_t i = 0; i < weight_data.size(); ++i) {
    weight_data[i] = i * 0.02f;
  }
  std::vector<float> bias_data(bias_weights.count);
  for (size_t i = 0; i < bias_data.size(); ++i) {
    bias_data[i] = i * 0.5f;
  }
  kernel_weights.values = weight_data.data();
  bias_weights.values = bias_data.data();
  nvinfer1::Dims ksize;
  ksize.nbDims = 2;
  ksize.d[0] = 3;
  ksize.d[1] = 3;
  auto* layer =
      network->addConvolutionNd(*data, 3, ksize, kernel_weights, bias_weights);
  CHECK_NOTNULL(layer);
  auto* out = layer->getOutput(0);
  out->setName(model_output);
  network->markOutput(*out);
  return network;
}

// sigmoid(x) = 1 / (1 + exp(-x))
inline float sigmoid(float x) { return 1.f / (1.f + exp(-1 * x)); }

TEST(trt, run_fc_static) {
  TrtEngine engine(0);
  auto net = ConstructFCNetwork(
      engine.GetTrtBuilder(), nvinfer1::Dims3{1, 28, 28}, true);
  BuildOptions build_options;
  build_options.max_batch = 4;
  build_options.workspace = 1024;
  engine.Build(std::move(net), build_options);

  InferenceOptions inference_options;
  inference_options.batch = 1;

  phi::GPUPlace place;
  phi::GPUContext context;
  context.PartialInitWithoutAllocator();
  context.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(place, context.stream())
                           .get());
  context.PartialInitWithAllocator();

  phi::DenseTensorMeta meta(
      phi::DataType::FLOAT32,
      phi::make_ddim({inference_options.batch, 1, 28, 28}));
  phi::DenseTensor input;
  input.set_meta(meta);
  context.Alloc<float>(&input, input.numel() * sizeof(float));
  std::vector<float> host_data(inference_options.batch * 1 * 28 * 28, 0);
  for (size_t i = 0; i < host_data.size(); ++i) {
    host_data[i] = i % 100 * 0.016f;
  }
  paddle::memory::Copy(place,
                       input.data<float>(),
                       phi::CPUPlace(),
                       host_data.data(),
                       sizeof(float) * host_data.size(),
                       context.stream());

  std::unordered_map<std::string, phi::DenseTensor*> inputs;
  inputs.emplace(std::make_pair(model_input, &input));
  engine.PrepareOutputHandle("output_0");
  engine.SetUpInference(inference_options, inputs);
  engine.GetEngineInfo();
  engine.Run(context);
  cudaStreamSynchronize(context.stream());
}

TEST(trt, run_conv_static) {
  TrtEngine engine(0);
  auto net = ConstructConvNetwork(
      engine.GetTrtBuilder(), nvinfer1::Dims3{3, 28, 28}, true);
  BuildOptions build_options;
  build_options.max_batch = 4;
  build_options.workspace = 1024;
  engine.Build(std::move(net), build_options);

  InferenceOptions inference_options;
  inference_options.batch = 1;

  phi::GPUPlace place;
  phi::GPUContext context;
  context.PartialInitWithoutAllocator();
  context.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(place, context.stream())
                           .get());
  context.PartialInitWithAllocator();

  phi::DenseTensorMeta meta(
      phi::DataType::FLOAT32,
      phi::make_ddim({inference_options.batch, 3, 28, 28}));
  phi::DenseTensor input;
  input.set_meta(meta);
  context.Alloc<float>(&input, input.numel() * sizeof(float));
  std::vector<float> host_data(inference_options.batch * 3 * 28 * 28, 0);
  for (size_t i = 0; i < host_data.size(); ++i) {
    host_data[i] = i % 100 * 0.016f;
  }
  paddle::memory::Copy(place,
                       input.data<float>(),
                       phi::CPUPlace(),
                       host_data.data(),
                       sizeof(float) * host_data.size(),
                       context.stream());

  std::unordered_map<std::string, phi::DenseTensor*> inputs;
  inputs.emplace(std::make_pair(model_input, &input));
  engine.PrepareOutputHandle("output_0");
  engine.SetUpInference(inference_options, inputs);
  engine.GetEngineInfo();
  engine.Run(context);
  cudaStreamSynchronize(context.stream());
}

TEST(trt, run_static) {
  TrtEngine static_trt_engine(0);
  auto net = ConstructNetwork(
      static_trt_engine.GetTrtBuilder(), nvinfer1::Dims3{3, 28, 28}, true);
  BuildOptions static_build_options;
  static_build_options.max_batch = 4;
  static_trt_engine.Build(std::move(net), static_build_options);
  InferenceOptions inference_options;
  inference_options.batch = 2;

  phi::GPUPlace place;
  phi::GPUContext context;
  context.PartialInitWithoutAllocator();
  context.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(place, context.stream())
                           .get());
  context.PartialInitWithAllocator();

  phi::DenseTensorMeta meta(
      phi::DataType::FLOAT32,
      phi::make_ddim({inference_options.batch, 3, 28, 28}));
  phi::DenseTensor input;
  input.set_meta(meta);
  context.Alloc<float>(&input, input.numel() * sizeof(float));
  std::vector<float> host_data(inference_options.batch * 3 * 28 * 28, 0);
  for (size_t i = 0; i < host_data.size(); ++i) {
    host_data[i] = i % 100 * 0.016f;
  }
  paddle::memory::Copy(place,
                       input.data<float>(),
                       phi::CPUPlace(),
                       host_data.data(),
                       sizeof(float) * host_data.size(),
                       context.stream());

  std::unordered_map<std::string, phi::DenseTensor*> inputs;
  inputs.emplace(std::make_pair(model_input, &input));
  static_trt_engine.PrepareOutputHandle("output_0");
  static_trt_engine.PrepareOutputHandle("output_1");
  static_trt_engine.SetUpInference(inference_options, inputs);
  static_trt_engine.GetEngineInfo();
  static_trt_engine.Run(context);

  phi::DenseTensor* output0 = static_trt_engine.GetOutput("output_0");
  phi::DenseTensor* output1 = static_trt_engine.GetOutput("output_1");
  std::vector<float> output_data1(inference_options.batch * 1 * 28 * 28, 0);
  std::vector<float> output_data2(inference_options.batch * 2 * 28 * 28, 0);
  paddle::memory::Copy(phi::CPUPlace(),
                       output_data1.data(),
                       place,
                       output0->data<float>(),
                       sizeof(float) * output_data1.size(),
                       context.stream());
  paddle::memory::Copy(phi::CPUPlace(),
                       output_data2.data(),
                       place,
                       output1->data<float>(),
                       sizeof(float) * output_data2.size(),
                       context.stream());
  cudaStreamSynchronize(context.stream());

  for (size_t i = 0; i < host_data.size(); ++i) {
    int w = i % 28;
    int h = (i / 28) % 28;
    int c = i / (28 * 28) % 3;
    int n = i / (28 * 28 * 3);
    if (c == 0) {
      CHECK_NEAR(
          sigmoid(host_data[i]), output_data1[n * 28 * 28 + h * 28 + w], 1e-5);
    } else {
      CHECK_NEAR(sigmoid(host_data[i]),
                 output_data2[n * 28 * 28 * 2 + (c - 1) * 28 * 28 + h * 28 + w],
                 1e-5);
    }
  }
}

TEST(trt, run_dynamic) {
  TrtEngine engine(0);
  auto net = ConstructNetwork(
      engine.GetTrtBuilder(), nvinfer1::Dims4{-1, 3, -1, -1}, false);
  BuildOptions build_options;
  build_options.max_batch = 4;
  build_options.workspace = 32;
  // build_options.fp16 = true;
  std::vector<int32_t> min_shape{1, 3, 16, 16};
  std::vector<int32_t> opt_shape{2, 3, 28, 28};
  std::vector<int32_t> max_shape{4, 3, 28, 28};
  build_options.shapes[model_input][0] = min_shape;
  build_options.shapes[model_input][1] = opt_shape;
  build_options.shapes[model_input][2] = max_shape;
  engine.Build(std::move(net), build_options);

  InferenceOptions inference_options;
  inference_options.batch = 2;

  phi::GPUPlace place;
  phi::GPUContext context;
  context.PartialInitWithoutAllocator();
  context.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(place, context.stream())
                           .get());
  context.PartialInitWithAllocator();

  phi::DenseTensorMeta meta(
      phi::DataType::FLOAT32,
      phi::make_ddim({inference_options.batch, 3, 16, 16}));
  phi::DenseTensor input, output, output2;
  input.set_meta(meta);
  context.Alloc<float>(&input, input.numel() * sizeof(float));
  std::vector<float> host_data(inference_options.batch * 3 * 16 * 16, 0);
  for (size_t i = 0; i < host_data.size(); ++i) {
    host_data[i] = i % 100 * 0.016f;
  }
  paddle::memory::Copy(place,
                       input.data<float>(),
                       phi::CPUPlace(),
                       host_data.data(),
                       sizeof(float) * host_data.size(),
                       context.stream());

  std::unordered_map<std::string, phi::DenseTensor*> inputs;
  inputs.emplace(std::make_pair(model_input, &input));
  engine.PrepareOutputHandle("output_0");
  engine.PrepareOutputHandle("output_1");
  engine.SetUpInference(inference_options, inputs);
  engine.GetEngineInfo();
  engine.Run(context);
  phi::DenseTensor* output0 = engine.GetOutput("output_0");
  phi::DenseTensor* output1 = engine.GetOutput("output_1");

  std::vector<float> output_data1(inference_options.batch * 1 * 16 * 16, 0);
  std::vector<float> output_data2(inference_options.batch * 2 * 16 * 16, 0);
  paddle::memory::Copy(phi::CPUPlace(),
                       output_data1.data(),
                       place,
                       output0->data<float>(),
                       sizeof(float) * output_data1.size(),
                       context.stream());
  paddle::memory::Copy(phi::CPUPlace(),
                       output_data2.data(),
                       place,
                       output1->data<float>(),
                       sizeof(float) * output_data2.size(),
                       context.stream());
  cudaStreamSynchronize(context.stream());

  for (size_t i = 0; i < host_data.size(); ++i) {
    int w = i % 16;
    int h = (i / 16) % 16;
    int c = i / (16 * 16) % 3;
    int n = i / (16 * 16 * 3);
    if (c == 0) {
      CHECK_NEAR(
          sigmoid(host_data[i]), output_data1[n * 16 * 16 + h * 16 + w], 1e-5);
    } else {
      CHECK_NEAR(sigmoid(host_data[i]),
                 output_data2[n * 16 * 16 * 2 + (c - 1) * 16 * 16 + h * 16 + w],
                 1e-5);
    }
  }
}

}  // namespace tensorrt
}  // namespace backends
}  // namespace infrt
