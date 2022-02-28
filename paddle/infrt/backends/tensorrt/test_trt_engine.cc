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
#include "glog/logging.h"
#include "gtest/gtest.h"
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

const char* model_input = "model_input";
const char* model_output = "model_output1";
const char* model_output2 = "model_output2";

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

// sigmoid(x) = 1 / (1 + exp(-x))
inline float sigmoid(float x) { return 1.f / (1.f + exp(-1 * x)); }

TEST(trt, run_static) {
  TRTEngine static_trt_engine(0);
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
  phi::DenseTensor output, output2;
  std::unordered_map<std::string, phi::DenseTensor*> outputs;
  outputs.emplace(std::make_pair(model_output, &output));
  outputs.emplace(std::make_pair(model_output2, &output2));

  static_trt_engine.SetUpInference(inference_options, inputs, &outputs);
  static_trt_engine.GetEngineInfo();
  static_trt_engine.Run(context);

  std::vector<float> output_data1(inference_options.batch * 1 * 28 * 28, 0);
  std::vector<float> output_data2(inference_options.batch * 2 * 28 * 28, 0);
  paddle::memory::Copy(phi::CPUPlace(),
                       output_data1.data(),
                       place,
                       output.data<float>(),
                       sizeof(float) * output_data1.size(),
                       context.stream());
  paddle::memory::Copy(phi::CPUPlace(),
                       output_data2.data(),
                       place,
                       output2.data<float>(),
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
  TRTEngine engine(0);
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
  std::unordered_map<std::string, phi::DenseTensor*> outputs;
  inputs.emplace(std::make_pair(model_input, &input));
  outputs.emplace(std::make_pair(model_output, &output));
  outputs.emplace(std::make_pair(model_output2, &output2));

  engine.SetUpInference(inference_options, inputs, &outputs);
  engine.GetEngineInfo();
  engine.Run(context);

  std::vector<float> output_data1(inference_options.batch * 1 * 16 * 16, 0);
  std::vector<float> output_data2(inference_options.batch * 2 * 16 * 16, 0);
  paddle::memory::Copy(phi::CPUPlace(),
                       output_data1.data(),
                       place,
                       output.data<float>(),
                       sizeof(float) * output_data1.size(),
                       context.stream());
  paddle::memory::Copy(phi::CPUPlace(),
                       output_data2.data(),
                       place,
                       output2.data<float>(),
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
