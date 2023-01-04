// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#include <NvInferRuntime.h>

#include "paddle/infrt/backends/tensorrt/trt_options.h"
#include "paddle/infrt/backends/tensorrt/trt_utils.h"
#include "paddle/phi/backends/dynload/tensorrt.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"

namespace infrt {
namespace backends {
namespace tensorrt {
using namespace nvinfer1;  // NOLINT

// The trt programing model as follows:
// 1. The build phase:
// IBuilder* builder = createInferBuilder(&logger_);
// 2. Create a network definition:
// INetworkDefinition* network = builder->createNetworkV2(...);
// 3. Build network:
// network->AddLayer(...)
// 4. Configure network:
// IBuilderConfig* config = builder->createBuilderConfig();
// config->setMaxWorkspaceSize(...)
// 5. Get cuda engine and deserializing a plan:
// IHostMemory* serialized_model = builder->buildSerializedNetwork(...);
// IRuntime* runtime = createInferRuntime(&logger_);
// ICudaEngine* engine = runtime->deserializeCudaEngine(...);
// 6. Get execution context:
// IExecutionContext* exec_context = engine->createExecutionContext();
// 7. Set input data:
// int32_t input_index = engine->getBindingIndex("input");
// int32_t output_index = engine->getBindingIndex("output");
// void* buffers[2];
// buffers[input_index] = input_buffer;
// buffers[output_index] = output_buffer;
// 8. Performance inference:
// exec_context->enqueueV2(buffers, stream, nullptr);
//
// We have encapsulated this logic, please use the following programming model.
//
// TrtEngine trt_engine;
// trt_engine.Build(...);
// trt_engine.SetUpInference(...);
// trt_engine.Run(...);
class TrtEngine {
 public:
  explicit TrtEngine(int device_id = 0);

  TrtEngine(const TrtEngine&) = delete;
  TrtEngine& operator=(const TrtEngine&) = delete;
  TrtEngine(TrtEngine&&) = default;
  TrtEngine& operator=(TrtEngine&&) = default;

  nvinfer1::IBuilder* GetTrtBuilder();

  // TODO(wilber): Modify signature after infrt-trt ready.
  void Build(TrtUniquePtr<nvinfer1::INetworkDefinition> network,
             const BuildOptions& build_options);

  // TODO(wilber): Modify signature after infrt-trt ready.
  void Run(const ::phi::GPUContext& ctx);

  // TODO(wilber): How to support multiple execution contexts?
  bool SetUpInference(const InferenceOptions& inference,
                      const std::unordered_map<std::string, ::Tensor*>& inputs);

  void GetEngineInfo();

  void PrepareOutputHandle(const std::string& out_name);

  // TODO(wilber): The output tensor names are: output_0, output_1, ...
  ::Tensor* GetOutput(const std::string&);

  size_t GetOutputNum() const;

 private:
  void FreshDeviceId();

  bool SetupNetworkAndConfig(const BuildOptions& build,
                             INetworkDefinition& network,  // NOLINT
                             IBuilderConfig& config);      // NOLINT

  bool NetworkToEngine(const BuildOptions& build);

  bool ModelToBuildEnv(TrtUniquePtr<nvinfer1::INetworkDefinition> network,
                       const BuildOptions& build);

  void StaticRun(const ::phi::GPUContext& ctx);

  void DynamicRun(const ::phi::GPUContext& ctx);

 private:
  std::unique_ptr<TrtLogger> logger_{nullptr};
  TrtUniquePtr<nvinfer1::IBuilder> builder_{nullptr};
  TrtUniquePtr<INetworkDefinition> network_{nullptr};
  std::unique_ptr<IHostMemory> serialized_engine_{nullptr};
  TrtUniquePtr<nvinfer1::ICudaEngine> engine_{nullptr};
  std::vector<TrtUniquePtr<nvinfer1::IExecutionContext>> contexts_;
  std::vector<std::unique_ptr<Bindings>> bindings_;
  int device_id_{0};
  bool is_dynamic_shape_{false};
  std::unordered_map<std::string, ::Tensor> outputs_;
};

}  // namespace tensorrt
}  // namespace backends
}  // namespace infrt
