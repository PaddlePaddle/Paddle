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

#include "paddle/infrt/backends/tensorrt/trt_engine.h"

#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>
#include <glog/logging.h>
#include "paddle/phi/backends/dynload/tensorrt.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/dense_tensor.h"

namespace infrt {
namespace backends {
namespace tensorrt {

// The following two API are implemented in TensorRT's header file, cannot load
// from the dynamic library. So create our own implementation and directly
// trigger the method from the dynamic library.
static nvinfer1::IBuilder* createInferBuilder(
    nvinfer1::ILogger& logger) {  // NOLINT
  return static_cast<nvinfer1::IBuilder*>(
      ::phi::dynload::createInferBuilder_INTERNAL(&logger,
                                                  NV_TENSORRT_VERSION));
}
static nvinfer1::IRuntime* createInferRuntime(
    nvinfer1::ILogger& logger) {  // NOLINT
  return static_cast<nvinfer1::IRuntime*>(
      ::phi::dynload::createInferRuntime_INTERNAL(&logger,
                                                  NV_TENSORRT_VERSION));
}

TrtEngine::TrtEngine(int device_id) : device_id_(device_id) {
  FreshDeviceId();
  logger_.reset(new TrtLogger());
  builder_.reset(createInferBuilder(logger_->GetTrtLogger()));
  ::phi::dynload::initLibNvInferPlugins(&logger_->GetTrtLogger(), "");
}

nvinfer1::IBuilder* TrtEngine::GetTrtBuilder() {
  CHECK_NOTNULL(builder_);
  return builder_.get();
}

void TrtEngine::Build(TrtUniquePtr<nvinfer1::INetworkDefinition> network,
                      const BuildOptions& build_options) {
  FreshDeviceId();
  ModelToBuildEnv(std::move(network), build_options);
  CHECK_NOTNULL(engine_);
}

bool TrtEngine::ModelToBuildEnv(
    TrtUniquePtr<nvinfer1::INetworkDefinition> network,
    const BuildOptions& build) {
  CHECK_NOTNULL(builder_);
  std::swap(network, network_);
  CHECK_NOTNULL(network_);
  // ModelToNetwork(network_, logger);
  NetworkToEngine(build);
  return true;
}

bool TrtEngine::NetworkToEngine(const BuildOptions& build) {
  TrtUniquePtr<IBuilderConfig> config{builder_->createBuilderConfig()};
  CHECK_NOTNULL(config);
  CHECK(SetupNetworkAndConfig(build, *network_, *config));

#if IS_TRT_VERSION_LT(8000)
  engine_.reset(builder_->buildEngineWithConfig(*network_, *config));
#else
  serialized_engine_.reset(
      builder_->buildSerializedNetwork(*network_, *config));
  CHECK_NOTNULL(serialized_engine_);

  TrtUniquePtr<IRuntime> runtime{createInferRuntime(logger_->GetTrtLogger())};
  CHECK_NOTNULL(runtime);
  engine_.reset(runtime->deserializeCudaEngine(serialized_engine_->data(),
                                               serialized_engine_->size()));
  CHECK_NOTNULL(engine_);
#endif
  return true;
}

bool TrtEngine::SetupNetworkAndConfig(const BuildOptions& build,
                                      INetworkDefinition& network,
                                      IBuilderConfig& config) {
  builder_->setMaxBatchSize(build.max_batch);
  // TODO(wilber): handle one engine - multi execution context case.
  IOptimizationProfile* profile{nullptr};
  if (!build.shapes.empty()) {
    profile = builder_->createOptimizationProfile();
    CHECK_NOTNULL(profile);
  }

  // Set formats and data types of inputs
  for (int32_t i = 0; i < network.getNbInputs(); ++i) {
    auto* input = network.getInput(i);
    if (!build.input_formats.empty()) {
      input->setType(build.input_formats[i].first);
      input->setAllowedFormats(build.input_formats[i].second);
    } else {
      switch (input->getType()) {
        case DataType::kINT32:
        case DataType::kBOOL:
        case DataType::kHALF:
          // Leave these as is.
          break;
        case DataType::kFLOAT:
        case DataType::kINT8:
          // User did not specify a floating-point format.  Default to kFLOAT.
          input->setType(DataType::kFLOAT);
          break;
      }
      input->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
    }

    if (profile) {
      Dims dims = input->getDimensions();
      // TODO(wilber): shape tensor.
      const bool is_dynamic_input = std::any_of(
          dims.d, dims.d + dims.nbDims, [](int dim) { return dim == -1; });
      if (is_dynamic_input) {
        is_dynamic_shape_ = true;
        auto shape = build.shapes.find(input->getName());

        // If no shape is provided
        if (shape == build.shapes.end()) {
          // TODO(wilber): add infomation.
          CHECK(false);
        }
        LOG(INFO) << "Run Paddle-TRT Dynamic Shape mode.";
        std::vector<int> profile_dims{};
        profile_dims =
            shape->second[static_cast<size_t>(OptProfileSelector::kMIN)];
        CHECK(profile->setDimensions(input->getName(),
                                     OptProfileSelector::kMIN,
                                     VecToDims(profile_dims)));
        profile_dims =
            shape->second[static_cast<size_t>(OptProfileSelector::kOPT)];
        CHECK(profile->setDimensions(input->getName(),
                                     OptProfileSelector::kOPT,
                                     VecToDims(profile_dims)));
        profile_dims =
            shape->second[static_cast<size_t>(OptProfileSelector::kMAX)];
        CHECK(profile->setDimensions(input->getName(),
                                     OptProfileSelector::kMAX,
                                     VecToDims(profile_dims)));
      }
    }
  }

  if (profile && is_dynamic_shape_) {
    CHECK(profile->isValid());  // Required optimization profile is invalid
    CHECK_NE(config.addOptimizationProfile(profile), -1);
  }

  // Set formats and data types of outputs
  for (int32_t i = 0, n = network.getNbOutputs(); i < n; i++) {
    auto* output = network.getOutput(i);
    if (!build.output_formats.empty()) {
      // int outputFormatIndex = broadcastOutputFormats ? 0 : i;
      output->setType(build.output_formats[i].first);
      output->setAllowedFormats(build.output_formats[i].second);
    } else {
      output->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
    }
  }

  config.setMaxWorkspaceSize(static_cast<size_t>(build.workspace) << 20);

  if (build.fp16) {
    config.setFlag(BuilderFlag::kFP16);
    bool support_fp16 = builder_->platformHasFastFp16();
    if (support_fp16) {
      LOG(INFO) << "Run INFRT-TRT FP16 mode";
    } else {
      LOG(INFO) << "You specify FP16 mode, but the hardware do not support "
                   "FP16 speed up, use FP32 instead.";
    }
  }

  if (build.tf32) {
    config.setFlag(BuilderFlag::kTF32);
    bool support_tf32 = builder_->platformHasTf32();
    if (support_tf32) {
      LOG(INFO) << "Run INFRT-TRT TF32 mode";
    } else {
      LOG(INFO) << "You specify TF32 mode, but the hardware do not support "
                   "TF32 speed up, use FP32 instead.";
    }
  }

  // TODO(wilber): other precision.

  // TODO(wilber): precision config.
  switch (build.precision_constraints) {
    case PrecisionConstraints::kNONE:
      // It's the default for TensorRT.
      break;
    case PrecisionConstraints::kOBEY:
      config.setFlag(BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
      break;
    case PrecisionConstraints::kPREFER:
      config.setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
      break;
  }

  // TODO(TRT): DLA config.

  // TODO(TRT): int8 config.
  // TODO(TRT): support int8
  if (build.int8) {
    assert(false);
    config.setFlag(BuilderFlag::kINT8);
    bool support_int8 = builder_->platformHasFastInt8();
    if (support_int8) {
      LOG(INFO) << "Run INFRT-TRT FP16 mode";
    }
  }

  // TODO(TRT): calib config.

  // TODO(TRT): sparse config.

  return true;
}

void TrtEngine::PrepareOutputHandle(const std::string& out_name) {
  ::phi::DenseTensor t;
  outputs_.emplace(out_name, t);
}

::phi::DenseTensor* TrtEngine::GetOutput(const std::string& name) {
  return &outputs_[name];
}

size_t TrtEngine::GetOutputNum() const { return outputs_.size(); }

bool TrtEngine::SetUpInference(
    const InferenceOptions& inference,
    const std::unordered_map<std::string, ::phi::DenseTensor*>& inputs) {
  // TODO(wilber): now only create one exec_context
  FreshDeviceId();
  CHECK(engine_ != nullptr);
  nvinfer1::IExecutionContext* ec = engine_->createExecutionContext();
  CHECK(ec != nullptr);
  contexts_.emplace_back(ec);
  bindings_.emplace_back(new Bindings());

  for (const auto& it : inputs) {
    const int bind_index = engine_->getBindingIndex(it.first.c_str());
    bindings_.front()->AddBinding(
        bind_index, it.first, true, it.second, nvinfer1::DataType::kFLOAT);
  }
  for (auto& it : outputs_) {
    const int bind_index = engine_->getBindingIndex(it.first.c_str());
    bindings_.front()->AddBinding(
        bind_index, it.first, false, &it.second, nvinfer1::DataType::kFLOAT);
  }

  return true;
}

void TrtEngine::Run(const ::phi::GPUContext& ctx) {
  if (is_dynamic_shape_) {
    DynamicRun(ctx);
  } else {
    StaticRun(ctx);
  }
}

void TrtEngine::StaticRun(const ::phi::GPUContext& ctx) {
  const int num_bindings = engine_->getNbBindings();
  std::vector<void*> buffers(num_bindings, nullptr);

  int runtime_batch = -1;
  auto input_binds = bindings_.front()->GetInputBindings();
  for (auto bind : input_binds) {
    const int bind_index = engine_->getBindingIndex(bind.name.c_str());
    buffers[bind_index] =
        const_cast<void*>(static_cast<const void*>(bind.buffer->data<float>()));
    if (runtime_batch != -1) {
      CHECK_EQ(runtime_batch,
               ::phi::vectorize<int64_t>(bind.buffer->dims())[0]);
    }
    runtime_batch = bind.buffer->dims()[0];
  }

  auto output_binds = bindings_.front()->GetOutputBindings();
  for (auto bind : output_binds) {
    const int bind_index = engine_->getBindingIndex(bind.name.c_str());
    std::vector<int32_t> ddim;
    auto dims = engine_->getBindingDimensions(bind_index);
    CHECK_NE(runtime_batch, -1) << "runtime_batch should not be -1.";
    ddim.push_back(runtime_batch);
    for (int i = 0; i < dims.nbDims; ++i) {
      ddim.push_back(dims.d[i]);
    }
    bind.buffer->Resize(::phi::make_ddim(ddim));
    // TODO(wilber): now only support float output.
    ctx.Alloc<float>(bind.buffer, sizeof(float) * bind.buffer->numel());
    buffers[bind_index] = static_cast<void*>(bind.buffer->data<float>());
  }

  contexts_.front()->enqueue(
      runtime_batch, buffers.data(), ctx.stream(), nullptr);
}

void TrtEngine::DynamicRun(const ::phi::GPUContext& ctx) {
  const int num_bindings = engine_->getNbBindings();
  std::vector<void*> buffers(num_bindings, nullptr);

  auto input_binds = bindings_.front()->GetInputBindings();
  for (auto bind : input_binds) {
    const int bind_index = engine_->getBindingIndex(bind.name.c_str());
    buffers[bind_index] =
        const_cast<void*>(static_cast<const void*>(bind.buffer->data<float>()));
    nvinfer1::Dims trt_dims;
    trt_dims.nbDims = bind.buffer->dims().size();

    for (int i = 0; i < trt_dims.nbDims; ++i) {
      trt_dims.d[i] = bind.buffer->dims()[i];
    }
    contexts_.front()->setBindingDimensions(bind_index, trt_dims);
  }

  CHECK(contexts_.front()->allInputDimensionsSpecified());

  auto output_binds = bindings_.front()->GetOutputBindings();
  for (auto bind : output_binds) {
    const int bind_index = engine_->getBindingIndex(bind.name.c_str());
    auto dims = contexts_.front()->getBindingDimensions(bind_index);
    std::vector<int32_t> ddim(dims.nbDims);
    for (int i = 0; i < dims.nbDims; ++i) {
      ddim[i] = dims.d[i];
    }
    bind.buffer->Resize(::phi::make_ddim(ddim));
    ctx.Alloc<float>(bind.buffer, sizeof(float) * bind.buffer->numel());
    buffers[bind_index] = static_cast<void*>(bind.buffer->data<float>());
  }

  contexts_.front()->enqueueV2(buffers.data(), ctx.stream(), nullptr);
}

void TrtEngine::FreshDeviceId() {
  int count;
  cudaGetDeviceCount(&count);
  CHECK_LT(device_id_, count);
  ::phi::backends::gpu::SetDeviceId(device_id_);
}

void TrtEngine::GetEngineInfo() {
#if IS_TRT_VERSION_GE(8200)
  LOG(INFO) << "====== engine info ======";
  std::unique_ptr<nvinfer1::IEngineInspector> infer_inspector(
      engine_->createEngineInspector());
  infer_inspector->setExecutionContext(contexts_.front().get());
  LOG(INFO) << infer_inspector->getEngineInformation(
      nvinfer1::LayerInformationFormat::kONELINE);
  LOG(INFO) << "====== engine info end ======";
#else
  LOG(INFO) << "Inspector needs TensorRT version 8.2 and after.";
#endif
}

}  // namespace tensorrt
}  // namespace backends
}  // namespace infrt
