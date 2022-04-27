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

#include "paddle/fluid/inference/api/onnxruntime_predictor.h"

#include <glog/logging.h>

#include <algorithm>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid//platform/device/gpu/gpu_types.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/version.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/analysis/passes/memory_optimize_pass.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/paddle_inference_pass.h"
#include "paddle/fluid/inference/utils/io_utils.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {

paddle_infer::DataType ConvertONNXType(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return paddle_infer::DataType::FLOAT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return paddle_infer::DataType::FLOAT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return paddle_infer::DataType::INT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return paddle_infer::DataType::INT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return paddle_infer::DataType::INT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return paddle_infer::DataType::UINT8;
    default:
      LOG(ERROR) << "unsupported ONNX Tensor Type: " << static_cast<int>(type);
      return paddle_infer::DataType::FLOAT32;
  }
}

bool CheckConvertToONNX(const AnalysisConfig &config) {
  if (!config.model_dir().empty()) {
    LOG(ERROR) << "Paddle2ONNX not support model_dir config";
    // TODO(heliqi jiangjiajun): Paddle2ONNX not support
    // config.model_dir() + "/__model__"
    // config.model_dir() + var_name
    return false;
  } else if (config.prog_file().empty() || config.params_file().empty()) {
    LOG(ERROR) << string::Sprintf(
        "not valid model path '%s' or program path '%s' or params path '%s'.",
        config.model_dir(), config.prog_file(), config.params_file());
    return false;
  }
  return paddle2onnx::IsExportable(config.prog_file(), config.params_file(),
                                   config.model_from_memory());
}

bool ONNXRuntimePredictor::Init() {
  VLOG(3) << "ONNXRuntime Predictor::init()";

  // Now ONNXRuntime only support CPU
  const char *device_name = config_.use_gpu() ? "Cuda" : "Cpu";
  if (config_.use_gpu()) {
    place_ = paddle::platform::CUDAPlace(config_.gpu_device_id());
  } else {
    place_ = paddle::platform::CPUPlace();
  }

  std::string onnx_proto;
  paddle2onnx::Export(config_.prog_file(), config_.params_file(), &onnx_proto,
                      config_.model_from_memory());

  Ort::SessionOptions session_options;
  if (config_.ort_optimization_enabled()) {
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
  }
  // Turn optimization off first, and then turn it on when it's stable
  // session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  // session_options.EnableCpuMemArena();
  // session_options.EnableMemPattern();
  // session_options.SetInterOpNumThreads(config_.cpu_math_library_num_threads());
  session_options.SetIntraOpNumThreads(config_.cpu_math_library_num_threads());
  VLOG(2) << "ONNXRuntime threads " << config_.cpu_math_library_num_threads();
  if (config_.profile_enabled()) {
    LOG(WARNING) << "ONNXRuntime Profiler is activated, which might affect the "
                    "performance";
#if defined(_WIN32)
    session_options.EnableProfiling(L"ONNX");
#else
    session_options.EnableProfiling("ONNX");
#endif
  } else {
    VLOG(2) << "ONNXRuntime Profiler is deactivated, and no profiling report "
               "will be "
               "generated.";
  }
  session_ = {env_, onnx_proto.data(), onnx_proto.size(), session_options};
  binding_ = std::make_shared<Ort::IoBinding>(session_);

  Ort::MemoryInfo memory_info(device_name, OrtDeviceAllocator,
                              place_.GetDeviceId(), OrtMemTypeDefault);
  Ort::Allocator allocator(session_, memory_info);

  size_t n_inputs = session_.GetInputCount();
  for (size_t i = 0; i < n_inputs; ++i) {
    auto input_name = session_.GetInputName(i, allocator);
    auto type_info = session_.GetInputTypeInfo(i);
    std::vector<int64_t> shape =
        type_info.GetTensorTypeAndShapeInfo().GetShape();
    ONNXTensorElementDataType data_type =
        type_info.GetTensorTypeAndShapeInfo().GetElementType();
    input_desc_.emplace_back(ONNXDesc{input_name, shape, data_type});
    allocator.Free(input_name);
  }

  size_t n_outputs = session_.GetOutputCount();
  for (size_t i = 0; i < n_outputs; ++i) {
    auto output_name = session_.GetOutputName(i, allocator);
    auto type_info = session_.GetOutputTypeInfo(i);
    std::vector<int64_t> shape =
        type_info.GetTensorTypeAndShapeInfo().GetShape();
    ONNXTensorElementDataType data_type =
        type_info.GetTensorTypeAndShapeInfo().GetElementType();
    output_desc_.emplace_back(ONNXDesc{output_name, shape, data_type});

    Ort::MemoryInfo out_memory_info(device_name, OrtDeviceAllocator,
                                    place_.GetDeviceId(), OrtMemTypeDefault);
    binding_->BindOutput(output_name, out_memory_info);

    allocator.Free(output_name);
  }
  return true;
}

template <>
std::unique_ptr<PaddlePredictor>
CreatePaddlePredictor<AnalysisConfig, PaddleEngineKind::kONNXRuntime>(
    const AnalysisConfig &config) {
  if (config.glog_info_disabled()) {
    FLAGS_logtostderr = 1;
    FLAGS_minloglevel = 2;  // GLOG_ERROR
  }

  PADDLE_ENFORCE_EQ(
      config.is_valid(), true,
      platform::errors::InvalidArgument(
          "Note: Each config can only be used for one predictor."));

  VLOG(3) << "create ONNXRuntimePredictor";

  std::unique_ptr<PaddlePredictor> predictor(new ONNXRuntimePredictor(config));
  // Each config can only be used for one predictor.
  config.SetInValid();
  auto predictor_p = dynamic_cast<ONNXRuntimePredictor *>(predictor.get());

  if (!predictor_p->Init()) {
    return nullptr;
  }

  return predictor;
}

std::vector<std::string> ONNXRuntimePredictor::GetInputNames() {
  std::vector<std::string> input_names;
  for (auto input_desc : input_desc_) {
    input_names.push_back(input_desc.name);
  }
  return input_names;
}

std::map<std::string, std::vector<int64_t>>
ONNXRuntimePredictor::GetInputTensorShape() {
  std::map<std::string, std::vector<int64_t>> input_shapes;
  for (auto input_desc : input_desc_) {
    input_shapes[input_desc.name] = input_desc.shape;
  }
  return input_shapes;
}

std::vector<std::string> ONNXRuntimePredictor::GetOutputNames() {
  std::vector<std::string> output_names;
  for (auto output_desc : output_desc_) {
    output_names.push_back(output_desc.name);
  }
  return output_names;
}

bool ONNXRuntimePredictor::FindONNXDesc(const std::string &name,
                                        bool is_input) {
  if (is_input) {
    for (auto i : input_desc_)
      if (i.name == name) return true;
  } else {
    for (auto i : output_desc_)
      if (i.name == name) return true;
  }
  return false;
}

std::unique_ptr<ZeroCopyTensor> ONNXRuntimePredictor::GetInputTensor(
    const std::string &name) {
  PADDLE_ENFORCE_EQ(FindONNXDesc(name, true), true,
                    platform::errors::PreconditionNotMet(
                        "The in variable named %s is not found in the "
                        "ONNXPredictor.",
                        name));
  std::unique_ptr<ZeroCopyTensor> res(new ZeroCopyTensor(nullptr));
  res->input_or_output_ = true;
  res->SetName(name);
  if (platform::is_cpu_place(place_)) {
    res->SetPlace(PaddlePlace::kCPU);
  } else {
    auto gpu_place = place_;
    res->SetPlace(PaddlePlace::kGPU, gpu_place.GetDeviceId());
  }
  res->SetOrtMark(true);
  res->SetOrtBinding(binding_);
  return res;
}

std::unique_ptr<ZeroCopyTensor> ONNXRuntimePredictor::GetOutputTensor(
    const std::string &name) {
  PADDLE_ENFORCE_EQ(FindONNXDesc(name, false), true,
                    platform::errors::PreconditionNotMet(
                        "The out variable named %s is not found in the "
                        "ONNXPredictor.",
                        name));
  std::unique_ptr<ZeroCopyTensor> res(new ZeroCopyTensor(nullptr));
  res->input_or_output_ = false;
  res->SetName(name);
  if (platform::is_cpu_place(place_)) {
    res->SetPlace(PaddlePlace::kCPU);
  } else {
    auto gpu_place = place_;
    res->SetPlace(PaddlePlace::kGPU, gpu_place.GetDeviceId());
  }
  res->SetOrtMark(true);
  res->SetOrtBinding(binding_);
  int size = output_desc_.size();
  for (int i = 0; i < size; ++i)
    if (output_desc_[i].name == name) {
      res->idx_ = i;
      res->dtype_ = ConvertONNXType(output_desc_[i].dtype);
      break;
    }
  return res;
}

bool ONNXRuntimePredictor::Run(const std::vector<PaddleTensor> &inputs,
                               std::vector<PaddleTensor> *output_data,
                               int batch_size) {
  LOG(ERROR) << "Not support Run";
  return false;
}

bool ONNXRuntimePredictor::ZeroCopyRun() {
  try {
    const char *device_name = place_ == PlaceType::kCPU ? "Cpu" : "Cuda";
    for (auto output : output_desc_) {
      Ort::MemoryInfo out_memory_info(device_name, OrtDeviceAllocator,
                                      place_.GetDeviceId(), OrtMemTypeDefault);
      binding_->BindOutput(output.name.c_str(), out_memory_info);
    }
    session_.Run({}, *(binding_.get()));
  } catch (const std::exception &e) {
    LOG(ERROR) << e.what();
    return false;
  }

  return true;
}

std::unique_ptr<PaddlePredictor> ONNXRuntimePredictor::Clone() {
  LOG(ERROR) << "Not support Clone(), Please create new Predictor";
  return nullptr;
}

uint64_t ONNXRuntimePredictor::TryShrinkMemory() {
  return paddle::memory::Release(place_);
}

ONNXRuntimePredictor::~ONNXRuntimePredictor() {
  binding_->ClearBoundInputs();
  binding_->ClearBoundOutputs();

  memory::Release(place_);
}

}  // namespace paddle
