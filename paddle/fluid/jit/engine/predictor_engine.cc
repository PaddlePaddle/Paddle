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

#include "paddle/fluid/jit/engine/predictor_engine.h"

#include "paddle/fluid/inference/api/analysis_predictor.h"
#include "paddle/fluid/inference/api/paddle_api.h"
#include "paddle/fluid/jit/function_utils.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace jit {

static PaddleTensor DenseTensorToPaddleTensor(DenseTensor *t);
static bool PaddleTensorToDenseTensor(const PaddleTensor &pt,
                                      DenseTensor *t,
                                      const platform::Place &place);

PredictorEngine::PredictorEngine(const std::shared_ptr<FunctionInfo> &info,
                                 const VariableMap &params_dict,
                                 const phi::Place &place)
    : info_(info), scope_(new framework::Scope()), place_(place) {
  utils::ShareParamsIntoScope(info_->ParamNames(), params_dict, scope_.get());
  VLOG(6) << framework::GenScopeTreeDebugInfo(scope_.get());

  // TODO(Aurelius84): Expose AnalysisConfig to user.
  AnalysisConfig config;
  config.SetProgFile(info->ProgramFilePath());
  if (platform::is_gpu_place(place_)) {
    config.EnableUseGpu(100, place_.GetDeviceId());
  } else if (platform::is_cpu_place(place_)) {
    config.DisableGpu();
    config.EnableMKLDNN();
    config.EnableMkldnnInt8();
    config.SetMkldnnCacheCapacity(0);
  }
  config.SetSkipLoadParams(true);
  config.SetApplyOptim(true);
  config.SwitchIrOptim(true);

  predictor_.reset(new AnalysisPredictor(config));

  predictor_->Init(
      scope_, std::make_shared<framework::ProgramDesc>(info_->ProgramDesc()));
}

std::vector<Tensor> PredictorEngine::operator()(
    const std::vector<Tensor> &inputs) {
  auto dense_tensors = utils::ToDenseTensors(inputs);
  return utils::ToTensors(this->operator()(dense_tensors));
}

std::vector<DenseTensor> PredictorEngine::operator()(
    const std::vector<DenseTensor> &inputs) {
  std::vector<PaddleTensor> pt_inputs;
  std::vector<PaddleTensor> pt_outputs;
  for (auto &t : inputs) {
    auto non_const_t = const_cast<DenseTensor *>(&t);
    pt_inputs.emplace_back(DenseTensorToPaddleTensor(non_const_t));
  }

  predictor_->Run(pt_inputs, &pt_outputs);

  std::vector<DenseTensor> outputs;
  for (auto &pt : pt_outputs) {
    DenseTensor t;
    PaddleTensorToDenseTensor(pt, &t, place_);
    outputs.emplace_back(t);
  }

  return outputs;
}

static PaddleTensor DenseTensorToPaddleTensor(DenseTensor *t) {
  PaddleTensor pt;
  switch (framework::TransToProtoVarType(t->dtype())) {
    case framework::proto::VarType::INT32: {
      pt.data.Reset(t->data(), t->numel() * sizeof(int32_t));
      pt.dtype = PaddleDType::INT32;
    } break;
    case framework::proto::VarType::INT64: {
      pt.data.Reset(t->data(), t->numel() * sizeof(int64_t));
      pt.dtype = PaddleDType::INT64;
    } break;
    case framework::proto::VarType::FP32: {
      pt.data.Reset(t->data(), t->numel() * sizeof(float));
      pt.dtype = PaddleDType::FLOAT32;
    } break;
    default:
      PADDLE_THROW(
          platform::errors::Unimplemented("Unsupported tensor date type. Now "
                                          "only supports INT64, FP32, INT32."));
  }
  pt.shape = phi::vectorize<int>(t->dims());
  return pt;
}

static bool PaddleTensorToDenseTensor(const PaddleTensor &pt,
                                      DenseTensor *t,
                                      const platform::Place &place) {
  framework::DDim ddim = phi::make_ddim(pt.shape);
  void *input_ptr;
  switch (pt.dtype) {
    case PaddleDType::INT64:
      input_ptr = t->mutable_data<int64_t>(ddim, place);
      break;
    case PaddleDType::FLOAT32:
      input_ptr = t->mutable_data<float>(ddim, place);
      break;
    case PaddleDType::INT32:
      input_ptr = t->mutable_data<int32_t>(ddim, place);
      break;
    case PaddleDType::FLOAT16:
      input_ptr = t->mutable_data<float16>(ddim, place);
      break;
    default:
      LOG(ERROR) << "unsupported feed type " << pt.dtype;
      return false;
  }

  PADDLE_ENFORCE_NOT_NULL(
      input_ptr,
      paddle::platform::errors::Fatal(
          "Cannot convert to LoDTensor because LoDTensor creation failed."));
  PADDLE_ENFORCE_NOT_NULL(
      pt.data.data(),
      paddle::platform::errors::InvalidArgument(
          "The data contained in the input PaddleTensor is illegal."));

  if (platform::is_cpu_place(place)) {
    // TODO(panyx0718): Init LoDTensor from existing memcpy to save a copy.
    std::memcpy(
        static_cast<void *>(input_ptr), pt.data.data(), pt.data.length());
  } else if (platform::is_ipu_place(place)) {
#ifdef PADDLE_WITH_IPU
    std::memcpy(
        static_cast<void *>(input_ptr), pt.data.data(), pt.data.length());
#else
    PADDLE_THROW(paddle::platform::errors::Fatal(
        "Not compile with WITH_IPU, should not reach here."));
#endif
  } else if (platform::is_gpu_place(place)) {
    PADDLE_ENFORCE_EQ(platform::is_xpu_place(place),
                      false,
                      platform::errors::InvalidArgument(
                          "Only one choice can be made between CPU and XPU."));
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto *dev_ctx = static_cast<const phi::GPUContext *>(pool.Get(place));
    auto dst_gpu_place = place;
    memory::Copy(dst_gpu_place,
                 static_cast<void *>(input_ptr),
                 platform::CPUPlace(),
                 pt.data.data(),
                 pt.data.length(),
                 dev_ctx->stream());
#else
    PADDLE_THROW(paddle::platform::errors::Fatal(
        "Not compile with CUDA, should not reach here."));
#endif
  } else if (platform::is_xpu_place(place)) {
#ifdef PADDLE_WITH_XPU
    auto dst_xpu_place = place;
    memory::Copy(dst_xpu_place,
                 static_cast<void *>(input_ptr),
                 platform::CPUPlace(),
                 pt.data.data(),
                 pt.data.length());
#else
    PADDLE_THROW(paddle::platform::errors::Fatal(
        "Not compile with XPU, should not reach here."));
#endif
  } else {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "The analysis predictor supports CPU, GPU and XPU now."));
  }
  return true;
}

}  // namespace jit
}  // namespace paddle
