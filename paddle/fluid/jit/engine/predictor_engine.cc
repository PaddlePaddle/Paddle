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
#include "paddle/phi/core/platform/device_context.h"

namespace paddle {
namespace jit {

PredictorEngine::PredictorEngine(
    const std::shared_ptr<FunctionInfo> &info,
    const std::shared_ptr<VariableMap> &params_dict,
    const phi::Place &place)
    : info_(info),
      params_dict_(params_dict),
      scope_(new framework::Scope()),
      place_(place) {
  utils::ShareParamsIntoScope(info_->ParamNames(), params_dict_, scope_.get());
  VLOG(6) << framework::GenScopeTreeDebugInfo(scope_.get());

  // TODO(Aurelius84): Expose AnalysisConfig to user.
  AnalysisConfig config;
  config.SetProgFile(info->ProgramFilePath());
  if (phi::is_gpu_place(place_)) {
    config.EnableUseGpu(100, place_.GetDeviceId());
  } else if (phi::is_cpu_place(place_)) {
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

PredictorEngine::PredictorEngine(
    const std::shared_ptr<FunctionInfo> &info,
    const std::shared_ptr<framework::Scope> &scope,
    const phi::Place &place,
    const std::shared_ptr<PaddlePredictor> &predictor)
    : info_(info),
      scope_(scope),
      place_(place),
      predictor_(std::dynamic_pointer_cast<AnalysisPredictor, PaddlePredictor>(
          predictor)) {}

std::unique_ptr<BaseEngine> PredictorEngine::Clone(void *stream) {
  auto *x =
      new PredictorEngine(info_, scope_, place_, predictor_->Clone(stream));
  return std::unique_ptr<BaseEngine>(x);
}

std::vector<Tensor> PredictorEngine::operator()(
    const std::vector<Tensor> &inputs) {
  std::vector<Tensor> outputs;
  predictor_->Run(inputs, &outputs);

  return outputs;
}

std::vector<DenseTensor> PredictorEngine::operator()(
    const std::vector<DenseTensor> &inputs) {
  return utils::ToDenseTensors(this->operator()(utils::ToTensors(inputs)));
}

}  // namespace jit
}  // namespace paddle
