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

#include "paddle/fluid/jit/engine/base_engine.h"
#include "paddle/fluid/jit/function_schema.h"
#include "paddle/fluid/jit/function_utils.h"

namespace paddle {
class AnalysisPredictor;
class PaddlePredictor;

namespace framework {
class Scope;
}

namespace jit {

class PredictorEngine : public BaseEngine {
 public:
  PredictorEngine(const std::shared_ptr<FunctionInfo> &info,
                  const std::shared_ptr<VariableMap> &params_dict,
                  const phi::Place &place);

  PredictorEngine(const std::shared_ptr<FunctionInfo> &info,
                  const std::shared_ptr<framework::Scope> &scope,
                  const phi::Place &place,
                  const std::shared_ptr<PaddlePredictor> &predictor);

  ~PredictorEngine() noexcept {}

  std::vector<Tensor> operator()(const std::vector<Tensor> &inputs) override;

  std::vector<DenseTensor> operator()(
      const std::vector<DenseTensor> &inputs) override;

  std::unique_ptr<BaseEngine> Clone(void *stream = nullptr) override;

 private:
  std::shared_ptr<FunctionInfo> info_;
  std::shared_ptr<VariableMap> params_dict_;
  std::shared_ptr<framework::Scope> scope_;
  phi::Place place_;
  std::shared_ptr<AnalysisPredictor> predictor_;
};

}  // namespace jit
}  // namespace paddle
