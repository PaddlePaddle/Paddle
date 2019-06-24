// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/api/cxx_api.h"
#include "paddle/fluid/lite/api/paddle_api.h"

namespace paddle {
namespace lite {

class CxxPaddleApiImpl : public lite_api::PaddlePredictor {
 public:
  CxxPaddleApiImpl();

  /// Create a new predictor from a config.
  void Init(const lite_api::CxxConfig &config);

  std::unique_ptr<lite_api::Tensor> GetInput(int i) override;

  std::unique_ptr<const lite_api::Tensor> GetOutput(int i) const override;

  void Run() override;

  std::unique_ptr<const lite_api::Tensor> GetTensor(
      const std::string &name) const override;

  void SaveOptimizedModel(const std::string &model_dir) override;

 private:
  Predictor raw_predictor_;
};

CxxPaddleApiImpl::CxxPaddleApiImpl() {}

void CxxPaddleApiImpl::Init(const lite_api::CxxConfig &config) {
  auto places = config.valid_places();
  places.emplace_back(TARGET(kHost), PRECISION(kAny), DATALAYOUT(kAny));
  raw_predictor_.Build(config.model_dir(), config.preferred_place(), places);
}

std::unique_ptr<lite_api::Tensor> CxxPaddleApiImpl::GetInput(int i) {
  auto *x = raw_predictor_.GetInput(i);
  return std::unique_ptr<lite_api::Tensor>(new lite_api::Tensor(x));
}

std::unique_ptr<const lite_api::Tensor> CxxPaddleApiImpl::GetOutput(
    int i) const {
  const auto *x = raw_predictor_.GetOutput(i);
  return std::unique_ptr<lite_api::Tensor>(new lite_api::Tensor(x));
}

void CxxPaddleApiImpl::Run() { raw_predictor_.Run(); }

std::unique_ptr<const lite_api::Tensor> CxxPaddleApiImpl::GetTensor(
    const std::string &name) const {
  auto *x = raw_predictor_.GetTensor(name);
  return std::unique_ptr<const lite_api::Tensor>(new lite_api::Tensor(x));
}

void CxxPaddleApiImpl::SaveOptimizedModel(const std::string &model_dir) {
  raw_predictor_.SaveModel(model_dir);
}

}  // namespace lite

namespace lite_api {

template <>
std::shared_ptr<PaddlePredictor> CreatePaddlePredictor(
    const CxxConfig &config) {
  auto x = std::make_shared<lite::CxxPaddleApiImpl>();
  x->Init(config);
  return x;
}

}  // namespace lite_api
}  // namespace paddle
