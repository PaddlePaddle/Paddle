/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

/*
 * This file contains the implementation of inference API with Anakin engine
 * embeded, this API can only support Anakin models.
 */

#pragma once

#include "paddle/contrib/inference/paddle_inference_api.h"
// NOTE This header file do not have namespace.
// TODO(Superjomn) Tell Anakin to provide better APIs.
#include "Anakin/test/framework/net/paddle_api.h"

namespace paddle {

class PaddleInferenceAnakinPredictor : public PaddlePredictor {
 public:
  PaddleInferenceAnakinPredictor(const AnakinConfig& config);

  // NOTE Unlike the native engine, the buffers of anakin engine's output_data
  // should be allocated first.
  // TODO(Superjomn) should unify all the behaviors of output_data accross all
  // the engines.
  bool Run(const std::vector<PaddleTensor>& inputs,
           std::vector<PaddleTensor>* output_data) override;

  std::unique_ptr<PaddlePredictor> Clone() override;

 private:
  bool Init(const AnakinConfig& config);

  AnakinEngine<anakin::NV, anakin::saber::AK_FLOAT, anakin::Precision::FP32>
      engine_;
};

}  // namespace paddle
