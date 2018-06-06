/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <memory>

#include "Layer.h"
#include "paddle/gserver/evaluators/Evaluator.h"

DECLARE_int32(trainer_id);

namespace paddle {

class ValidationLayer : public Layer {
 public:
  explicit ValidationLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  LayerPtr getOutputLayer() { return inputLayers_[0]; }

  LayerPtr getLabelLayer() { return inputLayers_[1]; }

  LayerPtr getInfoLayer() {
    assert(inputLayers_.size() > 2);
    return inputLayers_[2];
  }

  void forward(PassType passType) override;

  void backward(const UpdateCallback& callback = nullptr) override;

  virtual void validationImp(MatrixPtr outputValue, IVectorPtr label) = 0;

  void onPassEnd() override = 0;
};

/*
 * AucValidation
 */
class AucValidation : public ValidationLayer {
 public:
  explicit AucValidation(const LayerConfig& config)
      : ValidationLayer(config),
        cpuOutput_(nullptr),
        cpuLabel_(nullptr),
        cpuWeight_(nullptr) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void validationImp(MatrixPtr outputValue, IVectorPtr label) override;

  void onPassEnd() override;

  struct PredictionResult {
    PredictionResult(real __out, int __label) : out(__out), label(__label) {}
    real out;
    int label;
  };
  std::vector<PredictionResult> predictArray_;

 private:
  bool passBegin_;
  std::unique_ptr<Evaluator> evaluator_;
  MatrixPtr cpuOutput_;
  IVectorPtr cpuLabel_;
  MatrixPtr cpuWeight_;
};

/*
 * positive-negative pair rate Validation
 */
class PnpairValidation : public ValidationLayer {
 public:
  explicit PnpairValidation(const LayerConfig& config)
      : ValidationLayer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void validationImp(MatrixPtr outputValue, IVectorPtr label) override;

  void onPassEnd() override;

 private:
  bool passBegin_;
  std::unique_ptr<Evaluator> evaluator_;
};

typedef std::shared_ptr<ValidationLayer> ValidationLayerPtr;
}  // namespace paddle
