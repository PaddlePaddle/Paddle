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

#include <algorithm>
#include <fstream>
#include <memory>

#include "ValidationLayer.h"
#include "paddle/utils/Logging.h"

namespace paddle {

bool ValidationLayer::init(const LayerMap& layerMap,
                           const ParameterMap& parameterMap) {
  return Layer::init(layerMap, parameterMap);
}

void ValidationLayer::forward(PassType passType) {
  Layer::forward(passType);

  MatrixPtr output = getInputValue(*getOutputLayer());
  CHECK(output);
  IVectorPtr label = getInputLabel(*getLabelLayer());
  CHECK(label);
  validationImp(output, label);
}

void ValidationLayer::backward(const UpdateCallback& callback) {
  (void)callback;
}

bool AucValidation::init(const LayerMap& layerMap,
                         const ParameterMap& parameterMap) {
  bool ret = ValidationLayer::init(layerMap, parameterMap);
  EvaluatorConfig config;
  config.set_name(getName());
  config.set_type("last-column-auc");
  config.add_input_layers(inputLayers_[0]->getName());
  config.add_input_layers(inputLayers_[1]->getName());
  if (3 == inputLayers_.size()) {
    config.add_input_layers(inputLayers_[2]->getName());
  }
  evaluator_.reset(Evaluator::create(config));
  passBegin_ = false;
  return ret;
}

void AucValidation::validationImp(MatrixPtr output, IVectorPtr label) {
  if (!passBegin_) {
    passBegin_ = true;
    evaluator_->start();
  }

  bool supportWeight = (3 == inputLayers_.size()) ? true : false;
  MatrixPtr weight = supportWeight ? getInputValue(*inputLayers_[2]) : nullptr;
  if (dynamic_cast<GpuMatrix*>(output.get())) {
    size_t height = output->getHeight();
    size_t width = output->getWidth();
    Matrix::resizeOrCreate(cpuOutput_,
                           height,
                           width,
                           /* trans=*/false,
                           /* useGpu=*/false);
    cpuOutput_->copyFrom(*output);
    IVector::resizeOrCreate(cpuLabel_, height, false);
    cpuLabel_->copyFrom(*label);

    if (supportWeight) {
      Matrix::resizeOrCreate(cpuWeight_, height, (size_t)1, false, false);
      cpuWeight_->copyFrom(*weight);
    }

    output = cpuOutput_;
    label = cpuLabel_;
    weight = cpuWeight_;
  }

  for (size_t i = 0; i < output->getHeight(); i++) {
    float y1 = output->getData()[i * output->getWidth() + 1];
    int* labels = label->getData();
    predictArray_.push_back(PredictionResult(y1, labels[i]));
  }
  std::vector<Argument> arguments;
  if (3 == inputLayers_.size()) {
    arguments.resize(3);
    arguments[2].value = weight;
  } else {
    arguments.resize(2);
  }
  arguments[0].value = output;
  arguments[1].ids = label;
  evaluator_->evalImp(arguments);
}

void AucValidation::onPassEnd() {
  if (!FLAGS_predict_file.empty()) {
    std::ofstream fs(FLAGS_predict_file);
    CHECK(fs) << "Fail to open " << FLAGS_predict_file;
    for (auto& res : predictArray_) {
      fs << res.out << " " << res.label << std::endl;
    }
  }

  evaluator_->finish();
  LOG(INFO) << *evaluator_;
  passBegin_ = false;
  predictArray_.clear();
}

bool PnpairValidation::init(const LayerMap& layerMap,
                            const ParameterMap& parameterMap) {
  bool ret = ValidationLayer::init(layerMap, parameterMap);
  if (!ret) return ret;
  CHECK_GE(inputLayers_.size(), 3UL);
  CHECK_LE(inputLayers_.size(), 4UL);
  EvaluatorConfig config;
  config.set_name(getName());
  config.set_type("pnpair");
  config.add_input_layers(inputLayers_[0]->getName());
  config.add_input_layers(inputLayers_[1]->getName());
  config.add_input_layers(inputLayers_[2]->getName());
  if (4 == inputLayers_.size()) {
    config.add_input_layers(inputLayers_[3]->getName());
  }
  evaluator_.reset(Evaluator::create(config));
  passBegin_ = false;
  return true;
}

void PnpairValidation::validationImp(MatrixPtr output, IVectorPtr label) {
  if (!passBegin_) {
    passBegin_ = true;
    evaluator_->start();
  }
  MatrixPtr weight =
      (4 == inputLayers_.size()) ? getInputValue(*inputLayers_[3]) : nullptr;
  IVectorPtr info = getInputLabel(*getInfoLayer());
  std::vector<Argument> arguments;
  if (4 == inputLayers_.size()) {
    arguments.resize(4);
    arguments[3].value = weight;
  } else {
    arguments.resize(3);
  }
  arguments[0].value = output;
  arguments[1].ids = label;
  arguments[2].ids = info;
  evaluator_->evalImp(arguments);
}

void PnpairValidation::onPassEnd() {
  if (!FLAGS_predict_file.empty()) {
    (dynamic_cast<PnpairEvaluator*>(evaluator_.get()))->printPredictResults();
  }
  evaluator_->finish();
  LOG(INFO) << *evaluator_;
  passBegin_ = false;
}

}  // namespace paddle
