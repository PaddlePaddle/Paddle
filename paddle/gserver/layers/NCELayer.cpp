/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <random>

#include "Layer.h"
#include "MultinomialSampler.h"
#include "paddle/math/MathFunctions.h"

namespace paddle {

/**
 * Noise-contrastive estimation.
 * Implements the method in the following paper:
 * A fast and simple algorithm for training neural probabilistic language
 * models.
 *
 * The config file api is nce_layer.
 */
class NCELayer : public Layer {
  int numClasses_;
  /// number of input layer besides labelLayer and weightLayer
  int numInputs_;
  LayerPtr labelLayer_;
  /// weight layer, can be None
  LayerPtr weightLayer_;
  WeightList weights_;
  std::unique_ptr<Weight> biases_;
  std::unique_ptr<MultinomialSampler> sampler_;

  std::uniform_int_distribution<int> rand_;

  struct Sample {
    int sampleId;
    int labelId;
    bool target;
    real weight;
  };
  std::vector<Sample> samples_;
  /// whether samples_ is prepared
  bool prepared_;
  Argument sampleOut_;

  IVectorPtr labelIds_;

  // The temporary variables in CPU memory.
  MatrixPtr cpuInput_;
  MatrixPtr cpuInputGrad_;
  MatrixPtr cpuOutput_;
  MatrixPtr cpuBias_;
  MatrixPtr cpuBiasGrad_;
  MatrixPtr cpuWeight_;
  MatrixPtr cpuWeightGrad_;
  IVectorPtr cpuLabel_;
  MatrixPtr cpuLabelValue_;
  MatrixPtr cpuWeightLayerValue_;


public:
  explicit NCELayer(const LayerConfig& config)
      : Layer(config),
        numClasses_(config.num_classes()),
        rand_(0, config.num_classes() - 1),
        prepared_(false) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override {
    /* Initialize the basic parent class */
    Layer::init(layerMap, parameterMap);

    /* initialize the weightList */
    size_t i;
    for (i = 0; i < inputLayers_.size(); i++) {
      if (!parameters_[i]) break;
      size_t width = inputLayers_[i]->getSize();
      // create a new weight
      CHECK_EQ(parameters_[i]->getSize(), width * numClasses_);
      Weight* w = new Weight(numClasses_, width, parameters_[i]);

      // append the new weight to the list
      weights_.emplace_back(w);
    }

    CHECK_EQ(1U, getSize());

    numInputs_ = i;
    CHECK_GE(numInputs_, 1)
        << "Must have at least one input besides label and weight";
    CHECK_LT(i, inputLayers_.size()) << "Missing label layer";
    labelLayer_ = inputLayers_[i];
    if (++i < inputLayers_.size()) {
      weightLayer_ = inputLayers_[i];
      ++i;
    }
    CHECK_EQ(i, inputLayers_.size());

    /* initialize biases_ */
    if (biasParameter_.get() != NULL) {
      CHECK_EQ(biasParameter_->getSize(), (size_t)numClasses_);
      biases_.reset(new Weight(1, numClasses_, biasParameter_));
    }

    if (config_.neg_sampling_dist_size()) {
      CHECK_EQ(numClasses_, config_.neg_sampling_dist_size());
      sampler_.reset(MultinomialSampler::create(
          config_.neg_sampling_dist().data(), numClasses_));
    }

    return true;
  }

  void prepareSamples() {
    Argument label_layer = getInput(*labelLayer_);
    int batchSize = label_layer.getBatchSize();
    IVectorPtr label = label_layer.ids;
    MatrixPtr label_value = label_layer.value;
    if (useGpu_) {
        if (label) {
            IVector::resizeOrCreate(cpuLabel_,
                        label->getSize(),
                        false);
            cpuLabel_->copyFrom(*label);
        }
        if (label_value) {
            Matrix::resizeOrCreateSparseMatrix(
                           cpuLabelValue_,
                           label_value->getHeight(),
                           label_value->getWidth(),
                           label_value->getElementCnt(),
                           NO_VALUE,
                           SPARSE_CSR,
                           false,
                           false);
            cpuLabelValue_->copyFrom(*label_value, HPPL_STREAM_DEFAULT);
        }
    } else {
        if (label) {
            cpuLabel_ = label;
        }
        if (label_value) {
            cpuLabelValue_ = label_value;
        }
    }
    CpuSparseMatrixPtr multiLabel = std::dynamic_pointer_cast<CpuSparseMatrix>(
        cpuLabelValue_);

    CHECK(cpuLabel_ || multiLabel)
        << "The label layer must have ids or NonValueSparseMatrix value";

    auto& randEngine = ThreadLocalRandomEngine::get();

    samples_.clear();
    samples_.reserve(batchSize * (1 + config_.num_neg_samples()));

    real* weight = nullptr;
    if (weightLayer_) {
        const MatrixPtr& weight_value = getInputValue(*weightLayer_);
        if (useGpu_) {
            Matrix::resizeOrCreate(cpuWeightLayerValue_,
                                   weight_value->getHeight(),
                                   weight_value->getWidth(),
                                   false,
                                   false);
            cpuWeightLayerValue_->copyFrom(*weight_value);
        } else {
            cpuWeightLayerValue_ = weight_value;
        }
        weight = cpuWeightLayerValue_->getData();
    }

    for (int i = 0; i < batchSize; ++i) {
      real w = weight ? weight[i] : 1;
      if (cpuLabel_) {
        int* ids = cpuLabel_->getData();
        samples_.push_back({i, ids[i], true, w});
      } else {
        const int* cols = multiLabel->getRowCols(i);
        int n = multiLabel->getColNum(i);
        for (int j = 0; j < n; ++j) {
          samples_.push_back({i, cols[j], true, w});
        }
      }
      for (int j = 0; j < config_.num_neg_samples(); ++j) {
        int id = sampler_ ? sampler_->gen(randEngine) : rand_(randEngine);
        samples_.push_back({i, id, false, w});
      }
    }
    prepared_ = true;
  }

  void prefetch() override {
    prepareSamples();
    IVector::resizeOrCreate(labelIds_, samples_.size(), useGpu_);
    int* ids = labelIds_->getData();
    for (size_t i = 0; i < samples_.size(); ++i) {
      ids[i] = samples_[i].labelId;
    }

    for (int i = 0; i < numInputs_; ++i) {
      auto sparseParam =
          dynamic_cast<SparsePrefetchRowCpuMatrix*>(weights_[i]->getW().get());
      if (sparseParam) {
        sparseParam->addRows(labelIds_);
      }
    }
  }

  void forward(PassType passType) override {
    Layer::forward(passType);

    if (!prepared_) {
      if (passType == PASS_GC) {
        ThreadLocalRandomEngine::get().seed(ThreadLocalRand::getDefaultSeed());
      }
      prepareSamples();
    }
    prepared_ = false;

    /* malloc memory for the output_ if necessary */
    int batchSize = getInputValue(0)->getHeight();
    int size = getSize();
    resetOutput(batchSize, size);

    Matrix::resizeOrCreate(sampleOut_.value,
                           1,
                           samples_.size(),
                           false,
                           false);

    forwardBias();

    for (int l = 0; l < numInputs_; ++l) {
      forwardOneInput(l);
    }

    auto status = activation_->forward(sampleOut_);
    status.check();

    forwardCost();
  }

  void backward(const UpdateCallback& callback) override {
    Matrix::resizeOrCreate(sampleOut_.grad,
                           1,
                           samples_.size(),
                           false,
                           false);

    backwardCost();

    auto status = activation_->backward(sampleOut_);
    status.check();

    if (biases_->getWGrad()) {
      backwardBias(callback);
    }

    for (int l = 0; l < numInputs_; ++l) {
      backwardOneInput(l, callback);
    }
  }

  void forwardBias() {
    if (!biases_) {
      sampleOut_.value->zeroMem();
    } else {
      MatrixPtr bias_value = biases_->getW();
      if (useGpu_) {
          Matrix::resizeOrCreate(cpuBias_,
                         bias_value->getHeight(),
                         bias_value->getWidth(),
                         false,
                         false);
          cpuBias_->copyFrom(*bias_value);
      } else {
          cpuBias_ = bias_value;
      }
      real* bias = cpuBias_->getData();
      real* sampleOut = sampleOut_.value->getData();
      for (size_t i = 0; i < samples_.size(); ++i) {
        sampleOut[i] = bias[samples_[i].labelId];
      }
    }
  }

  void backwardBias(const UpdateCallback& callback) {
    if (!biases_) return;
    MatrixPtr bias_grad = biases_->getWGrad();
    if (useGpu_) {
        Matrix::resizeOrCreate(cpuBiasGrad_,
                         bias_grad->getHeight(),
                         bias_grad->getWidth(),
                         false,
                         false);
        cpuBiasGrad_->copyFrom(*bias_grad);
    } else {
        cpuBiasGrad_ = bias_grad;
    }
    real* bias = cpuBiasGrad_->getData();
    real* sampleOut = sampleOut_.grad->getData();
    for (size_t i = 0; i < samples_.size(); ++i) {
      bias[samples_[i].labelId] += sampleOut[i];
    }
    if (useGpu_) {
        bias_grad->copyFrom(*cpuBiasGrad_);
    } else {
        bias_grad = cpuBiasGrad_;
    }
    biases_->incUpdate(callback);
  }

  void forwardOneInput(int layerId) {
    const MatrixPtr& inputMat = getInputValue(layerId);
    const MatrixPtr& weightMat = weights_[layerId]->getW();
    if (useGpu_) {
        Matrix::resizeOrCreate(cpuInput_,
                         inputMat->getHeight(),
                         inputMat->getWidth(),
                         false,
                         false);
        cpuInput_->copyFrom(*inputMat);
        Matrix::resizeOrCreate(cpuWeight_,
                         weightMat->getHeight(),
                         weightMat->getWidth(),
                         false,
                         false);
        cpuWeight_->copyFrom(*weightMat);
    } else {
        cpuInput_ = inputMat;
        cpuWeight_ = weightMat;
    }
    int dim = inputMat->getWidth();
    real* sampleOut = sampleOut_.value->getData();

    for (size_t i = 0; i < samples_.size(); ++i) {
      sampleOut[i] += dotProduct(dim,
                                 cpuInput_->getRowBuf(samples_[i].sampleId),
                                 cpuWeight_->getRowBuf(samples_[i].labelId));
    }
  }

  void backwardOneInput(int layerId, const UpdateCallback& callback) {
    const MatrixPtr& inputMat = getInputValue(layerId);
    MatrixPtr inputGradMat = getInputGrad(layerId);
    const MatrixPtr& weightMat = weights_[layerId]->getW();
    MatrixPtr weightGradMat = weights_[layerId]->getWGrad();

    if (useGpu_) {
        Matrix::resizeOrCreate(cpuInput_,
                         inputMat->getHeight(),
                         inputMat->getWidth(),
                         false,
                         false);
        cpuInput_->copyFrom(*inputMat);
        if (inputGradMat) {
          Matrix::resizeOrCreate(cpuInputGrad_,
                         inputGradMat->getHeight(),
                         inputGradMat->getWidth(),
                         false,
                         false);
          cpuInputGrad_->copyFrom(*inputGradMat);
        }
        Matrix::resizeOrCreate(cpuWeight_,
                         weightMat->getHeight(),
                         weightMat->getWidth(),
                         false,
                         false);
        cpuWeight_->copyFrom(*weightMat);
        if (weightGradMat) {
          Matrix::resizeOrCreate(cpuWeightGrad_,
                         weightGradMat->getHeight(),
                         weightGradMat->getWidth(),
                         false,
                         false);
          cpuWeightGrad_->copyFrom(*weightGradMat);
        }
    } else {
        cpuInput_ = inputMat;
        if (inputGradMat) {
          cpuInputGrad_ = inputGradMat;
        }
        cpuWeight_ = weightMat;
        if (weightGradMat) {
          cpuWeightGrad_ = weightGradMat;
        }
    }
    int dim = cpuInput_->getWidth();
    real* sampleGrad = sampleOut_.grad->getData();

    if (weightGradMat) {
      for (size_t i = 0; i < samples_.size(); ++i) {
        axpy(dim,
             sampleGrad[i],
             cpuInput_->getRowBuf(samples_[i].sampleId),
             cpuWeightGrad_->getRowBuf(samples_[i].labelId));
      }
      if (useGpu_) {
        weightGradMat->copyFrom(*cpuWeightGrad_);
      } else {
        weightGradMat = cpuWeightGrad_;
      }
      weights_[layerId]->incUpdate(callback);
    }

    if (inputGradMat) {
      for (size_t i = 0; i < samples_.size(); ++i) {
        axpy(dim,
             sampleGrad[i],
             cpuWeight_->getRowBuf(samples_[i].labelId),
             cpuInputGrad_->getRowBuf(samples_[i].sampleId));
      }
      if (useGpu_) {
        inputGradMat->copyFrom(*cpuInputGrad_);
      } else {
        inputGradMat = cpuInputGrad_;
      }
    }
  }

  void forwardCost() {
    MatrixPtr output_value = output_.value;
    if (useGpu_) {
        Matrix::resizeOrCreate(cpuOutput_,
                         output_value->getHeight(),
                         output_value->getWidth(),
                         false,
                         false);
        cpuOutput_->copyFrom(*output_value);
    } else {
        cpuOutput_ = output_value;
    }
    real* out = cpuOutput_->getData();
    real* sampleOut = sampleOut_.value->getData();
    real b = 1. / numClasses_ * config_.num_neg_samples();
    for (size_t i = 0; i < samples_.size(); ++i) {
      real o = sampleOut[i];
      if (sampler_) {
        b = config_.num_neg_samples() *
            config_.neg_sampling_dist(samples_[i].labelId);
      }
      real cost = samples_[i].target ? -log(o / (o + b)) : -log(b / (o + b));
      out[samples_[i].sampleId] += samples_[i].weight * cost;
    }
    if (useGpu_) {
        output_value->copyFrom(*cpuOutput_);
    } else {
        output_value = cpuOutput_;
    }
  }

  void backwardCost() {
    real* sampleOut = sampleOut_.value->getData();
    real* sampleGrad = sampleOut_.grad->getData();

    real b = 1. / numClasses_ * config_.num_neg_samples();
    for (size_t i = 0; i < samples_.size(); ++i) {
      real o = sampleOut[i];
      if (sampler_) {
        b = config_.num_neg_samples() *
            config_.neg_sampling_dist(samples_[i].labelId);
      }
      real w = samples_[i].weight;
      sampleGrad[i] = samples_[i].target ? -w * b / (o * (o + b)) : w / (o + b);
    }
  }
};

REGISTER_LAYER(nce, NCELayer);

}  // namespace paddle
