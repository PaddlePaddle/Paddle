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

#include "LearningRateScheduler.h"
#include "paddle/utils/StringUtil.h"

namespace paddle {

ClassRegistrar<LearningRateScheduler, OptimizationConfig>
    LearningRateScheduler::registrar_;

LearningRateScheduler* LearningRateScheduler::create(
    const OptimizationConfig& config) {
  return registrar_.createByType(config.learning_rate_schedule(), config);
}

// LRS stands for LearningRateScheduler

class BaseLRS : public LearningRateScheduler {
public:
  explicit BaseLRS(const OptimizationConfig& config)
      : learningRate_(config.learning_rate()),
        a_(config.learning_rate_decay_a()),
        b_(config.learning_rate_decay_b()) {}

protected:
  real learningRate_;
  real a_;
  real b_;
};

class ConstLRS : public BaseLRS {
public:
  explicit ConstLRS(const OptimizationConfig& config) : BaseLRS(config) {}
  virtual real calcLearningRate(int64_t numSamplesProcessed, int64_t pass) {
    return learningRate_;
  }
};
REGISTER_LEARNING_RATE_SCHEDULER(constant, ConstLRS);

class PolyLRS : public BaseLRS {
public:
  explicit PolyLRS(const OptimizationConfig& config) : BaseLRS(config) {}
  virtual real calcLearningRate(int64_t numSamplesProcessed, int64_t pass) {
    return learningRate_ * pow(1.0 + a_ * numSamplesProcessed, -b_);
  }
};
REGISTER_LEARNING_RATE_SCHEDULER(poly, PolyLRS);

class CaffePolyLRS : public BaseLRS {
public:
  explicit CaffePolyLRS(const OptimizationConfig& config) : BaseLRS(config) {}
  virtual real calcLearningRate(int64_t numSamplesProcessed, int64_t pass) {
    if (numSamplesProcessed > a_) {
      LOG_FIRST_N(WARNING, 1)
          << "Using caffe_poly learning rate schedule, "
          << "learning rate hits ZERO when "
          << "numSamplesProcessed > config.learning_rate_decay_b(), "
          << "training is over and you can stop it. "
          << "See common/LearningRateScheduler.cpp for more info.";
      return 0;
    } else {
      return learningRate_ * pow(1.0 - numSamplesProcessed / a_, b_);
    }
  }
};
REGISTER_LEARNING_RATE_SCHEDULER(caffe_poly, CaffePolyLRS);

class ExpLRS : public BaseLRS {
public:
  explicit ExpLRS(const OptimizationConfig& config) : BaseLRS(config) {}
  virtual real calcLearningRate(int64_t numSamplesProcessed, int64_t pass) {
    double decayRatio = (double)numSamplesProcessed / b_;
    return learningRate_ * pow(a_, decayRatio);
  }
};
REGISTER_LEARNING_RATE_SCHEDULER(exp, ExpLRS);

class DiscreteExpLRS : public BaseLRS {
public:
  explicit DiscreteExpLRS(const OptimizationConfig& config) : BaseLRS(config) {}
  virtual real calcLearningRate(int64_t numSamplesProcessed, int64_t pass) {
    int numDecays = floor(numSamplesProcessed / b_);
    return learningRate_ * pow(a_, numDecays);
  }
};
REGISTER_LEARNING_RATE_SCHEDULER(discexp, DiscreteExpLRS);

class LinearLRS : public BaseLRS {
public:
  explicit LinearLRS(const OptimizationConfig& config) : BaseLRS(config) {}
  virtual real calcLearningRate(int64_t numSamplesProcessed, int64_t pass) {
    return std::max(learningRate_ - a_ * numSamplesProcessed, b_);
  }
};
REGISTER_LEARNING_RATE_SCHEDULER(linear, LinearLRS);

/*
  specify learning rate through
  learning_rate_args = 'seg0:rate0,seg1:rate1,...,segK:rateK'
  if seg_{i-1} <= numSamples <= seg_i,
  then learning_rate = learning_rate_base * rate_i
*/
class ManualLRS : public BaseLRS {
public:
  explicit ManualLRS(const OptimizationConfig& config)
      : BaseLRS(config), currentSegment_(0), lastNum_(0) {
    std::vector<std::string> pieces;
    str::split(config.learning_rate_args(), ',', &pieces);
    rates_.reserve(pieces.size());
    std::string s1, s2;

    for (auto& piece : pieces) {
      auto pos = piece.find(':');
      CHECK(pos != std::string::npos) << "Wrong format for learning_rate_args: "
                                      << config.learning_rate_args();
      segments_.push_back(str::to<int64_t>(piece.substr(0, pos)));
      rates_.push_back(str::to<real>(piece.substr(pos + 1)));
    }
  }

  virtual real calcLearningRate(int64_t numSamplesProcessed, int64_t pass) {
    return calc(numSamplesProcessed);
  }

  real calc(int64_t num) {
    // We assume that num never decreases.
    CHECK_LE(lastNum_, num);
    lastNum_ = num;
    while (currentSegment_ < rates_.size()) {
      if (num <= segments_[currentSegment_]) {
        return learningRate_ * rates_[currentSegment_];
      }
      ++currentSegment_;
      if (currentSegment_ < rates_.size()) {
        LOG(INFO) << " learning_rate changes to "
                  << learningRate_ * rates_[currentSegment_];
      }
    }
    return learningRate_ * rates_.back();
  }

protected:
  std::vector<real> rates_;
  std::vector<int64_t> segments_;
  size_t currentSegment_;
  int64_t lastNum_;
};

REGISTER_LEARNING_RATE_SCHEDULER(manual, ManualLRS);

class PassManualLRS : public ManualLRS {
public:
  explicit PassManualLRS(const OptimizationConfig& config)
      : ManualLRS(config) {}
  virtual real calcLearningRate(int64_t numSamplesProcessed, int64_t pass) {
    return calc(pass);
  }
};

REGISTER_LEARNING_RATE_SCHEDULER(pass_manual, PassManualLRS);
}  // namespace paddle
