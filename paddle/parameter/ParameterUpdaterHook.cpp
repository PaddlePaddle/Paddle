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

#include "ParameterUpdaterHook.h"

#include <algorithm>
#include <atomic>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include <iostream>

#include "paddle/math/Vector.h"
#include "paddle/parameter/Parameter.h"
#include "paddle/utils/Flags.h"
#include "paddle/utils/Util.h"

namespace paddle {

class ParameterPruningHook : public IParameterUpdaterHook {
public:
  explicit ParameterPruningHook() : initCount_(0) {}

  virtual void update(Parameter *para) {
    updateThreadChecker_.check();
    auto &vec = para->getBuf(PARAMETER_GRADIENT);
    if (vec) {
      vec->dotMul(*maskVec_);
    }
  }

  static bool sortPairAscend(const std::pair<real, size_t> &pair1,
                             const std::pair<real, size_t> &pair2) {
    return pair1.first > pair2.first;
  }


protected:
  std::atomic<size_t> initCount_;
  SameThreadChecker updateThreadChecker_;
  VectorPtr maskVec_;
};

/**
 * The static pruning hook
 * Static means user specify a sparsity_ratio before training started, and the
 * network will prune the parameters based on the sparsity_ratio. More details
 * can be found https://arxiv.org/pdf/1506.02626.pdf.
 */

class StaticPruningHook : public ParameterPruningHook {
public:
  explicit StaticPruningHook(const ParameterUpdaterHookConfig &hookConfig)
      : ParameterPruningHook() {
    this->sparsityRatio_ = hookConfig.sparsity_ratio();
  }

  void generateMask(Parameter *para, size_t nonZeroNum) {
    VectorPtr maskTemp = Vector::create(para->getSize(), false);
    maskTemp->zeroMem();
    real *maskTempData = maskTemp->getData();

    VectorPtr paraVec = para->getBuf(PARAMETER_VALUE);
    VectorPtr paraCpuCopy = Vector::create(para->getSize(), false);
    paraCpuCopy->copyFrom(*paraVec);
    std::vector<std::pair<real, size_t>> param;

    for (size_t i = 0; i < para->getSize(); i++)
      param.push_back(std::make_pair(fabs(paraCpuCopy->getData()[i]), i));

    std::partial_sort(
        param.begin(), param.begin() + nonZeroNum, param.end(), sortPairAscend);

    for (size_t i = 0; i < nonZeroNum; i++) maskTempData[param[i].second] = 1.0;

    if (para->useGpu()) {
      this-> maskVec_ = Vector::create(para->getSize(), para->useGpu());
      this-> maskVec_->copyFrom(*maskTemp);
    } else {
      this-> maskVec_ = maskTemp;
    }
  }

  void preprocess(Parameter *para, size_t currentPass, size_t currentBatch) override {}
  void init(Parameter *para) override {
    size_t initCount = this->initCount_.fetch_add(1);
    CHECK_EQ(initCount, 0UL) << "Currently the StaticPruningHook must invoke "
                                "in same ParamterUpdater";
    VLOG(3) << "Initialize Parameter " << para;
    SetDevice device(para->getDeviceId());

    size_t nonZeroNum = para->getSize() * (1 - sparsityRatio_);
    this->generateMask(para, nonZeroNum);

    auto &paraVec = para->getBuf(PARAMETER_VALUE);
    paraVec->dotMul(*this->maskVec_);
  }

private:
  real sparsityRatio_;
};

class DynamicPruningHook : public ParameterPruningHook {
public:
  explicit DynamicPruningHook(const ParameterUpdaterHookConfig &hookConfig)
      : ParameterPruningHook() {
    this->upperBound_ = hookConfig.upper_bound();
    this->interPass_ = hookConfig.inter_pass();
    this->endPass_ = hookConfig.end_pass();
  }

  void init(Parameter *para) override {
    // init mask
    size_t initCount = this->initCount_.fetch_add(1);
    CHECK_EQ(initCount, 0UL) << "Currently the StaticPruningHook must invoke "
                                "in same ParamterUpdater";
    VLOG(3) << "Initialize Parameter " << para;
    this->maskVec_ = Vector::create(para->getSize(), para->useGpu());
    this->maskVec_->reset(1.0);

    /*
    real *data = this->maskVec_->getData();
    for (size_t i = 0; i < para->getSize(); i++){
        std::cout  << data[i] << " " ;
    }
    */
  }

  void generateMask(Parameter *para, size_t ZeroNumDiff) {
    VectorPtr maskTemp = Vector::create(para->getSize(), false);
    maskTemp->copyFrom(*this->maskVec_);
    real *maskTempData = maskTemp->getData();

    VectorPtr paraVec = para->getBuf(PARAMETER_VALUE);
    VectorPtr paraCpuCopy = Vector::create(para->getSize(), false);
    paraCpuCopy->copyFrom(*paraVec);
    std::vector<std::pair<real, size_t>> param;

    for (size_t i = 0; i < para->getSize(); i++)
      param.push_back(std::make_pair(fabs(paraCpuCopy->getData()[i]), i));

    std::sort(
        param.begin(), param.end(), sortPairAscend);
    
    size_t numDiff = 0;
    for (int i  = para->getSize() - 1; i >= 0 && numDiff < ZeroNumDiff; i--) {
        if (maskTempData[param[i].second] == 1.0){
            maskTempData[param[i].second] = 0.0;
            numDiff += 1;
        }
    }
    
    if (para->useGpu()) {
      this-> maskVec_->copyFrom(*maskTemp);
    } else {
      this-> maskVec_ = maskTemp;
    }
  }
 
  void update(Parameter *para) override{/*do nothing*/}

  void preprocess(Parameter *para, size_t currentPass, size_t currentBatch) override {
    if (currentPass % interPass_ == 0 && currentPass <= endPass_ && currentBatch  == 0) {
      real boundWeight =
          this->upperBound_ / std::log(this->endPass_ / (real)this->interPass_ + 2);
      real sparsityRatio =
          boundWeight * std::log(2 + currentPass / (real)interPass_);
      real sparsityRatioDiff =
          boundWeight * (std::log(2 + currentPass / (real)interPass_) - 
                         std::log(1 + currentPass / (real)interPass_));
      size_t zeroNumDiff = para->getSize() * (sparsityRatioDiff);
      this->generateMask(para, zeroNumDiff);
      std::cout << para->getName() << " Current sparsity ratio: " <<
       sparsityRatio <<" " << zeroNumDiff<<std::endl;
    }
    auto &paraVec = para->getBuf(PARAMETER_VALUE);
    paraVec->dotMul(*this->maskVec_);
      /*
    VectorPtr paraCopyCpu = Vector::create(para->getSize(), false);
    paraCopyCpu->copyFrom(*paraVec);
    real *data = paraCopyCpu->getData();
    size_t sum_non = 0;
      for(size_t i = 0; i < para->getSize(); i++){
          if(data[i] != 0.0)
          sum_non += 1;
      }
    std::cout<<"sum_non: " <<sum_non << " " << para->getSize()<< std::endl;
   */ 

  }

private:
  real upperBound_;
  size_t interPass_;
  size_t endPass_;
};

IParameterUpdaterHook::IParameterUpdaterHook() {}
IParameterUpdaterHook::~IParameterUpdaterHook() {}

/**
 * A Hasher used by g_hooks.
 *
 * Use the independent hasher intendedly. There is a hasher in PServer for hash
 * ParameterBlock. But not to use same hasher to reduce dependency.
 *
 * May be extracted to Util.h to unify the hasher.
 */
class StringIntPairHasher {
public:
  size_t operator()(const std::pair<std::string, int> &k) const {
    return intHasher_(strHasher_(k.first) + k.second);
  }

private:
  std::hash<std::string> strHasher_;
  std::hash<int> intHasher_;
};

static WeakKVCache<std::pair<std::string, int>,
                   IParameterUpdaterHook,
                   StringIntPairHasher>
    g_hookCache_;

/**
 * ParameterUpdaterHook actually factory method.
 */
static IParameterUpdaterHook *createImpl(
    const ParameterUpdaterHookConfig &config) {
  auto &type = config.type();
  if (type == "pruning") {
    return new StaticPruningHook(config);
  } else if (type == "dpruning") {
    return new DynamicPruningHook(config);
  }

  LOG(FATAL) << "Unknown Hook type:  " << type;
  return nullptr;
}

std::shared_ptr<IParameterUpdaterHook> IParameterUpdaterHook::create(
    const ParameterConfig &paramConfig, int idx) {
  std::pair<std::string, int> key = {paramConfig.name(), idx};
  return g_hookCache_.get(
      key, [&] { return createImpl(paramConfig.update_hooks(idx)); });
}

}  // namespace paddle
