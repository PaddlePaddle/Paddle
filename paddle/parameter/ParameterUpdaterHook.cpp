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

#include <atomic>
#include <fstream>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "paddle/math/Vector.h"
#include "paddle/parameter/Parameter.h"
#include "paddle/utils/Flags.h"
#include "paddle/utils/Util.h"

namespace paddle {

/**
 * The static pruning hook
 * Static means user specific a sparsity_ratio map before training started. The
 * network will
 * hold the sparsity_ratio maximum numbers of parameters, and cut off the rest.
 */

class StaticPruningHook : public IParameterUpdaterHook {
public:
  explicit StaticPruningHook(const ParameterUpdaterHookConfig& hookConfig)
      : initCount_(0) {
    sparsityRatio_ = hookConfig.sparsity_ratio();
  }

  static bool sortPairAscend(const std::pair<real, size_t>& pair1,
                             const std::pair<real, size_t>& pair2) {
    return pair1.first > pair2.first;
  }

  void update(Parameter* para) {
    updateThreadChecker_.check();
    auto& vec = para->getBuf(PARAMETER_GRADIENT);
    if (vec) {
      vec->dotMul(*maskVec_);
    }
  }

  void generateMask(Parameter* para) {
    VectorPtr vec = para->getBuf(PARAMETER_VALUE);
    maskTemp_ = Vector::create(para->getSize(), false);
    maskTemp_->zeroMem();
    real* dataPtr = maskTemp_->getData();

    VectorPtr vecCpu = Vector::create(para->getSize(), false);
    vecCpu->copyFrom(*vec);
    std::vector<std::pair<real, size_t>> param;

    for (size_t i = 0; i < para->getSize(); i++)
      param.push_back(std::make_pair(fabs(vecCpu->getData()[i]), i));
    std::sort(param.begin(), param.end(), sortPairAscend);

    for (size_t i = 0; i < para->getSize() * sparsityRatio_; i++)
      dataPtr[param[i].second] = 1.0;
  }

  void init(Parameter* para) {
    generateMask(para);
    size_t initCount = this->initCount_.fetch_add(1);
    CHECK_EQ(initCount, 0UL) << "Currently the StaticPruningHook must invoke "
                                "in same ParamterUpdater";
    VLOG(3) << "Initialize Parameter " << para;
    SetDevice device(para->getDeviceId());

    // Currently just use a mask vector for hack.
    // @TODO(yuyang18): Implemented the mask operation in vector.
    if (para->useGpu()) {
      maskVec_ = Vector::create(para->getSize(), para->useGpu());
      maskVec_->copyFrom(*maskTemp_);
    } else {
      maskVec_ = maskTemp_;
    }

    auto& vec = para->getBuf(PARAMETER_VALUE);
    vec->dotMul(*maskVec_);
  }

private:
  SameThreadChecker updateThreadChecker_;
  std::atomic<size_t> initCount_;
  VectorPtr maskVec_;
  VectorPtr maskTemp_;
  real sparsityRatio_;
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
  size_t operator()(const std::pair<std::string, int>& k) const {
    return intHasher_(strHasher_(k.first) + k.second);
  }

private:
  std::hash<std::string> strHasher_;
  std::hash<int> intHasher_;
};

static WeakKVCache<std::pair<std::string, int>,
                   IParameterUpdaterHook,
                   StringIntPairHasher> g_hookCache_;

/**
 * ParameterUpdaterHook actually factory method.
 */
static IParameterUpdaterHook* createImpl(
    const ParameterUpdaterHookConfig& config) {
  auto& type = config.type();
  if (type == "pruning") {
    if (config.has_sparsity_ratio())
      return new StaticPruningHook(config);
    else
      LOG(FATAL) << "There must be sparsity_ratio parameter for " << type
                 << " Hook";
  }

  LOG(FATAL) << "Unknown Hook type:  " << type;
  return nullptr;
}

std::shared_ptr<IParameterUpdaterHook> IParameterUpdaterHook::create(
    const ParameterConfig& paramConfig, int idx) {
  std::pair<std::string, int> key = {paramConfig.name(), idx};
  return g_hookCache_.get(
      key, [&] { return createImpl(paramConfig.update_hooks(idx)); });
}

}  // namespace paddle
