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

#include "paddle/math/Vector.h"
#include "paddle/parameter/Parameter.h"
#include "paddle/utils/Flags.h"
#include "paddle/utils/Util.h"

namespace paddle {

/**
 * The static pruning hook
 *
 * Static means user load a mask map before training started. This map will
 * define which link/weight between neural is disabled.
 */
class StaticPruningHook : public IParameterUpdaterHook {
public:
  /**
   * The Mask Map Header.
   * The map file started with this header.
   *
   * In Version 0, reset file will be:
   *  contains header.size bit, each bit means such weight is enabled or not.
   *    if bit is 1, then such weight is enabled.
   *  at end, the file will round to byte, and the low bits of end byte will be
   *  filled by zero.
   *
   */
  struct StaticMaskHeader {
    uint32_t version;
    size_t size;
  } __attribute__((__packed__));

  explicit StaticPruningHook(const std::string& mask_filename) : initCount_(0) {
    bool ok = this->loadMaskFile(mask_filename);
    if (!ok) {
      LOG(WARNING) << "Fail to load mask file " << mask_filename
                   << " in current directory, searching in init_model_path";
      std::string combineMaskFilename =
          path::join(FLAGS_init_model_path, mask_filename);
      CHECK(this->loadMaskFile(combineMaskFilename))
          << "Cannot load " << mask_filename << " in ./" << mask_filename
          << " and " << combineMaskFilename;
    }
    VLOG(3) << mask_filename << " mask size = " << this->mask_.size();
  }

  void update(Parameter* para) {
    updateThreadChecker_.check();
    auto& vec = para->getBuf(PARAMETER_GRADIENT);
    if (vec) {
      vec->dotMul(*maskVec_);
    }
  }

  void init(Parameter* para) {
    size_t initCount = this->initCount_.fetch_add(1);
    CHECK_EQ(initCount, 0UL) << "Currently the StaticPruningHook must invoke "
                                "in same ParamterUpdater";
    VLOG(3) << "Initialize Parameter " << para;
    SetDevice device(para->getDeviceId());

    auto maskVec = Vector::create(this->mask_.size(), false);
    {  // Initialize maskVec with float mask vector
      real* dataPtr = maskVec->getData();
      size_t i = 0;
      for (bool m : mask_) {
        dataPtr[i++] = m ? 1.0 : 0.0;
      }
    }

    // Currently just use a mask vector for hack.
    // @TODO(yuyang18): Implemented the mask operation in vector.
    if (para->useGpu()) {
      maskVec_ = Vector::create(this->mask_.size(), para->useGpu());
      maskVec_->copyFrom(*maskVec);
    } else {
      maskVec_ = maskVec;
    }

    auto& vec = para->getBuf(PARAMETER_VALUE);
    vec->dotMul(*maskVec_);
  }

private:
  bool loadMaskFile(const std::string& mask_filename) {
    std::ifstream fin;
    fin.open(mask_filename);
    if (fin.is_open()) {
      StaticMaskHeader header;
      fin.read(reinterpret_cast<char*>(&header), sizeof(StaticMaskHeader));
      CHECK_EQ(header.version, 0UL);
      mask_.resize(header.size);
      uint8_t buf;
      for (size_t i = 0; i < header.size; ++i, buf <<= 1) {
        if (i % 8 == 0) {
          fin.read(reinterpret_cast<char*>(&buf), sizeof(uint8_t));
        }
        mask_[i] = buf & 0x80;
      }
      fin.close();
      return true;
    } else {
      return false;
    }
  }

  SameThreadChecker updateThreadChecker_;
  std::atomic<size_t> initCount_;
  VectorPtr maskVec_;
  std::vector<bool> mask_;
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
                   StringIntPairHasher>
    g_hookCache_;

/**
 * ParameterUpdaterHook actually factory method.
 */
static IParameterUpdaterHook* createImpl(
    const ParameterUpdaterHookConfig& config) {
  auto& type = config.type();
  if (type == "pruning") {
    if (config.has_purning_mask_filename()) {
      return new StaticPruningHook(config.purning_mask_filename());
    }
  }
  return nullptr;
}

std::shared_ptr<IParameterUpdaterHook> IParameterUpdaterHook::create(
    const ParameterConfig& paramConfig, int idx) {
  std::pair<std::string, int> key = {paramConfig.name(), idx};
  return g_hookCache_.get(
      key, [&] { return createImpl(paramConfig.update_hooks(idx)); });
}

}  // namespace paddle
