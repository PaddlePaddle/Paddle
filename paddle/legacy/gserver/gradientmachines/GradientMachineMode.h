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
#include "GradientMachine.h"
#include "unordered_map"

namespace paddle {

class IGradientMachineMode {
 public:
  virtual ~IGradientMachineMode() {}

 public:  // interfaces
          /**
           * @brief create current mode's gradient machine by model config.
           * @param config model config
           */
  virtual GradientMachine* create(const ModelConfig& config) = 0;

  /**
   * @brief shouldBeMe the current mode of GradientMachine should be this mode.
   * @param algo training algorithm name.
   * @param trainerCount trainer count.
   * @param isLocal is local mode (without pserver)
   * @param isGpu is using gpu.
   * @return true if mode should be this mode.
   */
  virtual bool shouldBeMe(const std::string& algo,
                          size_t trainerCount,
                          bool isLocal,
                          bool isGpu) const = 0;

  /**
   * @brief Is data must be in cpu even if using gpu mode.
   * @param trainerCount trainer count
   * @return true if data must be gpu.
   */
  virtual bool isDataMustInCpu(size_t trainerCount) const = 0;

  /**
   * @brief Need not to use mini-batch method, and should train all data in one
   * batch in one pass.
   */
  virtual bool needTrainWholeDataInOneBatch() const = 0;

 public:  // static methods.
          /**
           * @brief register a custom gradient machine mode.
           * @note For user to register a custom gradient machine mode, id should >=
           * kCustom.
           * @param mode mode id.
           * @param ptr mode description object.
           */
  static void regGradientMachineMode(
      int32_t mode, std::unique_ptr<IGradientMachineMode>&& ptr) {
    modes_.insert(std::make_pair(mode, std::move(ptr)));
  }

  /**
   * @brief get custom mode from mode id.
   * @param mode mode id
   * @return mode description object.
   */
  static IGradientMachineMode* mode(int32_t mode) {
    if (modes_.find(mode) != modes_.end()) {
      return modes_[mode].get();
    } else {
      return nullptr;
    }
  }

  /**
   * @brief helper function to test trainWholeDataInOneBatch or not for mode
   */
  static bool trainWholeDataInOneBatch(int32_t mode) {
    if (modes_.find(mode) != modes_.end()) {
      return modes_[mode]->needTrainWholeDataInOneBatch();
    } else {
      return false;
    }
  }

  /**
   * @brief Try to get custom mode if we can.
   * @param [out] mode the custom mode id.
   * @param [in] algo algorithm name
   * @param [in] trainerCount trainer count.
   * @param [in] isLocal is local or not
   * @param [in] isGpu using gpu or not.
   * @return true if there is a custom mode fit these conditions.
   */
  static bool tryGetMode(int* mode,
                         const std::string& algo,
                         int32_t trainerCount,
                         bool isLocal,
                         bool isGpu) {
    for (auto it = modes_.begin(); it != modes_.end(); ++it) {
      if (it->second->shouldBeMe(algo, trainerCount, isLocal, isGpu)) {
        *mode = it->first;
        return true;
      }
    }
    return false;
  }

  /**
   * @brief helper function for data must in cpu
   */
  static bool dataMustInCpu(int32_t mode, size_t trainerCount) {
    if (modes_.find(mode) != modes_.end()) {
      return modes_[mode]->isDataMustInCpu(trainerCount);
    } else {
      // provide data to cpu if using synchronized multi-gpu gradient machine.
      return trainerCount > 1;
    }
  }

  /**
   * @brief try to create gradient machine by mode & config.
   * @return nullptr if we cannot create a gradient machine by such mode.
   */
  static GradientMachine* tryCreateGradientMachine(int32_t mode,
                                                   const ModelConfig& config) {
    auto m = IGradientMachineMode::mode(mode);
    if (m) {
      return m->create(config);
    } else {
      return nullptr;
    }
  }

 private:
  static std::unordered_map<int32_t, std::unique_ptr<IGradientMachineMode>>
      modes_;
};

}  // namespace paddle
