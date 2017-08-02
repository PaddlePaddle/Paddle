/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserve.

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

#include "ActivationFunction.h"
#include "paddle/parameter/Argument.h"

#include "paddle/gserver/layers/MkldnnBase.h"
// #include "paddle/gserver/layers/MkldnnMemory.h"

namespace paddle {

/**
 * @brief Base class of MKLDNN Activation.
 *
 */
class MkldnnActivation {
public:
  // mkldnn
  std::shared_ptr<mkldnn::memory> botData_;
  std::shared_ptr<mkldnn::memory> botDiff_;
  std::shared_ptr<mkldnn::memory> topData_;
  std::shared_ptr<mkldnn::memory> topDiff_;

public:
  MkldnnActivation()
      : botData_(nullptr),
        botDiff_(nullptr),
        topData_(nullptr),
        topDiff_(nullptr) {}

  virtual ~MkldnnActivation() {}

  /**
   * each dnn layer should have function
   * to reset forward
   */
  virtual void resetFwd(const Argument& arg,
                        std::shared_ptr<void> topDataMD) = 0;

  /**
   * each dnn layer should have function
   * to reset backward
   */
  virtual void resetBwd(const Argument& arg,
                        std::shared_ptr<void> topDiffMD) = 0;
};

}  // namespace paddle
