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

#include <glog/logging.h>
#include <functional>
#include <string>
#include "OptimizerConfig.pb.h"
#include "lr_policy.h"
#include "serialization.h"
#include "tensor.h"

namespace paddle {
namespace optimizer {

class ParameterOptimizer {
 public:
  /**
   * @brief  update hook for algorithm need to traverse parameter more than
   * once.
   */
  ParameterOptimizer(Tensor *parameter, LrPolicy *lr)
      : parameter_(parameter), lr_policy_(lr), num_sample_passed_(0) {}
  virtual ~ParameterOptimizer() {
    delete parameter_;
    delete lr_policy_;
  }

  static ParameterOptimizer *Create(const std::string &config_proto,
                                    Tensor *parameter);
  virtual void Update(const Tensor *gradient) = 0;
  virtual float *get_weight(int *param_size) const;
  virtual std::string SerializeState() = 0;
  virtual void DeserializeState(const std::string &state) = 0;

 protected:
  Tensor *parameter_;
  // learning rate policy
  LrPolicy *lr_policy_;
  uint64_t num_sample_passed_;
};

}  // namespace optimizer
}  // namespace paddle
