/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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
#ifdef PADDLE_WITH_GCU
#include <string>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {
namespace gcu {

class GcuStrategy {
 public:
  GcuStrategy();
  ~GcuStrategy() = default;

 public:
  void SetTargetName(std::string target) { target_ = target; }
  void SetTrainFlag(bool flag) { is_training_ = flag; }
  bool GetTrainFlag() const { return is_training_; }
  void SetBatchSize(int32_t batch_num) { batch_num_ = batch_num; }
  int32_t GetBatchSize() { return batch_num_; }

 private:
  // training flag, true for training
  bool is_training_ = false;
  std::string target_ = "pavo";
  int32_t batch_num_ = 1;
};

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
#endif
