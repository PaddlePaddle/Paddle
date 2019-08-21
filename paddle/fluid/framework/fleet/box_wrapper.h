/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <vector>
#include "paddle/fluid/framework/fleet/boxps.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {

class BoxWrapper {
 public:
  virtual ~BoxWrapper() {}
  BoxWrapper() {}

  void FeedPass(const std::vector<uint64_t>& feasgin_to_box) const;
  void BeginPass() const;
  void EndPass() const;
  void PullSparse(const paddle::platform::Place& place,
                  const std::vector<const uint64_t*>& keys,
                  const std::vector<float*>& values,
                  const std::vector<int64_t>& slot_lengths,
                  const int hidden_size);
  void PushSparseGrad(const paddle::platform::Place& place,
                      const std::vector<const uint64_t*>& keys,
                      const std::vector<const float*>& grad_values,
                      const std::vector<int64_t>& slot_lengths,
                      const int hidden_size);

  static std::shared_ptr<BoxWrapper> GetInstance() {
    if (nullptr == s_instance_) {
      // If main thread is guaranteed to init this, this lock can be removed
      static std::mutex mutex;
      std::lock_guard<std::mutex> lock(mutex);
      if (nullptr == s_instance_) {
        s_instance_.reset(new paddle::framework::BoxWrapper());
        s_instance_->boxps_ptr_.reset(new paddle::boxps::FakeBoxPS());
        // TODO(hutuxian): should be exposed from pybind
        s_instance_->boxps_ptr_->InitializeCPU(nullptr, 0);
      }
    }
    return s_instance_;
  }

 private:
  static std::shared_ptr<paddle::boxps::BoxPSBase> boxps_ptr_;
  static std::shared_ptr<BoxWrapper> s_instance_;
  int GetDate() const;

 protected:
  static bool is_initialized_;  // no use now
};

}  // end namespace framework
}  // end namespace paddle
