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
#include "paddle/fluid/framework/data_set.h"
#ifdef PADDLE_WITH_BOX_PS
#include <boxps.h>
#endif
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
#ifdef PADDLE_WITH_BOX_PS
        s_instance_->boxps_ptr_.reset(new paddle::boxps::FakeBoxPS());
#endif
      }
    }
    return s_instance_;
  }

 private:
#ifdef PADDLE_WITH_BOX_PS
  static std::shared_ptr<paddle::boxps::BoxPSBase> boxps_ptr_;
#endif
  static std::shared_ptr<BoxWrapper> s_instance_;
  int GetDate() const;
};

class BoxHelper {
 public:
  explicit BoxHelper(paddle::framework::Dataset* dataset) : dataset_(dataset) {}
  virtual ~BoxHelper() {}

  void BeginPass() {
    auto box_ptr = BoxWrapper::GetInstance();
    box_ptr->BeginPass();
  }

  void EndPass() {
    auto box_ptr = BoxWrapper::GetInstance();
    box_ptr->EndPass();
  }
  void LoadIntoMemory() {
    dataset_->LoadIntoMemory();
    FeedPass();
  }
  void PreLoadIntoMemory() {
    dataset_->PreLoadIntoMemory();
    feed_data_thread_.reset(new std::thread([&]() {
      dataset_->WaitPreLoadDone();
      FeedPass();
    }));
  }
  void WaitFeedPassDone() { feed_data_thread_->join(); }

 private:
  Dataset* dataset_;
  std::shared_ptr<std::thread> feed_data_thread_;
  // notify boxps to feed this pass feasigns from SSD to memory
  void FeedPass() {
    auto box_ptr = BoxWrapper::GetInstance();
    auto input_channel_ =
        dynamic_cast<MultiSlotDataset*>(dataset_)->GetInputChannel();
    std::vector<Record> pass_data;
    std::vector<uint64_t> feasign_to_box;
    input_channel_->ReadAll(pass_data);
    for (const auto& ins : pass_data) {
      const auto& feasign_v = ins.uint64_feasigns_;
      for (const auto feasign : feasign_v) {
        feasign_to_box.push_back(feasign.sign().uint64_feasign_);
      }
    }
    input_channel_->Open();
    input_channel_->Write(pass_data);
    input_channel_->Close();
    box_ptr->FeedPass(feasign_to_box);
  }
};

}  // end namespace framework
}  // end namespace paddle
