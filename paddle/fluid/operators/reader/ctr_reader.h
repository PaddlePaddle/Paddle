// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <vector>
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/operators/reader/lod_tensor_blocking_queue.h"

namespace paddle {
namespace operators {
namespace reader {

class CTRReader : public framework::FileReader {
 public:
  explicit CTRReader(const std::shared_ptr<LoDTensorBlockingQueue>& queue)
      : framework::FileReader() {
    PADDLE_ENFORCE(queue != nullptr, "LoDTensorBlockingQueue must not be null");
    queue_ = queue;
  }

  void ReadNext(std::vector<framework::LoDTensor>* out) override {
    bool success;
    *out = queue_->Pop(&success);
    if (!success) out->clear();
  }

  ~CTRReader() { queue_->Close(); }

  void Shutdown() override { queue_->Close(); }

  void Start() override { queue_->ReOpen(); }

 private:
  std::shared_ptr<LoDTensorBlockingQueue> queue_;
};

}  // namespace reader
}  // namespace operators
}  // namespace paddle
