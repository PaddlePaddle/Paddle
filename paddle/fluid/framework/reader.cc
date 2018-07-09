//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/reader.h"

namespace paddle {
namespace framework {

void ReaderBase::ReadNext(std::vector<LoDTensor> *out) {
  std::lock_guard<std::mutex> lock(mu_);
  PADDLE_ENFORCE_EQ(status_, ReaderStatus::kRunning);
  ReadNextImpl(out);
}

void ReaderBase::Shutdown() {
  std::lock_guard<std::mutex> lock(mu_);
  if (status_ != ReaderStatus::kStopped) {
    ShutdownImpl();
    status_ = ReaderStatus::kStopped;
  }
}

void ReaderBase::Start() {
  std::lock_guard<std::mutex> lock(mu_);
  if (status_ != ReaderStatus::kRunning) {
    StartImpl();
    status_ = ReaderStatus::kRunning;
  }
}

ReaderBase::~ReaderBase() { Shutdown(); }

}  // namespace framework
}  // namespace paddle
