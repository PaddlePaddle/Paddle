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

#include "paddle/fluid/imperative/engine.h"

#include <mutex>  // NOLINT
#include <vector>

#include "glog/logging.h"

namespace paddle {
namespace imperative {

static std::once_flag init_engine;
static Engine* engine;

class DummyEngine : public Engine {
 public:
  void Enqueue(Runnable* runnable) override {
    queued_runnables_.push_back(runnable);
  }

  size_t Size() const override { return queued_runnables_.size(); }

  void Sync() override {
    for (Runnable* l : queued_runnables_) {
      LOG(INFO) << "running " << reinterpret_cast<void*>(l);
    }
    queued_runnables_.clear();
  }

 private:
  std::vector<Runnable*> queued_runnables_;
};

Engine* GetEngine() {
  std::call_once(init_engine, []() { engine = new DummyEngine(); });
  return engine;
}

}  // namespace imperative
}  // namespace paddle
