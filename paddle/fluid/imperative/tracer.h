// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <atomic>
#include <future>  // NOLINT
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "ThreadPool.h"
#include "paddle/fluid/imperative/engine.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace imperative {

class Tracer {
  DISABLE_COPY_AND_ASSIGN(Tracer);

 public:
  Tracer() : engine_(new BasicEngine()) {}

  ~Tracer() = default;

  void TraceOp(const std::string& type, const NameVarBaseMap& ins,
               const NameVarBaseMap& outs, framework::AttributeMap attrs,
               const platform::Place& place, bool trace_bacward);

  bool ComputeRequiredGrad(const NameVarBaseMap& ins,
                           const NameVarBaseMap& outs, bool trace_backward);

  void TraceBackward(const std::shared_ptr<OpBase>& fwd_op,
                     const framework::OpDesc& fwd_op_desc,
                     const NameVarBaseMap& ins, const NameVarBaseMap& outs);
  Engine* GetDefaultEngine() const { return engine_.get(); }

 private:
  static size_t GenerateUniqueId() {
    static std::atomic<size_t> id{0};
    return id.fetch_add(1);
  }

 private:
  std::unique_ptr<Engine> engine_;
};

}  // namespace imperative
}  // namespace paddle
