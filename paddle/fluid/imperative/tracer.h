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

#include "ThreadPool.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace imperative {

class Tracer {
  DISABLE_COPY_AND_ASSIGN(Tracer);

 public:
  explicit Tracer(const framework::BlockDesc* block_desc)
      : block_desc_(block_desc) {}

  ~Tracer() = default;

  void TraceOp(const std::string& type, const NameVarBaseMap& ins,
               const NameVarBaseMap& outs, framework::AttributeMap attrs,
               const platform::Place& place, bool trace_bacward);

  void TraceOp(const framework::OpDesc& op_desc, const NameVarBaseMap& ins,
               const NameVarBaseMap& outs, const platform::Place& place,
               bool trace_backward);

  void RemoveOp(size_t id) {
    PADDLE_ENFORCE(ops_.erase(id) > 0, "Op with id %d is not inside tracer",
                   id);
  }

  void RemoveOp(OpBase* op) {
    PADDLE_ENFORCE_NOT_NULL(op, "Cannot remove null op");
    auto iter = ops_.find(op->id());
    PADDLE_ENFORCE(iter != ops_.end() && iter->second.get() == op,
                   "Op is not inside tracer");
    ops_.erase(iter);
  }

  void Clear() { ops_.clear(); }

 private:
  static size_t GenerateUniqueId() {
    static std::atomic<size_t> id{0};
    return id.fetch_add(1);
  }

 private:
  std::unordered_map<size_t, std::shared_ptr<OpBase>> ops_;
  const framework::BlockDesc* block_desc_;
};

}  // namespace imperative
}  // namespace paddle
