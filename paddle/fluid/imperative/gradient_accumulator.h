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

#include <memory>
#include <utility>
#include <vector>
#include "paddle/fluid/imperative/layer.h"

namespace paddle {
namespace imperative {

class GradientAccumulator {
 public:
  explicit GradientAccumulator(VarBase* var) : var_(var) {}

  virtual void Add(std::shared_ptr<VarBase> var, size_t trace_id) = 0;

  virtual ~GradientAccumulator() = default;

  inline void IncreaseRefCnt() { ++ref_cnt_; }

  inline size_t RefCnt() const { return ref_cnt_; }

 protected:
  VarBase* var_;
  size_t ref_cnt_{0};
};

class EagerGradientAccumulator : public GradientAccumulator {
 public:
  using GradientAccumulator::GradientAccumulator;

  void Add(std::shared_ptr<VarBase> var, size_t trace_id) override;

 private:
  size_t cur_cnt_{0};
};

class SortedGradientAccumulator : public GradientAccumulator {
 public:
  using GradientAccumulator::GradientAccumulator;

  void Add(std::shared_ptr<VarBase> var, size_t trace_id) override;

 private:
  std::vector<std::pair<std::shared_ptr<VarBase>, size_t>> tmp_grad_vars_;
};

}  // namespace imperative
}  // namespace paddle
