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

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/inference/engine.h"
#include "paddle/fluid/inference/utils/singleton.h"

#include "framework/core/net/net.h"
#include "framework/core/types.h"
#include "framework/graph/graph.h"
#include "framework/graph/graph_global_mem.h"
#include "saber/saber_types.h"

using anakin::Precision;
using anakin::saber::NV;

namespace anakin {

template <typename, Precision, OpRunType>
class Net;

namespace graph {
template <typename, Precision>
class Graph;
}  // namespace graph
}  // namespace anakin

namespace paddle {
namespace inference {
namespace anakin {

template <typename TargetT, ::anakin::Precision PrecisionType,
          ::anakin::OpRunType RunType = ::anakin::OpRunType::ASYNC>
class AnakinEngine {
  using NetT = ::anakin::Net<TargetT, PrecisionType, RunType>;
  using GraphT = ::anakin::graph::Graph<TargetT, PrecisionType>;

 public:
  explicit AnakinEngine(
      bool need_summary = false, int device = 0, int max_batch_size = 1,
      std::map<std::string, std::vector<int>> max_input_shape = {});
  ~AnakinEngine();
  void InitGraph();
  void SetInputShape(const std::string &name, std::vector<int> shape);
  void AddOp(const std::string &name, const std::string &type,
             const std::vector<std::string> &inputs,
             const std::vector<std::string> &outputs);

  template <typename T>
  void AddOpAttr(const std::string &op_name, const std::string &attr_name,
                 const T &attr_value) {
    PADDLE_ENFORCE(graph_->AddOpAttr(op_name, attr_name, attr_value),
                   "Add operation's attribution.");
  }
  NetT *Net() { return net_.get(); }
  GraphT *Graph() { return graph_.get(); }
  std::unique_ptr<AnakinEngine> Clone();
  const std::map<std::string, std::vector<int>> &GetMaxInputShape() {
    return max_input_shape_;
  }
  void SetMaxInputShape(std::map<std::string, std::vector<int>> shape) {
    max_input_shape_ = shape;
  }
  int GetMaxBatchSize() { return max_batch_size_; }
  void Freeze();
  void Optimize();
  void Save(std::string path) { graph_->save(path); }
  int GetDevice() { return device_; }
  // void SaveSerializedData(std::string& data) { graph_->save_to_string(data);
  // }
  // void LoadSerializedData(const std::string& data) {
  // graph_->load_from_string(data); }
  void Execute(const std::map<std::string, framework::LoDTensor *> &inputs,
               const std::map<std::string, framework::LoDTensor *> &outputs,
               cudaStream_t stream);

 private:
  int max_batch_size_;
  std::map<std::string, std::vector<int>> max_input_shape_;
  int device_;
  std::unique_ptr<GraphT> graph_;
  std::unique_ptr<NetT> net_;
};

class AnakinEngineManager {
  using AnakinNvEngineT = AnakinEngine<NV, Precision::FP32>;

 public:
  bool HasEngine(const std::string &name) const {
    if (engines_.count(name) == 0) return false;
    return engines_.at(name).get() != nullptr;
  }
  AnakinNvEngineT *Get(const std::string &name) const {
    return engines_.at(name).get();
  }

  AnakinNvEngineT *Create(
      bool need_summary, int device, int max_batch_size,
      std::map<std::string, std::vector<int>> max_input_shape,
      std::string engine_name) {
    std::unique_lock<std::mutex> lk(mut_);
    auto *p = new AnakinEngine<NV, Precision::FP32>(
        need_summary, device, max_batch_size, max_input_shape);
    engines_[engine_name].reset(p);
    return p;
  }

  void DeleteALL() {
    for (auto &item : engines_) {
      item.second.reset(nullptr);
    }
  }

 private:
  std::unordered_map<std::string, std::unique_ptr<AnakinNvEngineT>> engines_;
  std::mutex mut_;
};
}  // namespace anakin
}  // namespace inference
}  // namespace paddle
