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
#ifdef EXIT  // NOLINT
#undef EXIT  // NOLINT
#endif       // NOLINT
#include "framework/core/net/net.h"
#include "framework/core/types.h"
#include "framework/graph/graph.h"
#include "framework/graph/graph_global_mem.h"
#include "saber/saber_types.h"

using anakin::Precision;

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
      std::map<std::string, std::vector<int>> max_input_shape = {},
      std::vector<std::string> program_inputs = {},
      bool auto_config_layout = false);
  ~AnakinEngine();
  void InitNet();
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
  const std::vector<std::string> &GetScalableInputs() {
    return program_inputs_;
  }
  void SetScalableInputs(std::vector<std::string> program_inputs) {
    program_inputs_ = program_inputs;
  }
  int GetMaxBatchSize() { return max_batch_size_; }
  void Freeze();
  void Optimize();
  void RegistBlock(::anakin::PBlock<TargetT> *block_p);
  void Save(std::string path) { graph_->save(path); }
  bool IsInit() { return initialized_; }
  int GetDevice() { return device_; }
  void AddTensorScale(const std::string &tensor_name, float scale) {
    tensor_scales_[tensor_name] = scale;
  }
  std::unordered_map<std::string, float> GetTensorScales() {
    return tensor_scales_;
  }
  void Execute(const std::map<std::string, framework::LoDTensor *> &inputs,
               const std::map<std::string, framework::LoDTensor *> &outputs);
#ifdef PADDLE_WITH_CUDA
  void Execute(const std::map<std::string, framework::LoDTensor *> &inputs,
               const std::map<std::string, framework::LoDTensor *> &outputs,
               cudaStream_t stream);
#endif

 private:
  void BindInput(const std::map<std::string, framework::LoDTensor *> &inputs);

 private:
  bool initialized_{false};
  int device_;
  int max_batch_size_;
  std::map<std::string, std::vector<int>> max_input_shape_;
  std::vector<std::string> program_inputs_;
  std::unique_ptr<GraphT> graph_;
  std::unique_ptr<NetT> net_;
  static std::once_flag init_anakin_;
  std::unordered_map<std::string, float> tensor_scales_;
  // Always be false in gpu mode but true in most cpu cases.
  bool auto_config_layout_;
};

template <typename TargetT, ::anakin::Precision PrecisionType>
class AnakinEngineManager {
  using AnakinEngineT = AnakinEngine<TargetT, PrecisionType>;

 public:
  bool HasEngine(const std::string &name) const {
    if (engines_.count(name) == 0) return false;
    return engines_.at(name).get() != nullptr;
  }
  AnakinEngineT *Get(const std::string &name) const {
    return engines_.at(name).get();
  }

  AnakinEngineT *Create(bool need_summary, int device, int max_batch_size,
                        std::map<std::string, std::vector<int>> max_input_shape,
                        std::vector<std::string> program_inputs,
                        bool auto_config_layout, std::string engine_name) {
    std::unique_lock<std::mutex> lk(mut_);
    auto *p = new AnakinEngine<TargetT, PrecisionType>(
        need_summary, device, max_batch_size, max_input_shape, program_inputs,
        auto_config_layout);
    engines_[engine_name].reset(p);
    return p;
  }

  void DeleteALL() {
    for (auto &item : engines_) {
      item.second.reset(nullptr);
    }
  }

 private:
  std::unordered_map<std::string, std::unique_ptr<AnakinEngineT>> engines_;
  std::mutex mut_;
};
}  // namespace anakin
}  // namespace inference
}  // namespace paddle
