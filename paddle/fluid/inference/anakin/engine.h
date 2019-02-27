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
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/inference/engine.h"
#include "paddle/fluid/inference/utils/singleton.h"

#include "framework/core/net/net.h"
#include "framework/core/types.h"
#include "framework/graph/graph.h"
#include "saber/saber_types.h"

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
 public:
  explicit AnakinEngine(bool need_summary = false);
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

  std::unique_ptr<AnakinEngine> Clone();
  void Freeze();
  void Optimize();
  void Execute(const std::map<std::string, framework::LoDTensor *> &inputs,
               const std::map<std::string, framework::LoDTensor *> &outputs);

 private:
  using NetT = ::anakin::Net<TargetT, PrecisionType, RunType>;
  using GraphT = ::anakin::graph::Graph<TargetT, PrecisionType>;
  std::unique_ptr<GraphT> graph_;
  std::unique_ptr<NetT> net_;
};

}  // namespace anakin
}  // namespace inference
}  // namespace paddle
