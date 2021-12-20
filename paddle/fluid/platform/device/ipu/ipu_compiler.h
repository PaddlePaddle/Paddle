// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <popart/builder.hpp>
#include <popart/graphtransformer.hpp>
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/ipu/common.h"
#include "paddle/fluid/platform/ipu/ipu_strategy.h"

namespace paddle {
namespace platform {
namespace ipu {

class Compiler {
 public:
  Compiler();
  ~Compiler();
  void RegisterOpFunc();
  void LowerBody(const framework::ir::Graph *graph);
  void InitInputs(framework::ir::Graph *graph,
                  const std::vector<std::string> &feed_list);
  void InitOutputs(const std::vector<std::string> &fetch_list);
  void LowerWeights(const framework::ir::Graph *graph,
                    const framework::Scope *scope_);

  void InsertTensors(const std::vector<std::string> &output_names,
                     const std::vector<std::string> &tensor_ids);
  void InsertTensors(const std::vector<std::string> &output_names,
                     const std::string &tensor_id);
  void SetIpuIndexStage(const std::vector<std::string> &tensor_ids,
                        const framework::OpDesc *op_desc);
  void SetIpuIndexStage(const std::string &tensor_id,
                        const framework::OpDesc *op_desc);

  std::vector<popart::TensorId> GetInputs() { return inputs_; }
  std::vector<popart::TensorId> GetOutputs() { return outputs_; }
  std::map<std::string, popart::TensorId> GetTensors() { return tensors_; }
  std::vector<popart::TensorId> &GetWeights();

  std::string GetModelProto();
  void SetIpuStrategy(const IpuStrategy &strategy) {
    ipu_strategy_ = &strategy;
  };
  void SaveModelProto(const std::string &path);
  void SaveModelProtoNoCheck(const std::string &path);
  void ConvertProtoToFp16();

 private:
  std::vector<std::string> GetOpInputs(const framework::OpDesc *op);
  const std::vector<std::string> &GetOpOutputs(const framework::OpDesc *op);
  popart::DebugContext BuildDebugContext(const framework::OpDesc *op);

 private:
  std::unique_ptr<popart::Builder> builder_;

  using OpFunc = std::function<void(framework::OpDesc *op_desc)>;
  std::unordered_map<std::string, OpFunc> name_function_;

  // stateful variable
  std::map<std::string, popart::TensorId> tensors_;

  // feed_list_ & fetch_list save paddle tensor id
  std::vector<std::string> feed_list_;
  std::vector<std::string> fetch_list_;

  // inputs_ & outputs_ save popart tensor id
  std::vector<popart::TensorId> inputs_;
  std::vector<popart::TensorId> outputs_;

  // weights info map
  std::vector<popart::TensorId> weights_;

  std::string converted_proto_ = "";
  const IpuStrategy *ipu_strategy_ = nullptr;
};

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
