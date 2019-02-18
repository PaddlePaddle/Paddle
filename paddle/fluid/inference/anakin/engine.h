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

#include <memory>
#include <set>
#include <string>
#include <vector>
#include "framework/core/types.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/inference/engine.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "saber/saber_types.h"

namespace anakin {
template <typename Ttype, Precision Ptype, OpRunType RunType>
class Net;

template <typename Ttype, Precision Ptype>
class Graph;
}

namespace paddle {
namespace inference {
namespace anakin {

enum class DataType;
enum class Place;
class Tensor;

template <typename TargetT, anakin::Precision PrecisionType,
          anakin::OpRunType RunType = anakin::OpRunType::ASYNC>
class AnakinEngine : public EngineBase {
 public:
  void DeclareInputs(const std::vector<std::string> &inputs);
  void DeclareOutputs(const std::vector<std::string> &outputs);

  void AddOp(const std::string &name, const std::string &type,
             const std::vector<std::string> &inputs,
             const std::vector<std::string> &outputs);

  template <typename AttrType>
  void AddOpAttr(const std::string &op_name, const std::string &attr_name,
                 const AttrType &attr_value) {
    PADDLE_ENFORCE(graph_->AddOp(op_name, attr_name, attr_value),
                   "Add operation's attribution.");
  }

  void AddVar(const std::string &id, DataType dtype,
              const std::vector<int> &shape);
  Tensor *AddWeight(const std::string &id, const Tensor &v);

  std::unique_ptr<AnakinEngine> Clone();

  void FreezeNetwork();

  void Execute(const std::vector<Tensor *> &inputs,
               const std::vector<Tensor *> &outputs);

 private:
  AnakinEngine(const AnakinEngine &);
  using AnakinNetT = anakin::Net<TargetT, PrecisionType, RunType>;
  using AnakinGraphT = anakin::Graph<TargetT, PrecisionType>;

  std::unique_ptr<AnakinNetT> engine_;
  std::unique_ptr<AnakinGraphT> graph_;

  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
};

class Tensor {
 public:
  void Resize(const std::vector<int> &shape);
  void SetName(const std::string &name);
  const std::string &name() const;
  template <typename T>
  T *mutable_data(Place place);

  template <typename T>
  T *data(Place *place, size_t size) const;

  DataType dtype() const;
  const std::vector<int> &shape() const;

 private:
  std::string name_;
  std::vector<int> shape_;
  void *data_;
};

enum class DataType { kUnk, kFloat32, kFloat64, kInt32 };
enum class Place { kCpu, kGpu };

template class AnakinEngine<anakin::saber::NV, anakin::Precision::FP32>;
template class AnakinEngine<anakin::saber::X86, anakin::Precision::FP32>;
}  // namespace anakin
}  // namespace inference
}  // namespace paddle
