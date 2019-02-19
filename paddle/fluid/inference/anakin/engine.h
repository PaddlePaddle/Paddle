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

namespace graph {
template <typename Ttype, Precision Ptype>
class Graph;
}  // namespace graph
}  // namespace anakin

namespace paddle {
namespace inference {
namespace anakin {

enum class DataType;
enum class Place;
class Tensor;

template <typename TargetT, ::anakin::Precision PrecisionType,
          ::anakin::OpRunType RunType = ::anakin::OpRunType::ASYNC>
class AnakinEngine : public EngineBase {
 public:
  AnakinEngine();
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

  std::vector<Tensor> Execute(const std::vector<Tensor *> &inputs);

 private:
  using NetT = ::anakin::Net<TargetT, PrecisionType, RunType>;
  using GraphT = ::anakin::graph::Graph<TargetT, PrecisionType>;
  std::unique_ptr<GraphT> graph_;
  std::unique_ptr<NetT> engine_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
};

class Tensor final {
 public:
  Tensor() = default;
  ~Tensor();
  void Resize(const std::vector<int> &shape) { shape_ = shape; }
  /*void Reshape(const std::vector<int> &shape) {
    shape_ = shape;
    size_ = std::accumulate(shape_.begin(), shape_.end(), 1,
                            std::multiplies<int>());
  }*/
  int size() const { return size_; }
  const std::vector<int> &shape() const { return shape_; }
  void SetName(const std::string &name) { name_ = name; }
  const std::string &name() const { return name_; }
  void SetPlace(const Place place) { place_ = place; }
  Place place() const { return place_; }
  void SetDataType(const DataType dtype) { dtype_ = dtype; }
  DataType dtype() const { return dtype_; }

  template <typename T>
  T *mutable_data(Place place) {
    place_ = place;
    return static_cast<T *>(data_);
  }

  /*template <typename T>
      T *data(Place *place, size_t *size) const {
          *place = place_;
      }*/
  template <typename T>
  const T *data() const {
    return static_cast<const T *>(data);
  }

 private:
  std::string name_;
  std::vector<int> shape_;
  Place place_;
  DataType dtype_;
  int size_;
  void *data_;
};

enum class DataType { kUnk, kFloat32, kFloat64, kInt32 };
enum class Place { kCpu, kGpu, kUnk };

template class AnakinEngine<::anakin::saber::NV, ::anakin::Precision::FP32>;
template class AnakinEngine<::anakin::saber::X86, ::anakin::Precision::FP32>;
}  // namespace anakin
}  // namespace inference
}  // namespace paddle
