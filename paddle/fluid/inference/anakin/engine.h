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

enum class DataType;
enum class Place;
class Tensor;

template <typename TargetT, ::anakin::Precision PrecisionType,
          ::anakin::OpRunType RunType = ::anakin::OpRunType::ASYNC>
class AnakinEngine /*: public EngineBase */ {
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

enum class DataType { kUnk, kFloat32, kFloat64, kInt32 };
enum class Place { kCpu, kGpu, kUnk };

class Tensor final {
 public:
  Tensor() = default;
  ~Tensor() {
    if (length_ > 0) {
      delete[] static_cast<char *>(data_);
      data_ = nullptr;
    }
  }
  void Reshape(const std::vector<int> &shape);
  const std::vector<int> &shape() const;
  void SetName(const std::string &name);
  const std::string &name() const;
  void SetDataType(const DataType dtype);
  DataType dtype() const;

  template <typename T>
  T *mutable_data(Place place) {
    int length = std::accumulate(shape_.begin(), shape_.end(), 1,
                                 std::multiplies<int>()) *
                 sizeof(T);
    if (place_ == Place::kCpu && place == Place::kCpu) {
      if (length_ < length) {
        length_ = length;
        delete[] static_cast<char *>(data_);
        data_ = new char[length_];
      }
      return static_cast<T *>(data_);
    }

#ifdef PADDLE_WITH_CUDA
    if (place == Place::kCpu) {
      cudaFree(data_);
      data_ = new char[length];
    } else if (place_ == Place::kCpu) {
      delete[] static_cast<char *>(data_);
      cudaMalloc((void **)&data_, length);
    } else {
      if (length_ < length) {
        length_ = length;
        cudaFree(data_);
        cudaMalloc((void **)&data_, length);
      }
    }
#endif
    place_ = place;
    return static_cast<T *>(data_);
  }

  template <typename T>
  T *data(Place *place, int *size) const {
    *place = place_;
    *size = length_;
    return static_cast<T *>(data_);
  }

 private:
  std::string name_;
  std::vector<int> shape_;
  Place place_;
  DataType dtype_;
  int length_{0};
  void *data_{nullptr};
};

template class AnakinEngine<::anakin::saber::NV, ::anakin::Precision::FP32>;
template class AnakinEngine<::anakin::saber::X86, ::anakin::Precision::FP32>;
}  // namespace anakin
}  // namespace inference
}  // namespace paddle
