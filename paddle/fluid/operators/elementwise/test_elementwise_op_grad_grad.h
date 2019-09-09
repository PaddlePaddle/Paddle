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

#include <algorithm>
#include <cstdlib>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {

void Memcpy(void *dst, const void *src, size_t n, bool copy_to_gpu) {
  if (copy_to_gpu) {
#ifdef PADDLE_WITH_CUDA
    PADDLE_ENFORCE(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice));
#else
    PADDLE_THROW("Not compiled with cuda");
#endif
  } else {
    std::memcpy(dst, src, n);
  }
}

// currently, this test class only support same dims
template <typename T>
class TestElementwiseOpGradGrad {
 public:
  TestElementwiseOpGradGrad(const std::string &op_type_,
                            const platform::Place &place_,
                            const framework::DDim &dims_,
                            const std::vector<std::string> &inputs_,
                            const std::vector<std::string> &outputs_)
      : op_type(op_type_),
        place(place_),
        dims(dims_),
        inputs(inputs_),
        outputs(outputs_) {}

  void InitVarInScope(std::string var_name) {
    in_out_tensors[var_name] =
        scope.Var(var_name)->GetMutable<framework::LoDTensor>();
    in_out_tensors[var_name]->Resize(dims);
    in_out_tensors[var_name]->mutable_data<T>(place);
  }

  void InitFeedData(std::string var_name, size_t size) {
    // random
    std::uniform_real_distribution<T> dist(static_cast<T>(10.0),
                                           static_cast<T>(20.0));
    std::mt19937 engine;
    std::vector<T> data(size);
    for (size_t i = 0; i < size; ++i) {
      data[i] = dist(engine);
    }
    feed_datas[var_name] = data;
  }

  void Setup() {
    size_t numel = static_cast<size_t>(framework::product(dims));
    for (auto var_name : inputs) {
      InitVarInScope(var_name);
      InitFeedData(var_name, numel);
    }
    for (auto var_name : outputs) {
      InitVarInScope(var_name);
    }
    // feeding: copy data to tensor, out tensor need init?
    bool is_gpu_place = platform::is_gpu_place(place);
    auto bytes = sizeof(T) * numel;
    for (auto &var_name : inputs) {
      auto tensor_ptr = in_out_tensors[var_name]->data<T>();
      auto feed_data_ptr = feed_datas[var_name].data();
      Memcpy(tensor_ptr, feed_data_ptr, bytes, is_gpu_place);
    }

    // calculate expected outputs
    ComputeExpectedOuts();
  }

  bool Check() {
    Setup();
    auto op = CreateTestOp();
    op->Run(this->scope, this->place);
    platform::DeviceContextPool::Instance().Get(this->place)->Wait();
    framework::LoDTensor cpu_out;
    PADDLE_ENFORCE(scope.kids().empty());
    bool all_equal = true;
    for (auto &out_name : outputs) {
      auto &out_tensor = scope.FindVar(out_name)->Get<framework::LoDTensor>();
      if (platform::is_gpu_place(place)) {
        framework::TensorCopySync(out_tensor, platform::CPUPlace(), &cpu_out);
      } else {
        cpu_out = out_tensor;
      }
      auto *out_ptr = cpu_out.data<T>();
      size_t numel = static_cast<size_t>(framework::product(dims));
      auto is_equal =
          std::equal(out_ptr, out_ptr + numel, expected_outs[out_name].data());
      if (!is_equal) {
        all_equal = false;
        break;
      }
    }
    return all_equal;
  }
  virtual std::unique_ptr<framework::OperatorBase> CreateTestOp() = 0;
  virtual void ComputeExpectedOuts() = 0;
  virtual ~TestElementwiseOpGradGrad() {}

 protected:
  std::string op_type;
  platform::Place place;
  framework::DDim dims;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::map<std::string, paddle::framework::LoDTensor *> in_out_tensors;
  std::map<std::string, std::vector<T>> feed_datas;
  std::map<std::string, std::vector<T>> expected_outs;
  framework::Scope scope;
};

}  // namespace operators
}  // namespace paddle
