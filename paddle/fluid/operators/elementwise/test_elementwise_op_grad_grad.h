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
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {

// currently, this test class only support same dims
template <typename T>
class TestElementwiseOpGradGrad {
 public:
  TestElementwiseOpGradGrad(const std::string &op_type,
                            const platform::Place &place,
                            const framework::DDim &dims,
                            const std::vector<std::string> &inputs,
                            const std::vector<std::string> &outputs)
      : op_type_(op_type),
        place_(place),
        dims_(dims),
        inputs_(inputs),
        outputs_(outputs) {}

  void InitVarInScope(std::string var_name) {
    in_out_tensors_[var_name] =
        scope_.Var(var_name)->template GetMutable<framework::LoDTensor>();
    in_out_tensors_[var_name]->Resize(dims_);
    in_out_tensors_[var_name]->template mutable_data<T>(place_);
  }

  void InitFeedData(std::string var_name, size_t size) {
    // generate random data
    std::uniform_real_distribution<T> dist(static_cast<T>(10.0),
                                           static_cast<T>(20.0));
    std::mt19937 engine;
    std::vector<T> data(size);
    for (size_t i = 0; i < size; ++i) {
      data[i] = dist(engine);
    }
    feed_datas_[var_name] = data;
  }

  void Setup() {
    size_t numel = static_cast<size_t>(phi::product(dims_));
    // init vars in scope and feed inputs
    for (auto in_name : inputs_) {
      InitVarInScope(in_name);
      InitFeedData(in_name, numel);
    }
    for (auto out_name : outputs_) {
      InitVarInScope(out_name);
    }

    // feeding: copy data to tensor, out tensor don't need init
    auto bytes = sizeof(T) * numel;
    for (auto &in_name : inputs_) {
      auto dst = in_out_tensors_[in_name]->template data<T>();
      auto src = feed_datas_[in_name].data();
      auto src_place = platform::CPUPlace();
      if (platform::is_cpu_place(place_)) {
        auto dst_place = place_;
        memory::Copy(dst_place, dst, src_place, src, bytes);
      } else if (platform::is_gpu_place(place_)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        auto dst_place = place_;
        memory::Copy(dst_place, dst, src_place, src, bytes, nullptr);
#else
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Check your paddle version, current version is not compiled with "
            "cuda"));
#endif
      }
    }

    // calculate expected outputs
    ComputeExpectedOuts();
  }

  bool Check() {
    Setup();
    auto op = CreateTestOp();
    op->Run(scope_, place_);
    platform::DeviceContextPool::Instance().Get(place_)->Wait();
    framework::LoDTensor cpu_out;
    PADDLE_ENFORCE_EQ(scope_.kids().empty(), true,
                      platform::errors::InvalidArgument(
                          "The scope can not have the child scopes,"
                          "please check your code."));

    // get outputs from scope and compare them with expected_outs
    bool all_equal = true;
    for (auto &out_name : outputs_) {
      auto &out_tensor =
          scope_.FindVar(out_name)->template Get<framework::LoDTensor>();
      if (platform::is_gpu_place(place_)) {
        framework::TensorCopySync(out_tensor, platform::CPUPlace(), &cpu_out);
      } else {
        cpu_out = out_tensor;
      }
      auto *out_ptr = cpu_out.data<T>();
      size_t numel = static_cast<size_t>(phi::product(dims_));
#ifdef PADDLE_WITH_HIP
      auto is_equal = std::equal(
          out_ptr, out_ptr + numel, expected_outs_[out_name].data(),
          [](const float &l, const float &r) { return fabs(l - r) < 1e-8; });
#else
      auto is_equal =
          std::equal(out_ptr, out_ptr + numel, expected_outs_[out_name].data());
#endif
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
  std::string op_type_;
  platform::Place place_;
  framework::DDim dims_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  std::map<std::string, paddle::framework::LoDTensor *> in_out_tensors_;
  std::map<std::string, std::vector<T>> feed_datas_;
  std::map<std::string, std::vector<T>> expected_outs_;
  framework::Scope scope_;
};

}  // namespace operators
}  // namespace paddle
