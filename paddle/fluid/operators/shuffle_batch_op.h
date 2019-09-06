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

#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "glog/logging.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/assert.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
#if defined(PADDLE_WITH_CUDA)
template <typename T>
using Vector = framework::Vector<T>;
#else
template <typename T>
using Vector = framework::CPUVector<T>;
#endif

template <typename T>
class ShuffleBatchKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *x = context.Input<LoDTensor>("X");
    auto *out = context.Output<LoDTensor>("Out");
    auto *shuffleidx = context.Output<LoDTensor>("ShuffleIdx");
    std::vector<int64_t> shuffleorder =
        context.Attr<std::vector<int64_t>>("ShuffleOrder");

    size_t x_embed_size = x->dims()[x->dims().size() - 1];
    size_t elem_size = 1;
    for (size_t i = 0; i < x->dims().size() - 1; i++) elem_size *= x->dims()[i];

    std::vector<int64_t> idx_vec;  // record shuffled order
    if (shuffleorder.empty()) {
      idx_vec.reserve(elem_size);
      for (size_t i = 0; i < elem_size; i++) {
        idx_vec.push_back(i);
      }
      std::random_device rd;
      std::shuffle(idx_vec.begin(), idx_vec.end(), std::mt19937(rd()));
    } else {
      PADDLE_ENFORCE_EQ(elem_size, shuffleorder.size(),
                        "The product of input dims (except last dim) must "
                        "equal to shuffle order length."
                        "%ld vs %ld",
                        elem_size, shuffleorder.size());
      idx_vec = shuffleorder;
    }

    // ShuffleIdx record shuffle order
    shuffleidx->Resize(framework::make_ddim({(int64_t)idx_vec.size()}));
    auto *shuffleidx_data =
        shuffleidx->mutable_data<int64_t>(context.GetPlace());
    for (size_t i = 0; i < idx_vec.size(); i++) {
      shuffleidx_data[i] = idx_vec[i];
    }
    // copy data according to idx_vec
    auto *x_data = x->data<T>();
    auto *out_data = out->mutable_data<T>(context.GetPlace());
    for (size_t i = 0; i < elem_size; i++) {
      memcpy(out_data + idx_vec[i] * x_embed_size, x_data + i * x_embed_size,
             x_embed_size * sizeof(T));
    }
  }
};

template <typename T>
class ShuffleBatchGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *out_grad = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto *shuffleidx = context.Input<LoDTensor>("ShuffleIdx");
    auto *x_grad = context.Output<LoDTensor>(framework::GradVarName("X"));

    size_t embed_size = out_grad->dims()[out_grad->dims().size() - 1];
    size_t elem_size = 1;
    for (size_t i = 0; i < out_grad->dims().size() - 1; i++)
      elem_size *= out_grad->dims()[i];

    std::vector<size_t> idx_vec_grad(elem_size);
    auto *shuffleidx_data = shuffleidx->data<int64_t>();
    for (size_t i = 0; i < idx_vec_grad.size(); i++) {
      idx_vec_grad[shuffleidx_data[i]] = i;
    }

    // copy data according to idx_vec_grad
    auto *out_grad_data = out_grad->data<T>();
    auto *x_grad_data = x_grad->mutable_data<T>(context.GetPlace());
    for (size_t i = 0; i < elem_size; i++) {
      memcpy(x_grad_data + idx_vec_grad[i] * embed_size,
             out_grad_data + i * embed_size, embed_size * sizeof(T));
    }
  }
};
}  // namespace operators
}  // namespace paddle
