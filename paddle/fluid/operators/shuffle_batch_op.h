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

#include <atomic>
#include <cstring>
#include <ctime>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include "glog/logging.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/timer.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename T>
using Vector = framework::Vector<T>;

template <typename T>
class ShuffleBatchKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *x = context.Input<LoDTensor>("X");
    auto *seed = context.Input<LoDTensor>("Seed");
    auto *out = context.Output<LoDTensor>("Out");
    auto *shuffleidx = context.Output<LoDTensor>("ShuffleIdx");
    auto *seed_out = context.Output<LoDTensor>("SeedOut");

    auto x_embed_size = x->dims()[x->dims().size() - 1];
    auto elem_size = 1;
    for (auto i = 0; i < x->dims().size() - 1; i++) elem_size *= x->dims()[i];

    std::vector<int64_t> idx_vec;  // record shuffled order
    idx_vec.reserve(elem_size);
    for (auto i = 0; i < elem_size; i++) {
      idx_vec.push_back(i);
    }
    int64_t seed_int = 0;
    if (seed->IsInitialized()) {
      seed_int = *seed->data<int64_t>();
    } else {
      seed_int = context.Attr<int>("startup_seed");
    }
    std::default_random_engine engine;
    engine.seed(seed_int);

    auto custom_random_shuffle = [&idx_vec]() {
      std::random_device rnd;
      int64_t seed_tmp = rnd();
      std::default_random_engine rng(seed_tmp);
      const int n = idx_vec.size();
      std::vector<int> v(n);
      std::iota(v.begin(), v.end(), 0);
      std::vector<bool> visit(n, false);
      while (!v.empty()) {
        std::shuffle(v.begin(), v.end(), rng);
        int tmp = v.back();
        v.pop_back();
        if (v.empty()) {
          std::uniform_int_distribution<int> distr(0, n - 2);
          idx_vec[tmp] = tmp;
          std::swap(idx_vec[tmp], idx_vec[(distr(rng) + tmp + 1) % n]);
          return;
        }
        visit[tmp] = true;
        std::shuffle(v.begin(), v.end(), rng);
        int curr = v.back();
        v.pop_back();
        v.push_back(tmp);
        idx_vec[tmp] = curr;
        while (!visit[curr]) {
          visit[curr] = true;
          std::shuffle(v.begin(), v.end(), rng);
          idx_vec[curr] = v.back();
          v.pop_back();
          curr = idx_vec[curr];
        }
      }
    };
    custom_random_shuffle();
    // change shuffle to custom_random_shuffle
    // std::shuffle(idx_vec.begin(), idx_vec.end(), engine);

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
    for (auto i = 0; i < elem_size; i++) {
      memcpy(out_data + idx_vec[i] * x_embed_size, x_data + i * x_embed_size,
             x_embed_size * sizeof(T));
    }
    // set new seed
    *seed_out->mutable_data<int64_t>(framework::make_ddim({1}),
                                     context.GetPlace()) = engine();
  }
};

template <typename T>
class ShuffleBatchGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *out_grad = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto *shuffleidx = context.Input<LoDTensor>("ShuffleIdx");
    auto *x_grad = context.Output<LoDTensor>(framework::GradVarName("X"));

    auto embed_size = out_grad->dims()[out_grad->dims().size() - 1];
    auto elem_size = 1;
    for (auto i = 0; i < out_grad->dims().size() - 1; i++)
      elem_size *= out_grad->dims()[i];

    std::vector<int> idx_vec_grad(elem_size);
    auto *shuffleidx_data = shuffleidx->data<int64_t>();
    for (size_t i = 0; i < idx_vec_grad.size(); i++) {
      idx_vec_grad[shuffleidx_data[i]] = i;
    }

    // copy data according to idx_vec_grad
    auto *out_grad_data = out_grad->data<T>();
    auto *x_grad_data = x_grad->mutable_data<T>(context.GetPlace());
    for (auto i = 0; i < elem_size; i++) {
      memcpy(x_grad_data + idx_vec_grad[i] * embed_size,
             out_grad_data + i * embed_size, embed_size * sizeof(T));
    }
  }
};
}  // namespace operators
}  // namespace paddle
