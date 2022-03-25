/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <algorithm>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class LoDResetKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    auto* in = ctx.Input<framework::LoDTensor>("X");
    auto* lod_t = ctx.Input<framework::LoDTensor>("Y");
    bool append = ctx.Attr<bool>("append");

    framework::TensorCopy(*in, in->place(), out);

    std::vector<int> level0;
    if (lod_t) {
      if (lod_t->lod().size() > 0) {
        auto y_lod = lod_t->lod();
        auto last_level = y_lod[y_lod.size() - 1];
        PADDLE_ENFORCE_EQ(
            static_cast<int64_t>(last_level.back()), in->dims()[0],
            platform::errors::InvalidArgument(
                "The last value of Input(Y)'s last level LoD should be equal "
                "to the first dimension of Input(X). But received the last "
                "value of Input(Y)'s last level LoD is %d, the first dimension "
                "of Input(X) is %d.",
                static_cast<int64_t>(last_level.back()), in->dims()[0]));
        out->set_lod(y_lod);
        return;  // early return, since lod already set
      } else {
        auto* lod = lod_t->data<int>();
        framework::Tensor lod_cpu;
        if (platform::is_gpu_place(lod_t->place())) {
          framework::TensorCopySync(*lod_t, platform::CPUPlace(), &lod_cpu);
          lod = lod_cpu.data<int>();
        }
        level0 = std::vector<int>(lod, lod + lod_t->numel());
      }
    } else {
      level0 = ctx.Attr<std::vector<int>>("target_lod");
    }

    PADDLE_ENFORCE_GT(
        level0.size(), 1UL,
        platform::errors::InvalidArgument(
            "The size of target LoD should be greater than 1. But received the "
            "size of target LoD is %d.",
            level0.size()));
    PADDLE_ENFORCE_EQ(static_cast<int64_t>(level0[0]), 0,
                      platform::errors::InvalidArgument(
                          "Target LoD should be a vector starting from 0. But "
                          "target LoD starts from %d.",
                          static_cast<int64_t>(level0[0])));
    PADDLE_ENFORCE_EQ(
        static_cast<int64_t>(level0.back()), in->dims()[0],
        platform::errors::InvalidArgument(
            "The last value of 'Target LoD''s last level LoD should be equal "
            "to the first dimension of Input(X). But received the 'Target LoD' "
            "is %s, Input(X)'s shape is is %s.",
            phi::make_ddim(level0), in->dims()));
    for (size_t i = 0; i < level0.size() - 1; ++i) {
      PADDLE_ENFORCE_GE(level0[i + 1], level0[i],
                        platform::errors::InvalidArgument(
                            "'Target LoD' should be an ascending "
                            "vector. But received the Target LoD is %s.",
                            phi::make_ddim(level0)));
    }

    // cast level0 to size_t
    std::vector<size_t> ulevel0(level0.size(), 0);
    std::transform(level0.begin(), level0.end(), ulevel0.begin(),
                   [](int a) { return static_cast<size_t>(a); });
    if (append) {
      auto* out_lod = out->mutable_lod();
      out_lod->push_back(ulevel0);
    } else {
      framework::LoD target_lod;
      target_lod.push_back(ulevel0);
      out->set_lod(target_lod);
    }
  }
};

template <typename DeviceContext, typename T>
class LoDResetGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* d_out = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* d_x = ctx.Output<framework::Tensor>(framework::GradVarName("X"));

    framework::TensorCopy(*d_out, d_out->place(), d_x);
  }
};
}  // namespace operators
}  // namespace paddle
