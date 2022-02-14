/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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
#include <map>
#include <random>
#include <utility>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {
namespace data {

using Tensor = framework::Tensor;

constexpr size_t dim_bitset_size = 64;

class RandomFlipGenerator {
  public:
    RandomFlipGenerator(int seed, float prob)
      : distribution_(prob),
      seed_(seed) {
        if (seed != 0) rng_.seed(seed);
        else rng_.seed(time(0));
      }
    
    ~RandomFlipGenerator() = default;

    bool Generate() { return distribution_(rng_); }

  private:
    std::bernoulli_distribution distribution_;
    int seed_;
    std::mt19937 rng_;
};

std::map<int, std::unique_ptr<RandomFlipGenerator>> seed_to_generator_;

static RandomFlipGenerator* CreateRandomFlipGenerator(int seed, float prob) {
  auto iter = seed_to_generator_.find(seed);
  if (iter == seed_to_generator_.end()) {
    seed_to_generator_[seed] = std::unique_ptr<RandomFlipGenerator>(
                                new RandomFlipGenerator(seed, prob));
  }

  return seed_to_generator_[seed].get();
}

template <typename T>
class RandomFlipCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    LOG(ERROR) << "RandomFlipCPUKernel enter";
    const Tensor* x = ctx.Input<Tensor>("X");
    Tensor* out = ctx.Output<Tensor>("Out");

    auto prob = ctx.Attr<float>("probability");
    auto seed = ctx.Attr<int>("seed");
    
    auto* data = out->mutable_data<bool>(ctx.GetPlace());
    auto* generator = CreateRandomFlipGenerator(seed, prob);
    for (int64_t i = 0; i < x->dims()[0]; i++) {
      data[i] = generator->Generate();
    }
    LOG(ERROR) << "RandomFlipCPUKernel finish";
  }
};

}  // namespace data
}  // namespace operators
}  // namespace paddle
