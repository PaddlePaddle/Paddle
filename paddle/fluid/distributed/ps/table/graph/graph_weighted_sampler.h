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
#include <ctime>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/distributed/ps/table/graph/graph_edge.h"
namespace paddle {
namespace distributed {

class Sampler {
 public:
  virtual ~Sampler() {}
  virtual void build(GraphEdgeBlob *edges) = 0;
  virtual std::vector<int> sample_k(
      int k, const std::shared_ptr<std::mt19937_64> rng) = 0;
};

class RandomSampler : public Sampler {
 public:
  virtual ~RandomSampler() {}
  virtual void build(GraphEdgeBlob *edges);
  virtual std::vector<int> sample_k(int k,
                                    const std::shared_ptr<std::mt19937_64> rng);
  GraphEdgeBlob *edges;
};

class WeightedSampler : public Sampler {
 public:
  WeightedSampler();
  virtual ~WeightedSampler();
  WeightedSampler *left, *right;
  float weight;
  int count;
  int idx;
  GraphEdgeBlob *edges;
  virtual void build(GraphEdgeBlob *edges);
  virtual void build_one(WeightedGraphEdgeBlob *edges, int start, int end);
  virtual std::vector<int> sample_k(int k,
                                    const std::shared_ptr<std::mt19937_64> rng);

 private:
  int sample(float query_weight,
             std::unordered_map<WeightedSampler *, float> &subtract_weight_map,
             std::unordered_map<WeightedSampler *, int> &subtract_count_map,
             float &subtract);
};
}  // namespace distributed
}  // namespace paddle
