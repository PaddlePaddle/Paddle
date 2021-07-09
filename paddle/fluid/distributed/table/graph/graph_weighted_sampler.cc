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

#include "paddle/fluid/distributed/table/graph/graph_weighted_sampler.h"
#include "paddle/fluid/framework/generator.h"
#include <iostream>
#include <unordered_map>
#include <memory>
namespace paddle {
namespace distributed {

void RandomSampler::build(GraphEdgeBlob *edges) { this->edges = edges; }

std::vector<int> RandomSampler::sample_k(int k, const std::shared_ptr<std::mt19937_64> rng) {
  int n = edges->size();
  if (k > n) {
    k = n;
  }
  std::vector<int> sample_result;
  for(int i = 0;i < k;i ++ ) {
      sample_result.push_back(i);
  }
  if (k == n) {
      return sample_result;
  }

  std::uniform_int_distribution<int> distrib(0, n - 1);
  std::unordered_map<int, int> replace_map;

  for(int i = 0; i < k; i ++) {
    int j = distrib(*rng);
    if (j >= i) {
      //  buff_nid[offset + i] = nid[j] if m.find(j) == m.end() else nid[m[j]]
      auto iter_j = replace_map.find(j);
      if(iter_j == replace_map.end()) {
        sample_result[i] = j;
      } else {
        sample_result[i] = iter_j -> second;
      }
      //  m[j] = i if m.find(i) == m.end() else m[i]
      auto iter_i = replace_map.find(i);
      if(iter_i == replace_map.end()) {
        replace_map[j] = i;
      } else {
        replace_map[j] = (iter_i -> second);
      }
    } else {
      sample_result[i] = sample_result[j];
      // buff_nid[offset + j] = nid[i] if m.find(i) == m.end() else nid[m[i]] 
      auto iter_i = replace_map.find(i);
      if(iter_i == replace_map.end()) {
        sample_result[j] = i;
      } else {
        sample_result[j] = (iter_i -> second);
      }
    }
  }
  return sample_result;
}

WeightedSampler::WeightedSampler() {
  left = nullptr;
  right = nullptr;
  edges = nullptr;
}

WeightedSampler::~WeightedSampler() {
  if (left != nullptr) {
    delete left;
    left = nullptr;
  }
  if (right != nullptr) {
    delete right;
    right = nullptr;
  }
}

void WeightedSampler::build(GraphEdgeBlob *edges) {
  if (left != nullptr) {
    delete left;
    left = nullptr;
  }
  if (right != nullptr) {
    delete right;
    right = nullptr;
  }
  return build_one((WeightedGraphEdgeBlob *)edges, 0, edges->size());
}

void WeightedSampler::build_one(WeightedGraphEdgeBlob *edges, int start,
                                int end) {
  count = 0;
  this->edges = edges;
  if (start + 1 == end) {
    left = right = nullptr;
    idx = start;
    count = 1;
    weight = edges->get_weight(idx);

  } else {
    left = new WeightedSampler();
    right = new WeightedSampler();
    left->build_one(edges, start, start + (end - start) / 2);
    right->build_one(edges, start + (end - start) / 2, end);
    weight = left->weight + right->weight;
    count = left->count + right->count;
  }
}
std::vector<int> WeightedSampler::sample_k(int k, const std::shared_ptr<std::mt19937_64> rng) {
  if (k >= count) {
    k = count;
    std::vector<int> sample_result;
    for (int i = 0; i < k; i++) {
      sample_result.push_back(i);
    }
    return sample_result;
  }
  std::vector<int> sample_result;
  float subtract;
  std::unordered_map<WeightedSampler *, float> subtract_weight_map;
  std::unordered_map<WeightedSampler *, int> subtract_count_map;
  std::uniform_real_distribution<float> distrib(0, 1.0);
  while (k--) {
    float query_weight = distrib(*rng);
    query_weight *= weight - subtract_weight_map[this];
    sample_result.push_back(sample(query_weight, subtract_weight_map,
                                   subtract_count_map, subtract));
  }
  return sample_result;
}

int WeightedSampler::sample(
    float query_weight,
    std::unordered_map<WeightedSampler *, float> &subtract_weight_map,
    std::unordered_map<WeightedSampler *, int> &subtract_count_map,
    float &subtract) {
  if (left == nullptr) {
    subtract_weight_map[this] = weight;
    subtract = weight;
    subtract_count_map[this] = 1;
    return idx;
  }
  int left_count = left->count - subtract_count_map[left];
  int right_count = right->count - subtract_count_map[right];
  float left_subtract = subtract_weight_map[left];
  int return_idx;
  if (right_count == 0 ||
      left_count > 0 && left->weight - left_subtract >= query_weight) {
    return_idx = left->sample(query_weight, subtract_weight_map,
                              subtract_count_map, subtract);
  } else {
    return_idx =
        right->sample(query_weight - (left->weight - left_subtract),
                      subtract_weight_map, subtract_count_map, subtract);
  }
  subtract_weight_map[this] += subtract;
  subtract_count_map[this]++;
  return return_idx;
}
}
}
