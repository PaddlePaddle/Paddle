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

#include "paddle/fluid/distributed/table/weighted_sampler.h"
#include <iostream>
namespace paddle {
namespace distributed {
void WeightedSampler::build(WeightedObject **v, int start, int end) {
  count = 0;
  if (start + 1 == end) {
    left = right = NULL;
    weight = v[start]->get_weight();
    object = v[start];
    count = 1;

  } else {
    left = new WeightedSampler();
    right = new WeightedSampler();
    left->build(v, start, start + (end - start) / 2);
    right->build(v, start + (end - start) / 2, end);
    weight = left->weight + right->weight;
    count = left->count + right->count;
  }
}
std::vector<WeightedObject *> WeightedSampler::sample_k(int k) {
  if (k > count) {
    k = count;
  }
  std::vector<WeightedObject *> sample_result;
  float subtract;
  std::unordered_map<WeightedSampler *, float> subtract_weight_map;
  std::unordered_map<WeightedSampler *, int> subtract_count_map;
  struct timespec tn;
  clock_gettime(CLOCK_REALTIME, &tn);
  srand(tn.tv_nsec);
  while (k--) {
    float query_weight = rand() % 100000 / 100000.0;
    query_weight *= weight - subtract_weight_map[this];
    sample_result.push_back(sample(query_weight, subtract_weight_map,
                                   subtract_count_map, subtract));
  }
  return sample_result;
}
WeightedObject *WeightedSampler::sample(
    float query_weight,
    std::unordered_map<WeightedSampler *, float> &subtract_weight_map,
    std::unordered_map<WeightedSampler *, int> &subtract_count_map,
    float &subtract) {
  if (left == NULL) {
    subtract_weight_map[this] = weight;
    subtract = weight;
    subtract_count_map[this] = 1;
    return object;
  }
  int left_count = left->count - subtract_count_map[left];
  int right_count = right->count - subtract_count_map[right];
  float left_subtract = subtract_weight_map[left];
  WeightedObject *return_id;
  if (right_count == 0 ||
      left_count > 0 && left->weight - left_subtract >= query_weight) {
    return_id = left->sample(query_weight, subtract_weight_map,
                             subtract_count_map, subtract);
  } else {
    return_id =
        right->sample(query_weight - (left->weight - left_subtract),
                      subtract_weight_map, subtract_count_map, subtract);
  }
  subtract_weight_map[this] += subtract;
  subtract_count_map[this]++;
  return return_id;
}
}
}
