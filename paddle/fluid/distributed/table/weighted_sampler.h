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
#include <unordered_map>
#include <vector>
namespace paddle {
namespace distributed {

class WeightedObject {
 public:
  WeightedObject() {}
  virtual ~WeightedObject() {}
  virtual uint64_t get_id() = 0;
  virtual float get_weight() = 0;
};

class Sampler {
public:
  virtual ~Sampler() {}
  virtual void build(std::vector<WeightedObject*>* edges) = 0;
  virtual std::vector<WeightedObject *> sample_k(int k) = 0;
};

class RandomSampler: public Sampler {
public:
  virtual ~RandomSampler() {}
  virtual void build(std::vector<WeightedObject*>* edges);
  virtual std::vector<WeightedObject *> sample_k(int k);
  std::vector<WeightedObject*>* edges;
};

class WeightedSampler: public Sampler {
 public:
  WeightedSampler();
  virtual ~WeightedSampler();
  WeightedSampler *left, *right;
  WeightedObject *object;
  int count;
  float weight;
  virtual void build(std::vector<WeightedObject*>* edges);
  virtual void build_one(WeightedObject **v, int start, int end);
  virtual std::vector<WeightedObject *> sample_k(int k);

 private:
  WeightedObject *sample(
      float query_weight,
      std::unordered_map<WeightedSampler *, float> &subtract_weight_map,
      std::unordered_map<WeightedSampler *, int> &subtract_count_map,
      float &subtract);
};
}
}
