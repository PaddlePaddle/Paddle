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
using namespace std;
namespace paddle {
namespace distributed {
class WeightedObject {
 public:
  WeightedObject() {}
  virtual ~WeightedObject() {}
  virtual unsigned long long get_id() { return id; }
  virtual double get_weight() { return weight; }

  virtual void set_id(unsigned long long id) { this->id = id; }
  virtual void set_weight(double weight) { this->weight = weight; }

 private:
  unsigned long long id;
  double weight;
};

class WeightedSampler {
 public:
  WeightedSampler *left, *right;
  WeightedObject *object;
  int count;
  double weight;
  void build(WeightedObject **v, int start, int end);
  vector<WeightedObject *> sample_k(int k);

 private:
  WeightedObject *sample(
      double query_weight,
      unordered_map<WeightedSampler *, double> &subtract_weight_map,
      unordered_map<WeightedSampler *, int> &subtract_count_map,
      double &subtract);
};
}
}
