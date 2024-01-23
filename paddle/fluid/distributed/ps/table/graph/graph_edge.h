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
#ifdef PADDLE_WITH_CUDA
#include <cuda_fp16.h>
#endif
#include <cstddef>
#include <cstdint>
#include <vector>
#include "paddle/common/macros.h"
namespace paddle {
namespace distributed {

class GraphEdgeBlob {
 public:
  GraphEdgeBlob() {}
  virtual ~GraphEdgeBlob() {}
  size_t size() { return id_arr.size(); }
  virtual void add_edge(int64_t id, float weight);
  int64_t get_id(int idx) { return id_arr[idx]; }
#ifdef PADDLE_WITH_CUDA
  virtual half get_weight(int idx UNUSED) { return (half)(1.0); }
#else
  virtual float get_weight(int idx UNUSED) { return 1.0; }
#endif
  std::vector<int64_t>& export_id_array() { return id_arr; }

 protected:
  std::vector<int64_t> id_arr;
};

class WeightedGraphEdgeBlob : public GraphEdgeBlob {
 public:
  WeightedGraphEdgeBlob() {}
  virtual ~WeightedGraphEdgeBlob() {}
  virtual void add_edge(int64_t id, float weight);
#ifdef PADDLE_WITH_CUDA
  virtual half get_weight(int idx) { return weight_arr[idx]; }
#else
  virtual float get_weight(int idx) { return weight_arr[idx]; }
#endif

 protected:
#ifdef PADDLE_WITH_CUDA
  std::vector<half> weight_arr;
#else
  std::vector<float> weight_arr;
#endif
};
}  // namespace distributed
}  // namespace paddle
