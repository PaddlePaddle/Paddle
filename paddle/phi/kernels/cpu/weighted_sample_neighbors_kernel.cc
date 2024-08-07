// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/weighted_sample_neighbors_kernel.h"

#include <cmath>
#include <queue>
#include <vector>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
struct GraphWeightedNode {
  T node_id;
  float weight_key;
  T eid;
  GraphWeightedNode() {
    node_id = 0;
    weight_key = 0;
    eid = 0;
  }
  GraphWeightedNode(T node_id, float weight_key, T eid = 0)
      : node_id(node_id), weight_key(weight_key), eid(eid) {}

  GraphWeightedNode(const GraphWeightedNode<T>& other) : weight_key(0) {
    if (this != &other) {
      this->node_id = other.node_id;
      this->weight_key = other.weight_key;
      this->eid = other.eid;
    }
  }

  GraphWeightedNode& operator=(const GraphWeightedNode<T>& other) {
    if (this != &other) {
      this->node_id = other.node_id;
      this->weight_key = other.weight_key;
      this->eid = other.eid;
      return *this;
    }

    return *this;
  }
  friend bool operator>(const GraphWeightedNode<T>& n1,
                        const GraphWeightedNode<T>& n2) {
    return n1.weight_key > n2.weight_key;
  }
};

template <typename T>
void SampleWeightedNeighbors(
    std::vector<T>& out_src,  // NOLINT
    const std::vector<float>& out_weight,
    std::vector<T>& out_eids,  // NOLINT
    int sample_size,
    std::mt19937& rng,                                         // NOLINT
    std::uniform_real_distribution<float>& dice_distribution,  // NOLINT
    bool return_eids) {
  std::priority_queue<phi::GraphWeightedNode<T>,
                      std::vector<phi::GraphWeightedNode<T>>,
                      std::greater<phi::GraphWeightedNode<T>>>  // NOLINT
      min_heap;
  for (size_t i = 0; i < out_src.size(); i++) {
    float weight_key = log2(dice_distribution(rng)) * (1 / out_weight[i]);
    if (static_cast<int>(i) < sample_size) {
      if (!return_eids) {
        min_heap.push(phi::GraphWeightedNode<T>(out_src[i], weight_key));
      } else {
        min_heap.push(
            phi::GraphWeightedNode<T>(out_src[i], weight_key, out_eids[i]));
      }
    } else {
      const phi::GraphWeightedNode<T>& small = min_heap.top();
      phi::GraphWeightedNode<T> cmp;
      if (!return_eids) {
        cmp = GraphWeightedNode<T>(out_src[i], weight_key);
      } else {
        cmp = GraphWeightedNode<T>(out_src[i], weight_key, out_eids[i]);
      }
      bool flag = cmp > small;
      if (flag) {
        min_heap.pop();
        min_heap.push(cmp);
      }
    }
  }

  int cnt = 0;
  while (!min_heap.empty()) {
    const phi::GraphWeightedNode<T>& tmp = min_heap.top();
    out_src[cnt] = tmp.node_id;
    if (return_eids) {
      out_eids[cnt] = tmp.eid;
    }
    cnt++;
    min_heap.pop();
  }
}

template <typename T>
void SampleNeighbors(const T* row,
                     const T* col_ptr,
                     const float* edge_weight,
                     const T* eids,
                     const T* input,
                     std::vector<T>* output,
                     std::vector<int>* output_count,
                     std::vector<T>* output_eids,
                     int sample_size,
                     int bs,
                     bool return_eids) {
  std::vector<std::vector<T>> out_src_vec;
  std::vector<std::vector<float>> out_weight_vec;
  std::vector<std::vector<T>> out_eids_vec;
  // `sample_cumsum_sizes` record the start position and end position
  // after sampling.
  std::vector<int> sample_cumsum_sizes(bs + 1);
  // `total_neighbors` the size of output after sample.
  int total_neighbors = 0;
  sample_cumsum_sizes[0] = total_neighbors;
  for (int i = 0; i < bs; i++) {
    T node = input[i];
    int cap = col_ptr[node + 1] - col_ptr[node];
    int k = cap > sample_size ? sample_size : cap;
    total_neighbors += k;
    sample_cumsum_sizes[i + 1] = total_neighbors;
    std::vector<T> out_src;
    out_src.resize(cap);
    out_src_vec.emplace_back(out_src);
    std::vector<float> out_weight;
    out_weight.resize(cap);
    out_weight_vec.emplace_back(out_weight);
    if (return_eids) {
      std::vector<T> out_eids;
      out_eids.resize(cap);
      out_eids_vec.emplace_back(out_eids);
    }
  }

  output_count->resize(bs);
  output->resize(total_neighbors);
  if (return_eids) {
    output_eids->resize(total_neighbors);
  }

  std::random_device rd;
  std::mt19937 rng{rd()};
  std::uniform_real_distribution<float> dice_distribution(0, 1);

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  // Sample the neighbors in parallelism.
  for (int i = 0; i < bs; i++) {
    T node = input[i];
    T begin = col_ptr[node], end = col_ptr[node + 1];
    int cap = end - begin;
    if (sample_size < cap) {  // sample_size < neighbor_len
      std::copy(row + begin, row + end, out_src_vec[i].begin());
      std::copy(
          edge_weight + begin, edge_weight + end, out_weight_vec[i].begin());
      if (return_eids) {
        std::copy(eids + begin, eids + end, out_eids_vec[i].begin());
      }
      SampleWeightedNeighbors(out_src_vec[i],
                              out_weight_vec[i],
                              out_eids_vec[i],
                              sample_size,
                              rng,
                              dice_distribution,
                              return_eids);
      *(output_count->data() + i) = sample_size;
    } else {  // sample_size >= neighbor_len, directly copy
      std::copy(row + begin, row + end, out_src_vec[i].begin());
      if (return_eids) {
        std::copy(eids + begin, eids + end, out_eids_vec[i].begin());
      }
      *(output_count->data() + i) = cap;
    }
  }

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  // Copy the results parallelism
  for (int i = 0; i < bs; i++) {
    int k = sample_cumsum_sizes[i + 1] - sample_cumsum_sizes[i];
    std::copy(out_src_vec[i].begin(),
              out_src_vec[i].begin() + k,
              output->data() + sample_cumsum_sizes[i]);
    if (return_eids) {
      std::copy(out_eids_vec[i].begin(),
                out_eids_vec[i].begin() + k,
                output_eids->data() + sample_cumsum_sizes[i]);
    }
  }
}

template <typename T, typename Context>
void WeightedSampleNeighborsKernel(const Context& dev_ctx,
                                   const DenseTensor& row,
                                   const DenseTensor& col_ptr,
                                   const DenseTensor& edge_weight,
                                   const DenseTensor& x,
                                   const paddle::optional<DenseTensor>& eids,
                                   int sample_size,
                                   bool return_eids,
                                   DenseTensor* out,
                                   DenseTensor* out_count,
                                   DenseTensor* out_eids) {
  const T* row_data = row.data<T>();
  const T* col_ptr_data = col_ptr.data<T>();
  const float* weights_data = edge_weight.data<float>();
  const T* x_data = x.data<T>();
  const T* eids_data =
      (eids.get_ptr() == nullptr ? nullptr : eids.get_ptr()->data<T>());
  int bs = static_cast<int>(x.dims()[0]);

  std::vector<T> output;
  std::vector<int> output_count;
  std::vector<T> output_eids;

  SampleNeighbors<T>(row_data,
                     col_ptr_data,
                     weights_data,
                     eids_data,
                     x_data,
                     &output,
                     &output_count,
                     &output_eids,
                     sample_size,
                     bs,
                     return_eids);

  if (return_eids) {
    out_eids->Resize({static_cast<int>(output_eids.size())});
    T* out_eids_data = dev_ctx.template Alloc<T>(out_eids);
    std::copy(output_eids.begin(), output_eids.end(), out_eids_data);
  }

  out->Resize({static_cast<int>(output.size())});
  T* out_data = dev_ctx.template Alloc<T>(out);
  std::copy(output.begin(), output.end(), out_data);
  out_count->Resize({bs});
  int* out_count_data = dev_ctx.template Alloc<int>(out_count);
  std::copy(output_count.begin(), output_count.end(), out_count_data);
}

}  // namespace phi

PD_REGISTER_KERNEL(weighted_sample_neighbors,
                   CPU,
                   ALL_LAYOUT,
                   phi::WeightedSampleNeighborsKernel,
                   int,
                   int64_t) {}
