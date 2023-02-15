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

#include <stdlib.h>

#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {

template <class bidiiter>
void SampleUniqueNeighbors(bidiiter begin, bidiiter end, int num_samples) {
  int left_num = std::distance(begin, end);
  std::random_device rd;
  std::mt19937 rng{rd()};
  std::uniform_int_distribution<int> dice_distribution(
      0, std::numeric_limits<int>::max());
  for (int i = 0; i < num_samples; i++) {
    bidiiter r = begin;
    int random_step = dice_distribution(rng) % left_num;
    std::advance(r, random_step);
    std::swap(*begin, *r);
    ++begin;
    --left_num;
  }
}

template <class bidiiter>
void SampleUniqueNeighborsWithEids(bidiiter src_begin,
                                   bidiiter src_end,
                                   bidiiter eid_begin,
                                   bidiiter eid_end,
                                   int num_samples) {
  int left_num = std::distance(src_begin, src_end);
  std::random_device rd;
  std::mt19937 rng{rd()};
  std::uniform_int_distribution<int> dice_distribution(
      0, std::numeric_limits<int>::max());
  for (int i = 0; i < num_samples; i++) {
    bidiiter r1 = src_begin, r2 = eid_begin;
    int random_step = dice_distribution(rng) % left_num;
    std::advance(r1, random_step);
    std::advance(r2, random_step);
    std::swap(*src_begin, *r1);
    std::swap(*eid_begin, *r2);
    ++src_begin;
    ++eid_begin;
    --left_num;
  }
}

template <typename T>
void SampleNeighbors(const T* src,
                     const T* dst_count,
                     const T* src_eids,
                     std::vector<T>* inputs,
                     std::vector<T>* outputs,
                     std::vector<T>* output_counts,
                     std::vector<T>* outputs_eids,
                     int k,
                     bool is_first_layer,
                     bool is_last_layer,
                     bool return_eids) {
  const size_t bs = inputs->size();
  // Allocate the memory of outputs
  // Collect the neighbors size
  std::vector<std::vector<T>> out_src_vec;
  std::vector<std::vector<T>> out_eids_vec;
  // `sample_cumsum_sizes` record the start position and end position after the
  //  sample.
  std::vector<size_t> sample_cumsum_sizes(bs + 1);
  size_t total_neighbors = 0;
  // `total_neighbors` the size of output after the sample
  sample_cumsum_sizes[0] = total_neighbors;
  for (size_t i = 0; i < bs; i++) {
    T node = inputs->data()[i];
    T begin = dst_count[node];
    T end = dst_count[node + 1];
    int cap = end - begin;
    int sample_size = cap > k ? k : cap;
    total_neighbors += sample_size;
    sample_cumsum_sizes[i + 1] = total_neighbors;
    std::vector<T> out_src;
    out_src.resize(cap);
    out_src_vec.emplace_back(out_src);
    if (return_eids) {
      std::vector<T> out_eids;
      out_eids.resize(cap);
      out_eids_vec.emplace_back(out_eids);
    }
  }
  if (is_first_layer) {
    PADDLE_ENFORCE_GT(total_neighbors,
                      0,
                      platform::errors::InvalidArgument(
                          "The input nodes `X` should have at "
                          "least one neighbors, but none of the "
                          "input nodes have neighbors."));
  }
  output_counts->resize(bs);
  outputs->resize(total_neighbors);
  if (return_eids) {
    outputs_eids->resize(total_neighbors);
  }

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  // Sample the neighbour parallelism
  for (size_t i = 0; i < bs; i++) {
    T node = inputs->data()[i];
    T begin = dst_count[node];
    T end = dst_count[node + 1];
    int cap = end - begin;
    if (k < cap) {
      std::copy(src + begin, src + end, out_src_vec[i].begin());
      if (return_eids) {
        std::copy(src_eids + begin, src_eids + end, out_eids_vec[i].begin());
        SampleUniqueNeighborsWithEids(out_src_vec[i].begin(),
                                      out_src_vec[i].end(),
                                      out_eids_vec[i].begin(),
                                      out_eids_vec[i].end(),
                                      k);
      } else {
        SampleUniqueNeighbors(out_src_vec[i].begin(), out_src_vec[i].end(), k);
      }
      *(output_counts->data() + i) = k;
    } else {
      std::copy(src + begin, src + end, out_src_vec[i].begin());
      if (return_eids) {
        std::copy(src_eids + begin, src_eids + end, out_eids_vec[i].begin());
      }
      *(output_counts->data() + i) = cap;
    }
  }

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  // Copy the results parallelism
  for (size_t i = 0; i < bs; i++) {
    int sample_size = sample_cumsum_sizes[i + 1] - sample_cumsum_sizes[i];
    std::copy(out_src_vec[i].begin(),
              out_src_vec[i].begin() + sample_size,
              outputs->data() + sample_cumsum_sizes[i]);
    if (return_eids) {
      std::copy(out_eids_vec[i].begin(),
                out_eids_vec[i].begin() + sample_size,
                outputs_eids->data() + sample_cumsum_sizes[i]);
    }
  }

  if (!is_last_layer) {
    std::sort(inputs->begin(), inputs->end());
    std::vector<T> outputs_sort(outputs->size());
    std::copy(outputs->begin(), outputs->end(), outputs_sort.begin());
    std::sort(outputs_sort.begin(), outputs_sort.end());
    auto outputs_sort_end =
        std::unique(outputs_sort.begin(), outputs_sort.end());
    outputs_sort.resize(std::distance(outputs_sort.begin(), outputs_sort_end));
    std::vector<T> unique_outputs(outputs_sort.size());

    auto unique_outputs_end = std::set_difference(outputs_sort.begin(),
                                                  outputs_sort.end(),
                                                  inputs->begin(),
                                                  inputs->end(),
                                                  unique_outputs.begin());

    inputs->resize(std::distance(unique_outputs.begin(), unique_outputs_end));
    std::copy(unique_outputs.begin(), unique_outputs_end, inputs->begin());
  }
}

template <typename DeviceContext, typename T>
class GraphKhopSamplerOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // 1. Get sample neighbors operators' inputs.
    auto* src = ctx.Input<phi::DenseTensor>("Row");
    auto* dst_count = ctx.Input<phi::DenseTensor>("Col_Ptr");
    auto* vertices = ctx.Input<phi::DenseTensor>("X");
    std::vector<int> sample_sizes = ctx.Attr<std::vector<int>>("sample_sizes");
    bool return_eids = ctx.Attr<bool>("return_eids");

    const T* src_data = src->data<T>();
    const T* dst_count_data = dst_count->data<T>();
    const T* p_vertices = vertices->data<T>();
    const size_t bs = vertices->dims()[0];

    // 2. Get unique input nodes(X).
    std::vector<T> inputs(bs);
    std::copy(p_vertices, p_vertices + bs, inputs.begin());
    auto unique_inputs_end = std::unique(inputs.begin(), inputs.end());
    inputs.resize(std::distance(inputs.begin(), unique_inputs_end));

    // 3. Sample neighbors. We should distinguish w/o "Eids".
    std::vector<T> outputs;
    std::vector<T> output_counts;
    std::vector<T> outputs_eids;
    std::vector<std::vector<T>> dst_vec;
    dst_vec.emplace_back(inputs);
    std::vector<std::vector<T>> outputs_vec;
    std::vector<std::vector<T>> output_counts_vec;
    std::vector<std::vector<T>> outputs_eids_vec;

    const size_t num_layers = sample_sizes.size();
    bool is_last_layer = false, is_first_layer = true;

    if (return_eids) {
      auto* src_eids = ctx.Input<phi::DenseTensor>("Eids");
      const T* src_eids_data = src_eids->data<T>();
      for (size_t i = 0; i < num_layers; i++) {
        if (i == num_layers - 1) {
          is_last_layer = true;
        }
        if (inputs.size() == 0) {
          break;
        }
        if (i > 0) {
          dst_vec.emplace_back(inputs);
          is_first_layer = false;
        }
        SampleNeighbors<T>(src_data,
                           dst_count_data,
                           src_eids_data,
                           &inputs,
                           &outputs,
                           &output_counts,
                           &outputs_eids,
                           sample_sizes[i],
                           is_first_layer,
                           is_last_layer,
                           return_eids);
        outputs_vec.emplace_back(outputs);
        output_counts_vec.emplace_back(output_counts);
        outputs_eids_vec.emplace_back(outputs_eids);
      }
    } else {
      for (size_t i = 0; i < num_layers; i++) {
        if (i == num_layers - 1) {
          is_last_layer = true;
        }
        if (inputs.size() == 0) {
          break;
        }
        if (i > 0) {
          is_first_layer = false;
          dst_vec.emplace_back(inputs);
        }
        SampleNeighbors<T>(src_data,
                           dst_count_data,
                           nullptr,
                           &inputs,
                           &outputs,
                           &output_counts,
                           &outputs_eids,
                           sample_sizes[i],
                           is_first_layer,
                           is_last_layer,
                           return_eids);
        outputs_vec.emplace_back(outputs);
        output_counts_vec.emplace_back(output_counts);
        outputs_eids_vec.emplace_back(outputs_eids);
      }
    }

    // 4. Concat intermediate sample results.
    int64_t unique_dst_size = 0, src_size = 0;
    for (size_t i = 0; i < num_layers; i++) {
      unique_dst_size += dst_vec[i].size();
      src_size += outputs_vec[i].size();
    }

    std::vector<T> unique_dst_merge(unique_dst_size);
    std::vector<T> src_merge(src_size);
    std::vector<T> dst_sample_counts_merge(unique_dst_size);
    auto unique_dst_merge_ptr = unique_dst_merge.begin();
    auto src_merge_ptr = src_merge.begin();
    auto dst_sample_counts_merge_ptr = dst_sample_counts_merge.begin();
    // TODO(daisiming): We may try to use std::move in the future.
    for (size_t i = 0; i < num_layers; i++) {
      if (i == 0) {
        unique_dst_merge_ptr = std::copy(
            dst_vec[i].begin(), dst_vec[i].end(), unique_dst_merge.begin());
        src_merge_ptr = std::copy(
            outputs_vec[i].begin(), outputs_vec[i].end(), src_merge.begin());
        dst_sample_counts_merge_ptr =
            std::copy(output_counts_vec[i].begin(),
                      output_counts_vec[i].end(),
                      dst_sample_counts_merge.begin());
      } else {
        unique_dst_merge_ptr = std::copy(
            dst_vec[i].begin(), dst_vec[i].end(), unique_dst_merge_ptr);
        src_merge_ptr = std::copy(
            outputs_vec[i].begin(), outputs_vec[i].end(), src_merge_ptr);
        dst_sample_counts_merge_ptr = std::copy(output_counts_vec[i].begin(),
                                                output_counts_vec[i].end(),
                                                dst_sample_counts_merge_ptr);
      }
    }

    // 5. Return eids results.
    if (return_eids) {
      std::vector<T> eids_merge(src_size);
      auto eids_merge_ptr = eids_merge.begin();
      for (size_t i = 0; i < num_layers; i++) {
        if (i == 0) {
          eids_merge_ptr = std::copy(outputs_eids_vec[i].begin(),
                                     outputs_eids_vec[i].end(),
                                     eids_merge.begin());
        } else {
          eids_merge_ptr = std::copy(outputs_eids_vec[i].begin(),
                                     outputs_eids_vec[i].end(),
                                     eids_merge_ptr);
        }
      }
      auto* out_eids = ctx.Output<phi::DenseTensor>("Out_Eids");
      out_eids->Resize({static_cast<int>(eids_merge.size())});
      T* p_out_eids = out_eids->mutable_data<T>(ctx.GetPlace());
      std::copy(eids_merge.begin(), eids_merge.end(), p_out_eids);
    }

    int64_t num_sample_edges = std::accumulate(
        dst_sample_counts_merge.begin(), dst_sample_counts_merge.end(), 0);
    PADDLE_ENFORCE_EQ(
        src_merge.size(),
        num_sample_edges,
        platform::errors::PreconditionNotMet(
            "Number of sample edges dismatch, the sample kernel has error."));

    // 6. Reindex edges.
    std::unordered_map<T, T> node_map;
    std::vector<T> unique_nodes;
    size_t reindex_id = 0;
    for (size_t i = 0; i < unique_dst_merge.size(); i++) {
      T node = unique_dst_merge[i];
      unique_nodes.emplace_back(node);
      node_map[node] = reindex_id++;
    }
    for (size_t i = 0; i < src_merge.size(); i++) {
      T node = src_merge[i];
      if (node_map.find(node) == node_map.end()) {
        unique_nodes.emplace_back(node);
        node_map[node] = reindex_id++;
      }
      src_merge[i] = node_map[node];
    }
    std::vector<T> dst_merge(src_merge.size());
    size_t cnt = 0;
    for (size_t i = 0; i < unique_dst_merge.size(); i++) {
      for (T j = 0; j < dst_sample_counts_merge[i]; j++) {
        T node = unique_dst_merge[i];
        dst_merge[cnt++] = node_map[node];
      }
    }

    // 7. Get Reindex_X for input nodes.
    auto* reindex_x = ctx.Output<phi::DenseTensor>("Reindex_X");
    T* p_reindex_x = reindex_x->mutable_data<T>(ctx.GetPlace());
    for (size_t i = 0; i < bs; i++) {
      p_reindex_x[i] = node_map[p_vertices[i]];
    }

    // 8. Get operator's outputs.
    auto* sample_index = ctx.Output<phi::DenseTensor>("Sample_Index");
    auto* out_src = ctx.Output<phi::DenseTensor>("Out_Src");
    auto* out_dst = ctx.Output<phi::DenseTensor>("Out_Dst");
    sample_index->Resize({static_cast<int>(unique_nodes.size())});
    out_src->Resize({static_cast<int>(src_merge.size()), 1});
    out_dst->Resize({static_cast<int>(src_merge.size()), 1});
    T* p_sample_index = sample_index->mutable_data<T>(ctx.GetPlace());
    T* p_out_src = out_src->mutable_data<T>(ctx.GetPlace());
    T* p_out_dst = out_dst->mutable_data<T>(ctx.GetPlace());
    std::copy(unique_nodes.begin(), unique_nodes.end(), p_sample_index);
    std::copy(src_merge.begin(), src_merge.end(), p_out_src);
    std::copy(dst_merge.begin(), dst_merge.end(), p_out_dst);
  }
};

}  // namespace operators
}  // namespace paddle
