// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/graph_sample_neighbors_kernel.h"

#include <vector>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <class bidiiter>
void SampleUniqueNeighbors(
    bidiiter begin,
    bidiiter end,
    int num_samples,
    std::mt19937& rng,                                        // NOLINT
    std::uniform_int_distribution<int>& dice_distribution) {  // NOLINT
  int left_num = std::distance(begin, end);
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
void SampleUniqueNeighborsWithEids(
    bidiiter src_begin,
    bidiiter src_end,
    bidiiter eid_begin,
    bidiiter eid_end,
    int num_samples,
    std::mt19937& rng,                                        // NOLINT
    std::uniform_int_distribution<int>& dice_distribution) {  // NOLINT
  int left_num = std::distance(src_begin, src_end);
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
void SampleNeighbors(const T* row,
                     const T* col_ptr,
                     const T* eids,
                     const T* input,
                     std::vector<T>* output,
                     std::vector<int>* output_count,
                     std::vector<T>* output_eids,
                     int sample_size,
                     int bs,
                     bool return_eids) {
  std::vector<std::vector<T>> out_src_vec;
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
  std::uniform_int_distribution<int> dice_distribution(
      0, std::numeric_limits<int>::max());

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  // Sample the neighbors in parallelism.
  for (int i = 0; i < bs; i++) {
    T node = input[i];
    T begin = col_ptr[node], end = col_ptr[node + 1];
    int cap = end - begin;
    if (sample_size < cap) {
      std::copy(row + begin, row + end, out_src_vec[i].begin());
      if (return_eids) {
        std::copy(eids + begin, eids + end, out_eids_vec[i].begin());
        SampleUniqueNeighborsWithEids(out_src_vec[i].begin(),
                                      out_src_vec[i].end(),
                                      out_eids_vec[i].begin(),
                                      out_eids_vec[i].end(),
                                      sample_size,
                                      rng,
                                      dice_distribution);
      } else {
        SampleUniqueNeighbors(out_src_vec[i].begin(),
                              out_src_vec[i].end(),
                              sample_size,
                              rng,
                              dice_distribution);
      }
      *(output_count->data() + i) = sample_size;
    } else {
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
void GraphSampleNeighborsKernel(
    const Context& dev_ctx,
    const DenseTensor& row,
    const DenseTensor& col_ptr,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& eids,
    const paddle::optional<DenseTensor>& perm_buffer,
    int sample_size,
    bool return_eids,
    bool flag_perm_buffer,
    DenseTensor* out,
    DenseTensor* out_count,
    DenseTensor* out_eids) {
  const T* row_data = row.data<T>();
  const T* col_ptr_data = col_ptr.data<T>();
  const T* x_data = x.data<T>();
  int bs = static_cast<int>(x.dims()[0]);

  std::vector<T> output;
  std::vector<int> output_count;

  if (return_eids) {
    const T* eids_data = eids.get_ptr()->data<T>();
    std::vector<T> output_eids;
    SampleNeighbors<T>(row_data,
                       col_ptr_data,
                       eids_data,
                       x_data,
                       &output,
                       &output_count,
                       &output_eids,
                       sample_size,
                       bs,
                       return_eids);
    out_eids->Resize({static_cast<int>(output_eids.size())});
    T* out_eids_data = dev_ctx.template Alloc<T>(out_eids);
    std::copy(output_eids.begin(), output_eids.end(), out_eids_data);
  } else {
    SampleNeighbors<T>(row_data,
                       col_ptr_data,
                       nullptr,
                       x_data,
                       &output,
                       &output_count,
                       nullptr,
                       sample_size,
                       bs,
                       return_eids);
  }
  out->Resize({static_cast<int>(output.size())});
  T* out_data = dev_ctx.template Alloc<T>(out);
  std::copy(output.begin(), output.end(), out_data);
  out_count->Resize({bs});
  int* out_count_data = dev_ctx.template Alloc<int>(out_count);
  std::copy(output_count.begin(), output_count.end(), out_count_data);
}

}  // namespace phi

PD_REGISTER_KERNEL(graph_sample_neighbors,
                   CPU,
                   ALL_LAYOUT,
                   phi::GraphSampleNeighborsKernel,
                   int,
                   int64_t) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT32);
}
