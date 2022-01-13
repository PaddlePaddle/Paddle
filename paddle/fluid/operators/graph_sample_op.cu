/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <ostream>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/graph_sample_op.h"
#include "paddle/fluid/operators/graph_sample_reindex.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/place.h"

constexpr int WARP_SIZE = 32;

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
struct MaxByFunctor {
  T cap;
  HOSTDEVICE explicit inline MaxByFunctor(T cap) { this->cap = cap; }
  HOSTDEVICE inline T operator()(T x) const {
    if (x > cap) {
      return cap;
    }
    return x;
  }
};

template <typename T>
struct DegreeFunctor {
  const T* dst_count;
  HOSTDEVICE explicit inline DegreeFunctor(const T* x) { this->dst_count = x; }
  HOSTDEVICE inline T operator()(T i) const {
    return dst_count[i + 1] - dst_count[i];
  }
};

template <typename T, int BLOCK_WARPS, int TILE_SIZE>
__global__ void GraphSampleCUDAKernel(const uint64_t rand_seed, int k,
                                      const int64_t num_rows, const T* in_rows,
                                      const T* src, const T* dst_count,
                                      T* outputs, T* output_ptr,
                                      T* output_idxs) {
  assert(blockDim.x == WARP_SIZE);
  assert(blockDim.y == BLOCK_WARPS);

  int64_t out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
  const int64_t last_row =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandState rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x,
              threadIdx.y * WARP_SIZE + threadIdx.x, 0, &rng);

  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t in_row_start = dst_count[row];
    const int64_t deg = dst_count[row + 1] - in_row_start;
    const int64_t out_row_start = output_ptr[out_row];

    if (deg <= k) {
      for (int idx = threadIdx.x; idx < deg; idx += WARP_SIZE) {
        const T in_idx = in_row_start + idx;
        outputs[out_row_start + idx] = src[in_idx];
      }
    } else {
      for (int idx = threadIdx.x; idx < k; idx += WARP_SIZE) {
        output_idxs[out_row_start + idx] = idx;
      }
      __syncwarp();

      for (int idx = k + threadIdx.x; idx < deg; idx += WARP_SIZE) {
        const int num = curand(&rng) % (idx + 1);
        if (num < k) {
          paddle::platform::CudaAtomicMax(output_idxs + out_row_start + num,
                                          idx);
        }
      }
      __syncwarp();

      for (int idx = threadIdx.x; idx < k; idx += WARP_SIZE) {
        const T perm_idx = output_idxs[out_row_start + idx] + in_row_start;
        outputs[out_row_start + idx] = src[perm_idx];
      }
    }

    out_row += BLOCK_WARPS;
  }
}

template <typename T, int BLOCK_WARPS, int TILE_SIZE>
__global__ void GetDstEdgeCUDAKernel(const int64_t num_rows, const T* in_rows,
                                     const T* dst_sample_counts,
                                     const T* dst_ptr, T* outputs) {
  assert(blockDim.x == WARP_SIZE);
  assert(blockDim.y == BLOCK_WARPS);

  int64_t out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
  const int64_t last_row =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t dst_sample_size = dst_sample_counts[out_row];
    const int64_t out_row_start = dst_ptr[out_row];
    for (int idx = threadIdx.x; idx < dst_sample_size; idx += WARP_SIZE) {
      outputs[out_row_start + idx] = row;
    }
    __syncwarp();

    out_row += BLOCK_WARPS;
  }
}

template <typename T>
void sample_neighbors(const framework::ExecutionContext& ctx, const T* src,
                      const T* dst_count, thrust::device_vector<T>* inputs,
                      thrust::device_vector<T>* outputs,
                      thrust::device_vector<T>* output_counts, T k) {
  const size_t bs = (*inputs).size();
  (*output_counts).resize(bs);

  // 1. Get degree.
  thrust::transform((*inputs).begin(), (*inputs).end(),
                    (*output_counts).begin(), DegreeFunctor<T>(dst_count));

  // 2. Apply sample size k.
  if (k >= 0) {
    thrust::transform((*output_counts).begin(), (*output_counts).end(),
                      (*output_counts).begin(), MaxByFunctor<T>(k));
  }

  // 3. Get the number of total sample neighbors and some necessary datas.
  T tos = thrust::reduce((*output_counts).begin(), (*output_counts).end());
  (*outputs).resize(tos);
  thrust::device_vector<T> output_ptr;
  thrust::device_vector<T> output_idxs;
  output_ptr.resize(bs);
  output_idxs.resize(tos);
  thrust::exclusive_scan((*output_counts).begin(), (*output_counts).end(),
                         output_ptr.begin(), 0);

  // 4. Sample Kernel.
  constexpr int BLOCK_WARPS = 128 / WARP_SIZE;
  constexpr int TILE_SIZE = BLOCK_WARPS * 16;
  const dim3 block(WARP_SIZE, BLOCK_WARPS);
  const dim3 grid((bs + TILE_SIZE - 1) / TILE_SIZE);
  GraphSampleCUDAKernel<T, BLOCK_WARPS, TILE_SIZE><<<
      grid, block, 0,
      reinterpret_cast<const platform::CUDADeviceContext&>(ctx.device_context())
          .stream()>>>(0, k, bs, thrust::raw_pointer_cast((*inputs).data()),
                       src, dst_count,
                       thrust::raw_pointer_cast((*outputs).data()),
                       thrust::raw_pointer_cast(output_ptr.data()),
                       thrust::raw_pointer_cast(output_idxs.data()));

  // 5. Get inputs = outputs - inputs: 对于采样一层的时候，下面的地方不需要跑，所以后面可以优化，判断一下.
  thrust::device_vector<T> unique_outputs((*outputs).size());
  thrust::sort((*inputs).begin(), (*inputs).end());
  thrust::device_vector<T> outputs_sort((*outputs).size());
  thrust::copy((*outputs).begin(), (*outputs).end(), outputs_sort.begin());
  thrust::sort(outputs_sort.begin(), outputs_sort.end());
  auto unique_outputs_end = thrust::set_difference(
      outputs_sort.begin(), outputs_sort.end(), (*inputs).begin(),
      (*inputs).end(), unique_outputs.begin());
  unique_outputs_end =
      thrust::unique(unique_outputs.begin(), unique_outputs_end);
  (*inputs).resize(
      thrust::distance(unique_outputs.begin(), unique_outputs_end));
  thrust::copy(unique_outputs.begin(), unique_outputs_end, (*inputs).begin());
}

template <typename T>
DeviceHashTable<T>
FillTableWithDuplicates(const framework::ExecutionContext& ctx, T* input, 
                        int64_t num_input, thrust::device_vector<T> &unique_items) {
  VLOG(0) << "Enter FillTableWithDuplicates function";

  VLOG(0) << "1. build DeviceHashTable";
  DeviceHashTable<T> hash_table = DeviceHashTable<T>(num_input, 1);
  
  // 暂且简单写
  int block = 1024;
  int grid = (num_input + block - 1) / block;
  VLOG(0) << "2. insert data into DeviceHashTable";
  build_hashtable_duplicates<T><<<
      grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                            ctx.device_context())
                            .stream()>>>(input, num_input, &hash_table);
  thrust::device_vector<int> item_count(num_input + 1, 0);
 
  VLOG(0) << "3. Get item index count"; 
  get_item_index_count<T><<<
      grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                            ctx.device_context())
                            .stream()>>>(input, thrust::raw_pointer_cast(item_count.data()),
                                         num_input, &hash_table);
 
  //using it = thrust::counting_iterator<T>;
  //using Mapping = typename DeviceHashTable<T>::Mapping;
  //VLOG(0) << "3.1 Begin!";
  //thrust::for_each(it(0), it(num_input),
  //                 [count = thrust::raw_pointer_cast(item_count.data()),
  //                  table = hash_table,
  //                  in = input] __device__(T i) mutable {
  //                      //in[i];
  //                     //Mapping mapping = *(table.Search(in[i]));
  //                     //if (mapping.index == i) { count[i] = 1; }
  //                });
  //VLOG(0) << "3.2 End!";
  //thrust::exclusive_scan(item_count.begin(), item_count.end(),
  //                       item_count.begin());
  //size_t tos = item_count[num_input];
  //unique_items.resize(tos);

  //// 填充 unique_items
  //VLOG(0) << "4. Get unique items";
  //fill_unique_items<T><<<
  //    grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
  //                          ctx.device_context())
  //                          .stream()>>>(input, num_input, 
  //                                       thrust::raw_pointer_cast(unique_items.data()),
  //                                       thrust::raw_pointer_cast(item_count.data()), 
  //                                       hash_table);

  return hash_table;
}

template <typename T>
void reindex_func(const framework::ExecutionContext& ctx,
                  thrust::device_vector<T> &inputs, 
                  thrust::device_vector<T> &outputs,
                  thrust::device_vector<T> &subset) {
  VLOG(0) << "Enter reindex function";
  subset.resize(inputs.size() + outputs.size());
  thrust::copy(inputs.begin(), inputs.end(), subset.begin());
  thrust::copy(outputs.begin(), outputs.end(), subset.begin() + inputs.size());
  thrust::device_vector<T> unique_items;
  unique_items.clear();

  VLOG(0) << "Begin to fill hash table";
  VLOG(0) << "subset.size(): " << subset.size();
  VLOG(0) << "Print subset";
  thrust::copy(subset.begin(), subset.end(), std::ostream_iterator<T>(std::cout, " "));
  std::cout << std::endl;
  DeviceHashTable<T> table = FillTableWithDuplicates(ctx, thrust::raw_pointer_cast(subset.data()),
                                  subset.size(), unique_items);
  subset.resize(unique_items.size());
  thrust::copy(unique_items.begin(), unique_items.end(), subset.begin());
  
  // Fill outputs with reindex result.
  VLOG(0) << "Fill src output with reindex result";
  int block = 1024;
  int grid = (outputs.size() + block - 1) / block;
  reindex_src_output<T><<<
      grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                            ctx.device_context())
                            .stream()>>>(thrust::raw_pointer_cast(outputs.data()),
                     outputs.size(), table);
}

template <typename DeviceContext, typename T>
class GraphSampleOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // 1. 获取inputs数据
    auto* src = ctx.Input<Tensor>("Src");
    auto* dst_count = ctx.Input<Tensor>("Dst_Count");
    auto* vertices = ctx.Input<Tensor>("X");
    std::vector<int> sample_sizes = ctx.Attr<std::vector<int>>("sample_sizes");

    const T* src_data = src->data<T>();
    const T* dst_count_data = dst_count->data<T>();
    const T* p_vertices = vertices->data<T>();

    // 2. 获取采样节点信息
    const size_t bs = vertices->dims()[0];

    // 3.
    // 处理基本变量，用于存储采样的邻居和个数统计，同时保存采样过程中的中间结果.
    thrust::device_vector<T> inputs;
    thrust::device_vector<T> outputs;
    thrust::device_vector<T> output_counts;
    inputs.resize(bs);
    thrust::copy(p_vertices, p_vertices + bs, inputs.begin());
    std::vector<thrust::device_vector<T>> dst_vec;
    dst_vec.push_back(inputs);
    std::vector<thrust::device_vector<T>> outputs_vec;
    std::vector<thrust::device_vector<T>> output_counts_vec;

    const size_t num_layers = sample_sizes.size();
    for (int i = 0; i < num_layers; i++) {
      if (inputs.size() == 0) {
        break;
      }
      if (i > 0) {
        dst_vec.push_back(inputs);
      }
      sample_neighbors<T>(ctx, src_data, dst_count_data, &inputs, &outputs,
                          &output_counts, sample_sizes[i]);
      outputs_vec.push_back(outputs);
      output_counts_vec.push_back(output_counts);
    }

    // 4. 对中间存储结果进行连接，得到采样后的Src 边(src_merge), unique 后的 Dst
    // 边(unique_dst_merge)，以及dst的sample count(dst_sample_counts_merge)
    thrust::device_vector<T> unique_dst_merge;         // unique dst
    thrust::device_vector<T> src_merge;                // src
    thrust::device_vector<T> dst_sample_counts_merge;  // dst degree
    int64_t unique_dst_size = 0, src_size = 0;
    for (int i = 0; i < num_layers; i++) {
      unique_dst_size += dst_vec[i].size();
      src_size += outputs_vec[i].size();
    }
    unique_dst_merge.resize(unique_dst_size);
    src_merge.resize(src_size);
    dst_sample_counts_merge.resize(unique_dst_size);
    auto unique_dst_merge_ptr = unique_dst_merge.begin();
    auto src_merge_ptr = src_merge.begin();
    auto dst_sample_counts_merge_ptr = dst_sample_counts_merge.begin();
    for (int i = 0; i < num_layers; i++) {
      if (i == 0) {
        unique_dst_merge_ptr = thrust::copy(
            dst_vec[i].begin(), dst_vec[i].end(), unique_dst_merge.begin());
        src_merge_ptr = thrust::copy(outputs_vec[i].begin(),
                                     outputs_vec[i].end(), src_merge.begin());
        dst_sample_counts_merge_ptr = thrust::copy(
            output_counts_vec[i].begin(), output_counts_vec[i].end(),
            dst_sample_counts_merge.begin());
      } else {
        unique_dst_merge_ptr = thrust::copy(
            dst_vec[i].begin(), dst_vec[i].end(), unique_dst_merge_ptr);
        src_merge_ptr = thrust::copy(outputs_vec[i].begin(),
                                     outputs_vec[i].end(), src_merge_ptr);
        dst_sample_counts_merge_ptr = thrust::copy(output_counts_vec[i].begin(),
                                                   output_counts_vec[i].end(),
                                                   dst_sample_counts_merge_ptr);
      }
    }

    int64_t num_sample_edges = thrust::reduce(dst_sample_counts_merge.begin(),
                                              dst_sample_counts_merge.end());

    PADDLE_ENFORCE_EQ(
        src_merge.size(), num_sample_edges,
        platform::errors::External("Number of sample edges dismatch."));

    // 5. 根据unique_dst_merge, src_merge 生成 hashtable，unique后的items(subset)，并且对 src_merge 进行reindex.
    VLOG(0) << "Begin to Reindex";
    thrust::device_vector<T> subset;
    reindex_func<T>(ctx, unique_dst_merge, src_merge, subset);

    auto* sample_index = ctx.Output<Tensor>("Sample_index");
    sample_index->Resize({static_cast<int>(subset.size())});
    T* p_sample_index = sample_index->mutable_data<T>(ctx.GetPlace());
    const size_t& sample_bytes = subset.size() * sizeof(T);
    cudaMemset(p_sample_index, 0, sample_bytes);
    thrust::copy(subset.begin(), subset.end(), p_sample_index);  // Done!

    // 6. 还原Dst边(根据dst_counts_merge进行copy 和 reindex.)
    thrust::device_vector<T> dst_merge(src_size);
    thrust::device_vector<T> unique_dst_merge_reindex(unique_dst_size);
    thrust::sequence(unique_dst_merge_reindex.begin(), unique_dst_merge_reindex.end());
    thrust::device_vector<T> dst_ptr(unique_dst_size);
    thrust::exclusive_scan(dst_sample_counts_merge.begin(),
                           dst_sample_counts_merge.end(), dst_ptr.begin());
    constexpr int BLOCK_WARPS = 128 / WARP_SIZE;  // 每个block的warp数量
    constexpr int TILE_SIZE = BLOCK_WARPS * 16;  // 设置每个block会处理的节点数?
    const dim3 block(WARP_SIZE, BLOCK_WARPS);
    const dim3 grid((unique_dst_size + TILE_SIZE - 1) / TILE_SIZE);

    GetDstEdgeCUDAKernel<T, BLOCK_WARPS,
                         TILE_SIZE>
        <<<grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                               ctx.device_context())
                               .stream()>>>(
            unique_dst_size, thrust::raw_pointer_cast(unique_dst_merge_reindex.data()),
            thrust::raw_pointer_cast(dst_sample_counts_merge.data()),
            thrust::raw_pointer_cast(dst_ptr.data()),
            thrust::raw_pointer_cast(dst_merge.data()));
 
    // 7. 赋值结果.
    auto* out_src = ctx.Output<Tensor>("Out_Src");
    auto* out_dst = ctx.Output<Tensor>("Out_Dst");
    out_src->Resize({static_cast<int>(src_merge.size())});
    out_dst->Resize({static_cast<int>(src_merge.size())});
    T* p_out_src = out_src->mutable_data<T>(ctx.GetPlace());
    T* p_out_dst = out_dst->mutable_data<T>(ctx.GetPlace());

    const size_t& memset_bytes = src_merge.size() * sizeof(T);
    cudaMemset(p_out_src, 0, memset_bytes);  // hipMemset
    cudaMemset(p_out_dst, 0, memset_bytes);

    thrust::copy(src_merge.begin(), src_merge.end(), p_out_src);
    thrust::copy(dst_merge.begin(), dst_merge.end(), p_out_dst);
  }
};

}  // namespace operators
}  // namespace paddle

using CUDA = paddle::platform::CUDADeviceContext;
namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(graph_sample, ops::GraphSampleOpCUDAKernel<CUDA, int>,
                        ops::GraphSampleOpCUDAKernel<CUDA, int64_t>);
