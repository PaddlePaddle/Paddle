// NOLINT
// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved. */

/*This code is copied from NVIDIA apex:
 *     https://github.com/NVIDIA/apex
 *     with minor changes. */

#include "paddle/phi/kernels/moe_ops_partial_nosoftmaxtopk_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/moe_fuse_op.h"
#include "paddle/phi/kernels/moe_kernel_impl.h"
#include "paddle/phi/kernels/slice_kernel.h"

namespace phi {

#define CUDACHECK(cmd)                          \
  do {                                          \
    cudaError_t e = cmd;                        \
    if (e != cudaSuccess) {                     \
      printf("Failed: Cuda error %s:%d '%s'\n", \
             __FILE__,                          \
             __LINE__,                          \
             cudaGetErrorString(e));            \
      exit(EXIT_FAILURE);                       \
    }                                           \
  } while (0)

// already defined need to revise!
// static inline size_t AlignTo16(const size_t &input){
//   static constexpr int ALIGNMENT = 16;
//   return ALIGNMENT * ((input + ALIGNMENT - 1) / ALIGNMENT);
// }
namespace {
// --------      getWorkspaceSize      -------- //
template <typename KeyT>
size_t getWorkspaceSize(const int num_rows,
                        const int hidden_size,
                        const int inter_size,
                        const int num_experts,
                        const int capacity,
                        const int k,
                        //  const int max_seq_len,
                        bool use_pad,
                        phi::CubKeyValueSorter &sorter) {  // NOLINT
  // const int buf_size = AlignTo16(k * num_rows * hidden_size);
  const int interbuf_size = AlignTo16(k * num_rows * inter_size);
  const int padded_experts = AlignTo16(num_experts);
  const int num_moe_inputs = AlignTo16(k * num_rows);
  const int num_dispatched_size = AlignTo16(num_experts * capacity);
  int num_softmax_outs = 0;

  // softmax output, permuted_rows and permuted_experts have moved to outside of
  // moe kernel, allocate them in Encoder or Decoder before invoking FfnLayer
  // forward.
  size_t total_ws_bytes =
      4 * num_moe_inputs *
      sizeof(int);  // source_rows_, permuted_rows_, permuted_experts_
  total_ws_bytes += 2 * num_dispatched_size * sizeof(int);
  total_ws_bytes +=
      padded_experts *
      sizeof(int64_t);  // Hold total_rows_before_expert_  // expert_cnt
  // total_ws_bytes += buf_size * sizeof(KeyT);                // permuted_data
  total_ws_bytes += num_softmax_outs * sizeof(KeyT);
  const int bytes_for_fc1_result = interbuf_size * sizeof(KeyT);
  const int sorter_ws_size_bytes =
      std::max(AlignTo16(sorter.getWorkspaceSize(k * num_rows)),
               AlignTo16(sorter.getWorkspaceSize(capacity)));
  // sorter.update_num_experts(num_experts+1); // +1 for filter out of capacity
  // // 用所有 bit 做排序,会降低些许性能,但是防止越界
  int bytes_for_intermediate_and_sorting = bytes_for_fc1_result;
  if (sorter_ws_size_bytes > bytes_for_fc1_result) {
    int remaining_bytes =
        AlignTo16(sorter_ws_size_bytes - bytes_for_fc1_result);
    bytes_for_intermediate_and_sorting += remaining_bytes;
  }
  // std::cout<<"num_softmax_outs --"<< num_softmax_outs << std::endl;
  total_ws_bytes +=
      bytes_for_intermediate_and_sorting;  // intermediate (fc1) output + cub
                                           // sorting workspace
  // std::cout<<"buf_size --"<< buf_size<<"   "<<interbuf_size<< "
  // "<<padded_experts<< "    "<<num_moe_inputs<<  "  "<<total_ws_bytes<< " "<<
  // bytes_for_fc1_result<< "   "<<sorter_ws_size_bytes  << "  "<<std::endl;
  return total_ws_bytes;
}
}  // namespace

template <typename T, typename Context>
void apply_moe_dispatch_fwd(
    const Context &dev_ctx,
    const DenseTensor &x,
    int64_t num_rows,
    int64_t num_experts,
    int64_t hidden_size,
    int64_t capacity,
    int64_t k,
    int64_t expert_start_index,
    int64_t expert_end_index,
    bool reverse_token_drop,
    thrust::host_vector<int64_t> &expert_offset_host,  // NOLINT
    DenseTensor *y,
    float *combine_weights,
    int *scatter_index,
    int *scatter_index_rev,
    int64_t *expert_offset_global,
    int64_t *expert_nums_local,
    int *expert_id,
    bool use_pad,
    cudaStream_t stream) {
  phi::CubKeyValueSorter sorter(stream);
  // paddle::Tensor expanded_source_row_to_expanded_dest_row_tensor =
  //     paddle::empty({num_rows, k}, paddle::DataType::INT32, place);
  // int* expanded_source_row_to_expanded_dest_row =
  //     expanded_source_row_to_expanded_dest_row_tensor.data<int>();

  // paddle::Tensor expert_scales_tensor_float = paddle::empty({num_rows, k},
  // paddle::DataType::FLOAT32, place); float* expert_scales_float =
  // expert_scales_tensor_float.data<float>();

  // paddle::Tensor expert_for_source_row_tensor = paddle::empty({num_rows, k},
  // paddle::DataType::INT32, place); int* expert_for_source_row =
  // expert_for_source_row_tensor.data<int>(); paddle::Tensor active_cnt_tensor
  // = paddle::empty({1}, paddle::DataType::INT32, place);

  int64_t bytes = getWorkspaceSize<T>(num_rows,
                                      hidden_size,  // hidden-size=0
                                      0,            // inter-size=0
                                      num_experts,
                                      capacity,
                                      k,
                                      use_pad,
                                      sorter);

  DenseTensor ws_ptr_tensor = phi::Empty<int8_t, Context>(dev_ctx, {bytes});
  int8_t *ws_ptr = ws_ptr_tensor.data<int8_t>();

  phi::memory_utils::ThrustAllocator<cudaStream_t> allocator(dev_ctx.GetPlace(),
                                                             dev_ctx.stream());

  // Pointers
  int *source_rows_;
  int *permuted_rows_;
  int *permuted_experts_;
  int *expert_id_;
  int *source_rows_for_seqsort_;
  int *source_rows_for_seqsort_out_;
  int *source_pos_for_seqsort_;
  int *source_pos_for_seqsort_out_;
  int64_t *expert_offset_;  // local-expert-offset

  char *sorter_ws_;
  // T* permuted_data_;
  float *softmax_out_;
  // int64_t* total_rows_before_expert_;
  T *fc1_result_;

  const int sorter_ws_size_bytes =
      AlignTo16(sorter.getWorkspaceSize(k * num_rows));
  const int sorter_ws_size_bytes_seqsort =
      AlignTo16(sorter.getWorkspaceSize(capacity));

  const int buf_size = AlignTo16(k * num_rows * hidden_size);
  // const int interbuf_size  = AlignTo16(k * num_rows * 0);
  const int padded_experts = AlignTo16(num_experts);
  const int num_moe_inputs = AlignTo16(k * num_rows);
  const int num_dispatched_size = AlignTo16(num_experts * capacity);

  // 4:ints [k*row]
  source_rows_ = reinterpret_cast<int *>(ws_ptr);
  permuted_rows_ = source_rows_ + num_moe_inputs;
  permuted_experts_ = permuted_rows_ + num_moe_inputs;
  expert_id_ = permuted_experts_ + num_moe_inputs;
  // 4:ints: [E*C]
  source_rows_for_seqsort_ = expert_id_ + num_moe_inputs;
  source_rows_for_seqsort_out_ = source_rows_for_seqsort_ + num_dispatched_size;
  // 1:ints: [E]
  expert_offset_ = reinterpret_cast<int64_t *>(source_rows_for_seqsort_out_ +
                                               num_dispatched_size);
  // permuted_data_ = reinterpret_cast<T *>(expert_offset_ + padded_experts);
  // total_rows_before_expert_ = reinterpret_cast<int64_t*>(permuted_experts_ +
  // buf_size);

  // only use one number
  // num_active   = reinterpret_cast<int64_t*>(permuted_experts_ +
  // num_moe_inputs);
  fc1_result_ = reinterpret_cast<T *>(expert_offset_ + padded_experts);
  // fc1_result_ = reinterpret_cast<T *>(permuted_data_ + buf_size);

#ifdef DEBUG_MOE_OP
  // print_to_screen1(gate_logits, 8, 16, std::string("gate_logits
  // before_topk")); print_to_screen1(finished, 2, 16, std::string("finished
  // before_topk"));
#endif

  thrust::transform(thrust::cuda::par.on(stream),
                    thrust::device_pointer_cast(source_rows_),
                    thrust::device_pointer_cast(source_rows_) + num_rows * k,
                    thrust::counting_iterator<int>(0),
                    thrust::device_pointer_cast(source_rows_),
                    [num_rows, k] __device__(int i, int cnt) {
                      int k_idx = cnt % k;
                      int block_row = cnt / k;
                      return k_idx * num_rows + block_row;
                    });

#ifdef DEBUG_MOE_OP
  // phi::CastKernel<float>(ctx, expert_scales_tensor_float,
  // expert_scales_tensor.dtype(), &expert_scales_tensor);
  print_to_screen1(
      combine_weights, 8, 16, std::string("expert_scales_float after topk"));
  print_to_screen1<int>(
      expert_id, 8, 16, std::string("expert-id before permute"));
  print_to_screen1<int>(
      source_rows_, 8, 16, std::string("desc->src idx before permute"));
#endif

  // compute global expert offset, **not** consider capacity
  // 必须在 modify_and_mask_expert_id_launcher 之前算出**全局 expert-offset**

  compute_global_expert_offset(expert_id,
                               expert_id_,  // buffer
                               expert_offset_global,
                               num_rows * k,
                               num_experts,
                               capacity,
                               stream,
                               allocator);

  // modify expert-id according to k
  modify_and_mask_expert_id_launcher(expert_id,
                                     expert_id_,
                                     k,
                                     num_rows,
                                     static_cast<int>(num_experts),
                                     static_cast<int>(expert_start_index),
                                     static_cast<int>(expert_end_index),
                                     stream);

#ifdef DEBUG_MOE_OP
  print_to_screen1<int>(
      expert_id_, 8, 16, std::string("expert-id after modified 22"));
#endif
  sorter.run(
      fc1_result_,
      sorter_ws_size_bytes,
      expert_id_,         // key in
      permuted_experts_,  // key out // [num_row, k]: expert-id
      source_rows_,       // value in
      permuted_rows_,  // value out //[num_row, k]: id在原 activation 中的位置
      k * num_rows,  // num_rows
      false,
      stream);

  unmodify_expert_id_launcher(
      permuted_experts_, permuted_experts_, k, num_rows, num_experts, stream);

#ifdef DEBUG_MOE_OP
  print_to_screen1<int>(
      permuted_experts_, 8, 16, std::string("expert-id after permute"));
  print_to_screen1<int>(
      permuted_rows_, 8, 16, std::string("dest->src idx after permute"));
#endif

  compute_local_expert_offset(permuted_experts_,
                              expert_offset_,
                              expert_nums_local,
                              num_rows * k,
                              num_experts,
                              capacity,
                              stream,
                              allocator);

  CUDACHECK(cudaMemcpyAsync(expert_offset_host.data(),
                            expert_offset_,
                            num_experts * sizeof(int64_t),
                            cudaMemcpyDeviceToHost,
                            stream));
  CUDACHECK(cudaStreamSynchronize(stream));

#ifdef DEBUG_MOE_OP
  std::cerr << "[DEBUG] num_active v2: " << expert_offset_host.back()
            << std::endl;
  print_to_screen1(
      expert_offset_global, 8, 16, std::string("expert_offset global"));
  print_to_screen1(expert_offset_, 8, 16, std::string("expert_offset local"));
  print_to_screen1<int>(permuted_experts_,
                        8,
                        16,
                        std::string("<reprint>expert-id after permute"));
  // print_to_screen1(permuted_experts_, 4096, 8192,
  // std::string("<reprint>expert-id after permute"));
#endif

  // calc expert-size
  // 不 use-pad 的情况下，在此处标记截断位置。之后需要再 sort 一遍把截断 id
  // 放到句尾
  if (!use_pad) {  // 2sort
    cal_expert_size_and_filter_launcher(permuted_experts_,
                                        expert_offset_,
                                        expert_offset_host.back(),
                                        num_experts,
                                        capacity,
                                        expert_start_index,
                                        expert_end_index,
                                        reverse_token_drop,
                                        stream);
    // 2sort
    sorter.run(
        fc1_result_,
        sorter_ws_size_bytes,
        permuted_experts_,  // key in
        permuted_experts_,  // key out // [num_row, k]: expert-id
        permuted_rows_,     // value in
        permuted_rows_,  // value out //[num_row, k]: id在原 activation 中的位置
        k * num_rows,  // num_rows
        false,
        stream);

    compute_local_expert_offset(permuted_experts_,
                                expert_offset_,
                                expert_nums_local,
                                num_rows * k,
                                num_experts,
                                capacity,
                                stream,
                                allocator);

    CUDACHECK(cudaMemcpyAsync(expert_offset_host.data(),
                              expert_offset_,
                              num_experts * sizeof(int64_t),
                              cudaMemcpyDeviceToHost,
                              stream));
    CUDACHECK(cudaStreamSynchronize(stream));

#ifdef DEBUG_MOE_OP
    std::cerr << "[DEBUG](after 2sort) num_active v2: "
              << expert_offset_host.back() << std::endl;
    print_to_screen1<int>(
        expert_id_, 8, 16, std::string("<before 2sort> permuted_experts"));
    print_to_screen1<int>(permuted_experts_,
                          8,
                          16,
                          std::string("<after 2sort> permuted_experts"));
    print_to_screen1(
        permuted_rows_, 8, 16, std::string("<after 2sort> dest->src idx"));
#endif
  }

  thrust::fill(
      thrust::cuda::par.on(stream),
      thrust::device_ptr<int>(scatter_index_rev),
      thrust::device_ptr<int>(scatter_index_rev) + num_experts * capacity,
      num_rows);
  build_seqsort_kv_pairs_kernel_launcher(
      scatter_index_rev,         // padded_to_unpermuted_input
      source_rows_for_seqsort_,  // seqsort-value
      permuted_rows_,
      // scatter_index, // 对截断位置置0
      permuted_experts_,
      expert_offset_,
      combine_weights,  // 对截断位置置0
      static_cast<int>(num_rows),
      static_cast<int>(k),
      expert_offset_host.back(),  // num_active
      capacity,
      expert_start_index,  // expert start index
      use_pad,
      stream);

#ifdef DEBUG_MOE_OP

  // print_to_screen1<int>(scatter_index, 8, 16, std::string("scatter_index
  // after build_seqsort_kv_pairs_kernel_launcher"));
  print_to_screen1<int>(source_rows_for_seqsort_,
                        8,
                        16,
                        std::string("source_rows_for_seqsort_ after "
                                    "build_seqsort_kv_pairs_kernel_launcher"));
  print_to_screen1<int>(
      scatter_index_rev,
      8,
      16,
      std::string(
          "scatter_index_rev after build_seqsort_kv_pairs_kernel_launcher"));
#endif
  if (use_pad) {
    for (auto iexpert = 0; iexpert != expert_end_index - expert_start_index;
         ++iexpert) {
      sorter.run(fc1_result_,
                 sorter_ws_size_bytes_seqsort,
                 scatter_index_rev + (iexpert * capacity),         // key in
                 scatter_index_rev + (iexpert * capacity),         // key out
                 source_rows_for_seqsort_ + (iexpert * capacity),  // value in
                 source_rows_for_seqsort_ +
                     (iexpert * capacity),  // value out //[num_row, k]: id在原
                                            // activation 中的位置
                 capacity,  // num_rows
                 false,
                 stream);
    }
  } else {
    auto sort_iter = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::device_pointer_cast(permuted_experts_),  // key1
        thrust::device_pointer_cast(scatter_index_rev),  // key2
        thrust::device_pointer_cast(source_rows_for_seqsort_)));
    thrust::stable_sort(thrust::cuda::par.on(stream),
                        sort_iter,
                        sort_iter + expert_offset_host.back(),
                        [] __device__(auto lhs, auto rhs) {
                          if (thrust::get<0>(lhs) < thrust::get<0>(rhs))
                            return true;
                          else if (thrust::get<0>(lhs) > thrust::get<0>(rhs))
                            return false;
                          else
                            return thrust::get<1>(lhs) < thrust::get<1>(rhs);
                        });
  }
  if (use_pad) {
    int64_t num_experts_diff = expert_end_index - expert_start_index;
    y->Resize({num_experts_diff * capacity, x.dims()[1]});
    dev_ctx.template Alloc<T>(y);
  } else {
    y->Resize({expert_offset_host.back(), x.dims()[1]});
    dev_ctx.template Alloc<T>(y);
  }
  phi::Full<T, Context>(
      dev_ctx, phi::IntArray(common::vectorize(y->dims())), 0, y);
  copy_unpermuted_to_permuted_kernelLauncher(
      x.data<T>(),
      y->data<T>(),              // out
      scatter_index_rev,         // padded_out_to_unpermuted_input
      source_rows_for_seqsort_,  // padded_out_to_expanded_input
      scatter_index,             // out
      use_pad ? (expert_end_index - expert_start_index) * capacity
              : expert_offset_host.back(),  // num_active
      num_rows,
      k,
      hidden_size,
      stream);
  // cudaDeviceSynchronize(); //debug
  // turn expert_offset_ptr into experts_num
  return;
}

template <typename T, typename Context>
void moe_dispatch_fwd(
    const Context &dev_ctx,
    const DenseTensor &x,
    int64_t num_rows,
    int64_t num_experts,
    int64_t hidden_size,
    int64_t capacity,
    int64_t k,
    int64_t expert_start_index,
    int64_t expert_end_index,
    bool reverse_token_drop,
    thrust::host_vector<int64_t> &expert_offset_host,  // NOLINT
    DenseTensor *y,
    const DenseTensor &combine_weights,
    const DenseTensor &scatter_index,
    const DenseTensor &scatter_index_rev,
    const DenseTensor &expert_offset,
    const DenseTensor &expert_nums_local,
    const DenseTensor &expert_id,
    bool use_pad) {
  apply_moe_dispatch_fwd<T, Context>(
      dev_ctx,
      x,
      num_rows,
      num_experts,
      hidden_size,
      capacity,
      k,
      expert_start_index,
      expert_end_index,
      reverse_token_drop,
      expert_offset_host,
      y,
      const_cast<float *>(combine_weights.data<float>()),
      const_cast<int *>(scatter_index.data<int>()),
      const_cast<int *>(scatter_index_rev.data<int>()),
      const_cast<int64_t *>(expert_offset.data<int64_t>()),
      const_cast<int64_t *>(expert_nums_local.data<int64_t>()),
      const_cast<int *>(expert_id.data<int>()),
      use_pad,
      dev_ctx.stream());
}

template <typename T, typename Context>
void MoeGateDispatchPartialNoSoftMaxTopkKernel(
    const Context &dev_ctx,
    const DenseTensor &x,
    const DenseTensor &combine_weights,
    const DenseTensor &expert_id,
    int64_t k,
    int64_t capacity,
    int64_t num_experts,
    bool use_pad,
    int64_t expert_start_index,
    int64_t expert_end_index,
    bool reverse_token_drop,
    DenseTensor *y,
    DenseTensor *combine_weights_out,
    DenseTensor *scatter_index,
    DenseTensor *scatter_index_rev,
    DenseTensor *expert_offset,
    DenseTensor *expert_nums_local) {
  dev_ctx.template Alloc<int32_t>(scatter_index);
  dev_ctx.template Alloc<int32_t>(scatter_index_rev);
  dev_ctx.template Alloc<int64_t>(expert_offset);
  dev_ctx.template Alloc<int64_t>(expert_nums_local);
  dev_ctx.template Alloc<float>(combine_weights_out);
  phi::Full<int32_t, Context>(
      dev_ctx,
      phi::IntArray(common::vectorize(scatter_index->dims())),
      0,
      scatter_index);
  phi::Full<int32_t, Context>(
      dev_ctx,
      phi::IntArray(common::vectorize(scatter_index_rev->dims())),
      0,
      scatter_index_rev);
  phi::Full<int64_t, Context>(
      dev_ctx,
      phi::IntArray(common::vectorize(expert_offset->dims())),
      0,
      expert_offset);
  phi::Full<int64_t, Context>(
      dev_ctx,
      phi::IntArray(common::vectorize(expert_nums_local->dims())),
      0,
      expert_nums_local);
  phi::Full<float, Context>(
      dev_ctx,
      phi::IntArray(common::vectorize(combine_weights_out->dims())),
      0,
      combine_weights_out);
  phi::Copy(
      dev_ctx, combine_weights, dev_ctx.GetPlace(), false, combine_weights_out);
  const auto &x_shape = x.dims();
  int64_t num_rows = x_shape[0];
  int64_t hidden_size = x_shape[1];
  thrust::host_vector<int64_t> expert_offset_host(num_experts);
  int64_t num_experts_diff = expert_end_index - expert_start_index;
  moe_dispatch_fwd<T, Context>(dev_ctx,
                               x,
                               num_rows,
                               num_experts,
                               hidden_size,
                               capacity,
                               k,
                               expert_start_index,
                               expert_end_index,
                               reverse_token_drop,
                               expert_offset_host,
                               y,
                               *combine_weights_out,
                               *scatter_index,
                               *scatter_index_rev,
                               *expert_offset,  // global-offset
                               *expert_nums_local,
                               expert_id,
                               use_pad);
  if (use_pad) {
    // scatter_index_rev = scatter_index_rev.slice(0, num_experts_diff *
    // capacity);
    *scatter_index_rev = phi::Slice<int32_t, Context>(
        dev_ctx, *scatter_index_rev, {0}, {0}, {num_experts_diff * capacity});
  } else {
    if (expert_offset_host.back() > 0) {
      // scatter_index_rev = scatter_index_rev.slice(0,
      // expert_offset_host.back());
      *scatter_index_rev = phi::Slice<int32_t, Context>(
          dev_ctx, *scatter_index_rev, {0}, {0}, {expert_offset_host.back()});
    } else {
      *y = phi::Empty<T, Context>(dev_ctx, {1, x_shape[1]});
      *scatter_index_rev =
          phi::Empty<int32_t, Context>(dev_ctx, {});  // special treatment
    }
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(moe_gate_dispatch_partial_nosoftmaxtopk,
                   GPU,
                   ALL_LAYOUT,
                   phi::MoeGateDispatchPartialNoSoftMaxTopkKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
