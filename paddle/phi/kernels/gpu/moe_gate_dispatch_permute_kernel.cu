// NOLINT
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

#include "paddle/phi/kernels/moe_gate_dispatch_permute_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/moe_fuse_op.h"
namespace phi {

namespace {
// --------      getWorkspaceSize      -------- //
template <typename KeyT>
size_t getWorkspaceSize(const int num_rows,
                        const int hidden_size,
                        const int inter_size,
                        const int num_experts,
                        const int k,
                        //  const int max_seq_len,
                        phi::CubKeyValueSorter &sorter  // NOLINT
) {
  // const int buf_size = AlignTo16(k * num_rows * hidden_size);
  // const int interbuf_size = AlignTo16(k * num_rows * inter_size);
  // const int padded_experts = AlignTo16(num_experts);
  const int num_moe_inputs = AlignTo16(k * num_rows);
  int num_softmax_outs = 0;

  // softmax output, permuted_rows and permuted_experts have moved to outside of
  // moe kernel, allocate them in Encoder or Decoder before invoking FfnLayer
  // forward.
  size_t total_ws_bytes =
      4 * num_moe_inputs *
      sizeof(int);  // source_rows_, permuted_rows_, permuted_experts_
  // total_ws_bytes += buf_size * sizeof(KeyT);                // permuted_data
  // total_ws_bytes += padded_experts * sizeof(int64_t);        // Hold
  // total_rows_before_expert_  // expert_cnt total_ws_bytes += num_softmax_outs
  // * sizeof(KeyT); const int bytes_for_fc1_result = interbuf_size *
  // sizeof(KeyT);
  const int sorter_ws_size_bytes =
      AlignTo16(sorter.getWorkspaceSize(k * num_rows));
  // sorter.update_num_experts(num_experts+1); // +1 for filter out of capacity
  // // 用所有 bit 做排序,会降低些许性能,但是防止越界
  total_ws_bytes += sorter_ws_size_bytes;  // intermediate (fc1) output + cub
                                           // sorting workspace
  // std::cout<<"sorter_ws_size_bytes = "<<sorter_ws_size_bytes  << "
  // num_moe_inputs = "<<num_moe_inputs<<", total_ws_bytes =
  // "<<total_ws_bytes<<std::endl;
  return total_ws_bytes;
}
}  // namespace

template <typename T, typename Context>
void apply_moe_dispatch_fwd(const Context &dev_ctx,
                            const T *x,
                            const float *gate_logits,
                            const float *corr_bias,
                            int64_t num_rows,
                            int64_t num_experts,
                            int64_t hidden_size,
                            int64_t capacity,
                            int64_t k,
                            T *y,
                            float *combine_weights,
                            int *scatter_index,
                            int64_t *expert_offset,
                            int *expert_id,
                            bool use_pad,
                            bool use_all2all_permute,
                            int64_t world_size,
                            int64_t num_local_experts,
                            cudaStream_t stream) {
  phi::CubKeyValueSorter sorter(stream);
  // phi::funcs::SetConstant<phi::GPUContext, bool> zero;
  // zero(ctx, &finished_tensor, false);

  DenseTensor xpanded_source_row_to_expanded_dest_row_tensor =
      phi::Empty<int, Context>(dev_ctx, IntArray({num_rows, k}));
  // int* expanded_source_row_to_expanded_dest_row =
  //     expanded_source_row_to_expanded_dest_row_tensor.data<int>();

  // paddle::Tensor expert_scales_tensor_float = paddle::empty({num_rows, k},
  // paddle::DataType::FLOAT32, place); float* expert_scales_float =
  // expert_scales_tensor_float.data<float>();

  // paddle::Tensor expert_for_source_row_tensor = paddle::empty({num_rows, k},
  // paddle::DataType::INT32, place); int* expert_for_source_row =
  // expert_for_source_row_tensor.data<int>();
  DenseTensor active_cnt_tensor =
      phi::Empty<int, Context>(dev_ctx, IntArray({1}));

  int64_t bytes = getWorkspaceSize<T>(num_rows,
                                      hidden_size,  // hidden-size=0
                                      0,            // inter-size=0
                                      num_experts,
                                      k,
                                      sorter);

  DenseTensor ws_ptr_tensor =
      phi::Empty<int8_t, Context>(dev_ctx, IntArray({bytes}));
  int8_t *ws_ptr = ws_ptr_tensor.data<int8_t>();

  // Pointers
  int *source_rows_;
  int *permuted_rows_;
  int *permuted_experts_;
  int *expert_id_;

  // T* permuted_data_;
  float *softmax_out_;
  // int64_t* total_rows_before_expert_;
  T *fc1_result_;

  const int sorter_ws_size_bytes =
      AlignTo16(sorter.getWorkspaceSize(k * num_rows));
  // const int buf_size = AlignTo16(k * num_rows * hidden_size);
  // const int interbuf_size  = AlignTo16(k * num_rows * 0);
  const int padded_experts = AlignTo16(num_experts);
  const int num_moe_inputs = AlignTo16(k * num_rows);

  source_rows_ = reinterpret_cast<int *>(ws_ptr);
  permuted_rows_ = source_rows_ + num_moe_inputs;
  permuted_experts_ = permuted_rows_ + num_moe_inputs;
  expert_id_ = permuted_experts_ + num_moe_inputs;

  // permuted_data_ = reinterpret_cast<T *>(expert_id_ + num_moe_inputs);
  // total_rows_before_expert_ = reinterpret_cast<int64_t*>(permuted_experts_ +
  // buf_size);

  // only use one number
  // num_active   = reinterpret_cast<int64_t*>(permuted_experts_ +
  // num_moe_inputs);

  fc1_result_ = reinterpret_cast<T *>(expert_id_ + num_moe_inputs);
  softmax_out_ = nullptr;

#ifdef DEBUG_MOE_OP
  // print_to_screen1(gate_logits, 8, 16, std::string("gate_logits
  // before_topk")); print_to_screen1(finished, 2, 16, std::string("finished
  // before_topk"));
#endif

  topk_gating_softmax_kernelLauncher<float>(gate_logits,
                                            corr_bias,
                                            combine_weights,  // output
                                            softmax_out_,     // no use
                                            expert_id,        // output
                                            source_rows_,     // output
                                            num_rows,
                                            num_experts,
                                            k,
                                            stream);

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
  // modify expert-id according to k
  if (use_pad)  // 为了区分 k=1 选择和 k=2 选择，修改 expert-id
    modify_expert_id_launcher(
        expert_id, expert_id_, k, num_rows, num_experts, stream);

    // calc expert-size
    /*
      if (!use_pad)
        cal_expert_size_and_filter_launcher(expert_id,
                                            k * num_rows,
                                            num_experts,
                                            capacity,
                                            stream);
    */
#ifdef DEBUG_MOE_OP
  print_to_screen1<int>(
      expert_id, 8, 16, std::string("expert-id after modified"));
#endif
  sorter.run(
      fc1_result_,
      sorter_ws_size_bytes,
      use_pad ? expert_id_ : expert_id,  // key in
      permuted_experts_,                 // key out // [num_row, k]: expert-id
      source_rows_,                      // value in
      permuted_rows_,  // value out //[num_row, k]: id在原 activation 中的位置
      k * num_rows,  // num_rows
      false,
      stream);

  if (use_pad)
    unmodify_expert_id_launcher(
        permuted_experts_, permuted_experts_, k, num_rows, num_experts, stream);

#ifdef DEBUG_MOE_OP
  print_to_screen1<int>(
      permuted_experts_, 8, 16, std::string("expert-id after permute"));
  print_to_screen1<int>(
      permuted_rows_, 8, 16, std::string("dest->src idx after permute"));
#endif

  compute_total_rows_before_expert(
      permuted_experts_, k * num_rows, num_experts, expert_offset, stream);

#ifdef DEBUG_MOE_OP
  print_to_screen1(expert_offset, 8, 16, std::string("expert_offset"));
  int64_t num_active_host_v2;
  cudaMemcpy(&num_active_host_v2,
             expert_offset + num_experts - 1,
             sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  std::cerr << "[DEBUG] num_active v2: " << num_active_host_v2 << std::endl;
  print_to_screen1<int>(permuted_experts_,
                        8,
                        num_active_host_v2 + 2,
                        std::string("<reprint>expert-id after permute"));
  // print_to_screen1(permuted_experts_, 4096, 8192,
  // std::string("<reprint>expert-id after permute"));
#endif

  if (!use_all2all_permute) {
    initialize_moe_routing_kernelLauncher(x,
                                          y,
                                          permuted_rows_,
                                          scatter_index,
                                          permuted_experts_,
                                          expert_offset,
                                          combine_weights,
                                          static_cast<int>(num_rows),
                                          static_cast<int>(hidden_size),
                                          static_cast<int>(k),
                                          capacity,
                                          use_pad,
                                          stream);
  } else {
    PD_CHECK(num_experts > 0);
    PD_CHECK(world_size > 0);
    initialize_moe_routing_permute_kernelLauncher(x,
                                                  y,
                                                  permuted_rows_,
                                                  scatter_index,
                                                  permuted_experts_,
                                                  expert_offset,
                                                  combine_weights,
                                                  static_cast<int>(num_rows),
                                                  static_cast<int>(hidden_size),
                                                  static_cast<int>(k),
                                                  capacity,
                                                  world_size,
                                                  num_local_experts,
                                                  stream);
  }

  // turn expert_offset_ptr into experts_num
  // auto expert_offset_ptr = thrust::device_pointer_cast(expert_offset);
  // thrust::adjacent_difference(
  //   expert_offset_ptr, expert_offset_ptr + num_experts, expert_offset_ptr
  // );
#ifdef DEBUG_MOE_OP
  print_to_screen1<int>(
      scatter_index, 8, 16, std::string("scatter_index after pad"));
#endif
  // cudaMemcpy(scatter_index, permuted_rows_, sizeof(int64_t) * k * num_rows,
  // cudaMemcpyDeviceToDevice); cudaMemcpy(combine_weights, expert_scales_float,
  // sizeof(float) * k * num_rows, cudaMemcpyDeviceToDevice);
  return;
}

template <typename T, typename Context>
void moe_dispatch_fwd(const Context &dev_ctx,
                      const DenseTensor &x,
                      const DenseTensor &gate_logits,
                      const paddle::optional<DenseTensor> &corr_bias,
                      int64_t num_rows,
                      int64_t num_experts,
                      int64_t hidden_size,
                      int64_t capacity,
                      int64_t k,
                      const DenseTensor &y,
                      const DenseTensor &combine_weights,
                      const DenseTensor &scatter_index,
                      const DenseTensor &expert_offset,
                      const DenseTensor &expert_id,
                      bool use_pad,
                      int64_t use_all2all_permute = false,
                      int64_t world_size = -1,
                      int64_t num_local_experts = -1) {
  apply_moe_dispatch_fwd<T, Context>(
      dev_ctx,
      x.data<T>(),
      gate_logits.data<float>(),
      corr_bias ? corr_bias.get_ptr()->data<float>() : nullptr,
      num_rows,
      num_experts,
      hidden_size,
      capacity,
      k,
      const_cast<T *>(y.data<T>()),
      const_cast<float *>(combine_weights.data<float>()),
      const_cast<int *>(scatter_index.data<int>()),
      const_cast<int64_t *>(expert_offset.data<int64_t>()),
      const_cast<int *>(expert_id.data<int>()),
      use_pad,
      use_all2all_permute,
      world_size,
      num_local_experts,
      dev_ctx.stream());
}

template <typename T, typename Context>
void MoEDispatchPermuteKernel(const Context &dev_ctx,
                              const DenseTensor &x,
                              const DenseTensor &gate_logits,
                              const paddle::optional<DenseTensor> &corr_bias,
                              int64_t k,
                              int64_t capacity,
                              int64_t world_size,
                              DenseTensor *y,
                              DenseTensor *combine_weights,
                              DenseTensor *scatter_index,
                              DenseTensor *expert_offset,
                              DenseTensor *expert_id) {
  dev_ctx.template Alloc<int>(expert_id);
  dev_ctx.template Alloc<int64_t>(expert_offset);
  dev_ctx.template Alloc<int>(scatter_index);
  dev_ctx.template Alloc<float>(combine_weights);
  dev_ctx.template Alloc<T>(y);
  phi::Full<T, Context>(
      dev_ctx, phi::IntArray(common::vectorize(y->dims())), 0, y);
  const auto &x_shape = x.dims();
  const auto &gate_logits_shape = gate_logits.dims();
  int64_t num_rows = x_shape[0];
  int64_t hidden_size = x_shape[1];
  int64_t num_experts = gate_logits_shape[1];
  int64_t num_local_experts = num_experts / world_size;
  moe_dispatch_fwd<T, Context>(dev_ctx,
                               x,
                               gate_logits,
                               corr_bias,
                               num_rows,
                               num_experts,
                               hidden_size,
                               capacity,
                               k,
                               *y,
                               *combine_weights,
                               *scatter_index,
                               *expert_offset,
                               *expert_id,
                               true, /*use_pad*/
                               true, /*use_all2all_permute*/
                               world_size,
                               num_local_experts);
}
}  // namespace phi

PD_REGISTER_KERNEL(moe_gate_dispatch_permute,
                   GPU,
                   ALL_LAYOUT,
                   phi::MoEDispatchPermuteKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
