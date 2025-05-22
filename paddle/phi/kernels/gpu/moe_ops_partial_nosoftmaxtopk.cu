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
#include <cassert>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "paddle/extension.h"
#include "paddle/phi/api/all.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/extension.h"
#include "fused_moe_op.h"
#include "fused_moe_bwd_op.h"
#include "fleety_utils.h"
#ifdef MOE_OPS_AUTO
#include "paddle/phi/infermeta/spmd_rules/rules.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"
#include "paddle/phi/api/ext/spmd_infer.h"
#endif

#define CHECK_CUDA(x) PD_CHECK(!x.is_cpu(), #x " must be a CUDA tensor")
#define DEFAULT_THROW(NAME, TYPE)                           \
  default:                                                  \
    do                                                      \
    {                                                       \
      PD_THROW(#NAME, " not implemented for '", TYPE, "'"); \
    } while (0);                                            \
    break

#define CUDACHECK(cmd)                                                         \
    do {                                                                       \
        cudaError_t e = cmd;                                                   \
        if (e != cudaSuccess) {                                                \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(e));                                     \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(TYPEIN, NAME, ...) \
  switch (TYPEIN)                                                     \
  {                                                                   \
  case paddle::DataType::FLOAT32:                                     \
  {                                                                   \
    using scalar_t_in = float;                                        \
    __VA_ARGS__;                                                      \
    break;                                                            \
  }                                                                   \
  case paddle::DataType::FLOAT16:                                     \
  {                                                                   \
    using scalar_t_in = phi::dtype::float16;                          \
    __VA_ARGS__;                                                      \
    break;                                                            \
  }                                                                   \
  case paddle::DataType::BFLOAT16:                                    \
  {                                                                   \
    using scalar_t_in = phi::dtype::bfloat16;                         \
    __VA_ARGS__;                                                      \
    break;                                                            \
  }                                                                   \
    std::cerr << "dispatch failed" << std::endl; \
    DEFAULT_THROW(NAME, TYPEIN);                                      \
  }

static inline size_t AlignTo16(const size_t &input)
{
  static constexpr int ALIGNMENT = 16;
  return ALIGNMENT * ((input + ALIGNMENT - 1) / ALIGNMENT);
}
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
                        phi::CubKeyValueSorter &sorter)
{

  // const int buf_size = AlignTo16(k * num_rows * hidden_size);
  const int interbuf_size = AlignTo16(k * num_rows * inter_size);
  const int padded_experts = AlignTo16(num_experts);
  const int num_moe_inputs = AlignTo16(k * num_rows);
  const int num_dispatched_size = AlignTo16(num_experts * capacity);
  int num_softmax_outs = 0;

  // softmax output, permuted_rows and permuted_experts have moved to outside of moe kernel, allocate them
  // in Encoder or Decoder before invoking FfnLayer forward.
  size_t total_ws_bytes = 4 * num_moe_inputs * sizeof(int); // source_rows_, permuted_rows_, permuted_experts_
  total_ws_bytes += 2 * num_dispatched_size * sizeof(int);
  total_ws_bytes += padded_experts * sizeof(int64_t);        // Hold total_rows_before_expert_  // expert_cnt
  // total_ws_bytes += buf_size * sizeof(KeyT);                // permuted_data
  total_ws_bytes += num_softmax_outs * sizeof(KeyT);
  const int bytes_for_fc1_result = interbuf_size * sizeof(KeyT);
  const int sorter_ws_size_bytes = 
    std::max(AlignTo16(sorter.getWorkspaceSize(k * num_rows)), 
             AlignTo16(sorter.getWorkspaceSize(capacity)));
  //sorter.update_num_experts(num_experts+1); // +1 for filter out of capacity // 用所有 bit 做排序,会降低些许性能,但是防止越界
  int bytes_for_intermediate_and_sorting = bytes_for_fc1_result;
  if (sorter_ws_size_bytes > bytes_for_fc1_result)
  {
    int remaining_bytes = AlignTo16(sorter_ws_size_bytes - bytes_for_fc1_result);
    bytes_for_intermediate_and_sorting += remaining_bytes;
  }
  // std::cout<<"num_softmax_outs --"<< num_softmax_outs << std::endl;
  total_ws_bytes += bytes_for_intermediate_and_sorting; // intermediate (fc1) output + cub sorting workspace
  // std::cout<<"buf_size --"<< buf_size<<"   "<<interbuf_size<< "   "<<padded_experts<< "    "<<num_moe_inputs<<  "  "<<total_ws_bytes<< "   "<< bytes_for_fc1_result<< "   "<<sorter_ws_size_bytes  << "  "<<std::endl;
  return total_ws_bytes;
}

template <typename T>
void apply_moe_dispatch_fwd(
    const T *x,
    int64_t num_rows,
    int64_t num_experts,
    int64_t hidden_size,
    int64_t capacity,
    int64_t k,
    int64_t expert_start_index,
    int64_t expert_end_index,
    bool reverse_token_drop,
    thrust::host_vector<int64_t>& expert_offset_host,
    T *y,
    float *combine_weights,
    int *scatter_index,
    int * scatter_index_rev,
    int64_t *expert_offset_global,
    int64_t* expert_nums_local,
    int *expert_id,
    bool use_pad,
    cudaStream_t stream,
    const phi::Place &place)
{
  phi::CubKeyValueSorter sorter(stream);
  paddle::Tensor expanded_source_row_to_expanded_dest_row_tensor =
      paddle::empty({num_rows, k}, paddle::DataType::INT32, place);
  // int* expanded_source_row_to_expanded_dest_row =
  //     expanded_source_row_to_expanded_dest_row_tensor.data<int>();

  // paddle::Tensor expert_scales_tensor_float = paddle::empty({num_rows, k}, paddle::DataType::FLOAT32, place);
  // float* expert_scales_float = expert_scales_tensor_float.data<float>();

  // paddle::Tensor expert_for_source_row_tensor = paddle::empty({num_rows, k}, paddle::DataType::INT32, place);
  // int* expert_for_source_row = expert_for_source_row_tensor.data<int>();
  // paddle::Tensor active_cnt_tensor = paddle::empty({1}, paddle::DataType::INT32, place);

  int64_t bytes = getWorkspaceSize<T>(num_rows,
                                      hidden_size, // hidden-size=0
                                      0,           // inter-size=0
                                      num_experts,
                                      capacity,
                                      k,
                                      use_pad,
                                      sorter);

  paddle::Tensor ws_ptr_tensor = paddle::empty({bytes}, paddle::DataType::INT8, place);
  int8_t *ws_ptr = ws_ptr_tensor.data<int8_t>();

  phi::memory_utils::ThrustAllocator<cudaStream_t> allocator(place, stream);

  // Pointers
  int *source_rows_;
  int *permuted_rows_;
  int *permuted_experts_;
  int *expert_id_;
  int *source_rows_for_seqsort_;
  int *source_rows_for_seqsort_out_;
  int *source_pos_for_seqsort_;
  int *source_pos_for_seqsort_out_;
  int64_t *expert_offset_; // local-expert-offset

  char *sorter_ws_;
  // T* permuted_data_;
  float *softmax_out_;
  // int64_t* total_rows_before_expert_;
  T *fc1_result_;

  const int sorter_ws_size_bytes = AlignTo16(sorter.getWorkspaceSize(k * num_rows));
  const int sorter_ws_size_bytes_seqsort = AlignTo16(sorter.getWorkspaceSize(capacity));

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
  expert_offset_ = reinterpret_cast<int64_t *> (source_rows_for_seqsort_out_ + num_dispatched_size);
  // permuted_data_ = reinterpret_cast<T *>(expert_offset_ + padded_experts);
  // total_rows_before_expert_ = reinterpret_cast<int64_t*>(permuted_experts_ + buf_size);

  // only use one number
  // num_active   = reinterpret_cast<int64_t*>(permuted_experts_ + num_moe_inputs);
  fc1_result_ = reinterpret_cast<T *>(expert_offset_ + padded_experts);
  // fc1_result_ = reinterpret_cast<T *>(permuted_data_ + buf_size);
 
#ifdef DEBUG_MOE_OP
  // print_to_screen1(gate_logits, 8, 16, std::string("gate_logits before_topk"));
  // print_to_screen1(finished, 2, 16, std::string("finished before_topk"));
#endif

  thrust::transform(
    thrust::cuda::par.on(stream),
    thrust::device_pointer_cast(source_rows_),
    thrust::device_pointer_cast(source_rows_) + num_rows * k,
    thrust::counting_iterator<int>(0),
    thrust::device_pointer_cast(source_rows_),
    [num_rows, k] __device__ (int i, int cnt) {
      int k_idx = cnt % k;
      int block_row = cnt / k;
      return k_idx * num_rows + block_row;
    }
  );
  
#ifdef DEBUG_MOE_OP
  // phi::CastKernel<float>(ctx, expert_scales_tensor_float, expert_scales_tensor.dtype(), &expert_scales_tensor);
  print_to_screen1(combine_weights, 8, 16, std::string("expert_scales_float after topk"));
  print_to_screen1<int>(expert_id, 8, 16, std::string("expert-id before permute"));
  print_to_screen1<int>(source_rows_, 8, 16, std::string("desc->src idx before permute"));
#endif

  // compute global expert offset, **not** consider capacity
  // 必须在 modify_and_mask_expert_id_launcher 之前算出**全局 expert-offset**

  compute_global_expert_offset(expert_id, 
    expert_id_, //buffer
    expert_offset_global, 
    num_rows * k, 
    num_experts, 
    capacity,
    stream,
    allocator);

  // modifiy expert-id according to k
  modify_and_mask_expert_id_launcher(expert_id, 
    expert_id_, 
    k, 
    num_rows, 
    static_cast<int>(num_experts), 
    static_cast<int>(expert_start_index), 
    static_cast<int>(expert_end_index), 
    stream);


 #ifdef DEBUG_MOE_OP
  print_to_screen1<int>(expert_id_, 8, 16, std::string("expert-id after modified 22"));
#endif
  sorter.run(fc1_result_,
             sorter_ws_size_bytes,
             expert_id_,         // key in
             permuted_experts_, // key out // [num_row, k]: expert-id
             source_rows_,      // value in
             permuted_rows_,    // value out //[num_row, k]: id在原 activation 中的位置
             k * num_rows,      // num_rows
             false,
             stream);

  unmodify_expert_id_launcher(permuted_experts_, permuted_experts_, k, num_rows, num_experts, stream);

#ifdef DEBUG_MOE_OP
  print_to_screen1<int>(permuted_experts_, 8, 16, std::string("expert-id after permute"));
  print_to_screen1<int>(permuted_rows_, 8, 16, std::string("dest->src idx after permute"));
#endif

  compute_local_expert_offset(
    permuted_experts_,
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
  std::cerr << "[DEBUG] num_active v2: " << expert_offset_host.back() << std::endl;
  print_to_screen1(expert_offset_global, 8, 16, std::string("expert_offset global"));
  print_to_screen1(expert_offset_, 8, 16, std::string("expert_offset local"));
  print_to_screen1<int>(permuted_experts_, 8, 16, std::string("<reprint>expert-id after permute"));
  // print_to_screen1(permuted_experts_, 4096, 8192, std::string("<reprint>expert-id after permute"));
#endif
  
  // calc expert-size
  // 不 use-pad 的情况下，在此处标记截断位置。之后需要再 sort 一遍把截断 id 放到句尾
  if (!use_pad){ // 2sort
    cal_expert_size_and_filter_launcher(permuted_experts_,
                                        expert_offset_,
                                        expert_offset_host.back(),
                                        num_experts,
                                        capacity,
                                        expert_start_index,
                                        expert_end_index,
                                        reverse_token_drop,
                                        stream);
    //2sort
    sorter.run(fc1_result_,
              sorter_ws_size_bytes,
              permuted_experts_,         // key in
              permuted_experts_, // key out // [num_row, k]: expert-id
              permuted_rows_,    // value in
              permuted_rows_,    // value out //[num_row, k]: id在原 activation 中的位置
              k * num_rows,      // num_rows
              false,
              stream);

    compute_local_expert_offset(
      permuted_experts_,
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
    std::cerr << "[DEBUG](after 2sort) num_active v2: " << expert_offset_host.back() << std::endl;
    print_to_screen1<int>(expert_id_, 8, 16, std::string("<before 2sort> permuted_experts"));
    print_to_screen1<int>(permuted_experts_, 8, 16, std::string("<after 2sort> permuted_experts"));
    print_to_screen1(permuted_rows_, 8,16, std::string("<after 2sort> dest->src idx"));
#endif              
  }

  thrust::fill(
    thrust::cuda::par.on(stream),
    thrust::device_ptr<int>(scatter_index_rev),
    thrust::device_ptr<int>(scatter_index_rev) + num_experts * capacity,
    num_rows
  );
  build_seqsort_kv_pairs_kernel_launcher(scatter_index_rev, //padded_to_unpermuted_input
                                        source_rows_for_seqsort_, //seqsort-value
                                        permuted_rows_,
                                        // scatter_index, // 对截断位置置0
                                        permuted_experts_,
                                        expert_offset_,
                                        combine_weights, // 对截断位置置0
                                        static_cast<int>(num_rows),
                                        static_cast<int>(k),                                       
                                        expert_offset_host.back(), //num_active
                                        capacity,
                                        expert_start_index, // expert start index
                                        use_pad,
                                        stream);

#ifdef DEBUG_MOE_OP

  // print_to_screen1<int>(scatter_index, 8, 16, std::string("scatter_index after build_seqsort_kv_pairs_kernel_launcher"));
  print_to_screen1<int>(source_rows_for_seqsort_, 8, 16, std::string("source_rows_for_seqsort_ after build_seqsort_kv_pairs_kernel_launcher"));
  print_to_screen1<int>(scatter_index_rev, 8, 16, std::string("scatter_index_rev after build_seqsort_kv_pairs_kernel_launcher"));
#endif
  if (use_pad){
    for (auto iexpert = 0; iexpert != expert_end_index - expert_start_index; ++iexpert){      
      sorter.run(fc1_result_,
                sorter_ws_size_bytes_seqsort,
                scatter_index_rev + (iexpert * capacity),      // key in
                scatter_index_rev  + (iexpert * capacity), // key out 
                source_rows_for_seqsort_ + (iexpert * capacity),         // value in
                source_rows_for_seqsort_ + (iexpert * capacity),     // value out //[num_row, k]: id在原 activation 中的位置
                capacity,      // num_rows
                false,
                stream);    
    }
  }else{
    auto sort_iter = thrust::make_zip_iterator(thrust::make_tuple(
      thrust::device_pointer_cast(permuted_experts_), //key1
      thrust::device_pointer_cast(scatter_index_rev), //key2
      thrust::device_pointer_cast(source_rows_for_seqsort_)
    ));
    thrust::stable_sort(
      thrust::cuda::par.on(stream),
      sort_iter,
      sort_iter + expert_offset_host.back(),
      []__device__(auto lhs, auto rhs){
        if (thrust::get<0>(lhs) < thrust::get<0>(rhs))
          return true;
        else if(thrust::get<0>(lhs) > thrust::get<0>(rhs))
          return false;
        else
          return thrust::get<1>(lhs) < thrust::get<1>(rhs);         
      }      
    );
  }
#ifdef DEBUG_MOE_OP
    print_to_screen1<int>(source_rows_for_seqsort_, 8, 16, std::string("padded to unpermuted_input after 2sort"));
    print_to_screen1<int>(scatter_index_rev, 8, 16, std::string("scatter_index_rev after 2sort"));
#endif                
  // cudaDeviceSynchronize(); //debug

  copy_unpermuted_to_permuted_kernelLauncher(x, 
    y, //out
    scatter_index_rev, //padded_out_to_unpermuted_input
    source_rows_for_seqsort_,  //padded_out_to_expanded_input
    scatter_index, //out
    use_pad? (expert_end_index - expert_start_index) * capacity : expert_offset_host.back(),  //num_active
    num_rows,
    k,
    hidden_size,
    stream);
  // cudaDeviceSynchronize(); //debug
  // turn expert_offset_ptr into experts_num
  return;
}

void moe_dispatch_fwd(const paddle::Tensor &x,            
                      int64_t num_rows,
                      int64_t num_experts,
                      int64_t hidden_size,
                      int64_t capacity,
                      int64_t k,                      
                      int64_t expert_start_index,
                      int64_t expert_end_index,
                      bool reverse_token_drop,
                      thrust::host_vector<int64_t>& expert_offset_host,                      
                      const paddle::Tensor &y,
                      const paddle::Tensor &combine_weights,
                      const paddle::Tensor &scatter_index,
                      const paddle::Tensor &scatter_index_rev,
                      const paddle::Tensor &expert_offset,
                      const paddle::Tensor &expert_nums_local,
                      const paddle::Tensor &expert_id,
                      bool use_pad
                    )

{
  DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
      x.type(),
      "apply_moe_dispatch_fwd",
      apply_moe_dispatch_fwd(
          x.data<scalar_t_in>(),
          num_rows,
          num_experts,
          hidden_size,
          capacity,
          k,
          expert_start_index,
          expert_end_index,
          reverse_token_drop,
          expert_offset_host,
          const_cast<scalar_t_in *>(y.data<scalar_t_in>()),
          const_cast<float *>(combine_weights.data<float>()),
          const_cast<int *>(scatter_index.data<int>()),
          const_cast<int *>(scatter_index_rev.data<int>()),
          const_cast<int64_t *>(expert_offset.data<int64_t>()),
          const_cast<int64_t *>(expert_nums_local.data<int64_t>()),
          const_cast<int *>(expert_id.data<int>()),
          use_pad,
          x.stream(),
          x.place()));
}

std::vector<paddle::Tensor> MoEDispatchFwd(const paddle::Tensor &x,
                                           const paddle::Tensor &combine_weights,
                                           const paddle::Tensor &expert_id,
                                           int64_t k,
                                           int64_t capacity,
                                           int64_t num_experts,
                                           bool use_pad,
                                           int64_t expert_start_index,
                                           int64_t expert_end_index,
                                           bool reverse_token_drop)
{

  const auto &x_shape = x.shape();
  const auto &combine_weights_shape = combine_weights.shape();

  PD_CHECK(x_shape.size() == 2);
  PD_CHECK(combine_weights_shape.size() == 2);

  int64_t num_rows = x_shape[0];
  int64_t hidden_size = x_shape[1];
  PD_CHECK(expert_end_index > expert_start_index);
  int64_t num_experts_diff = expert_end_index - expert_start_index;
  PD_CHECK(num_rows == combine_weights_shape[0]);
  PD_CHECK(combine_weights.type() == paddle::DataType::FLOAT32);
  PD_CHECK(expert_id.type() == paddle::DataType::INT32);
  PD_CHECK(k>0);
  PD_CHECK(num_rows>0);
  CHECK_CUDA(x);
  PD_CHECK(num_experts >= k);
  PD_CHECK(!reverse_token_drop || !use_pad); //use_pad=false 模式才支持设置 reverse_token_drop=true
  PD_CHECK(combine_weights.type() == paddle::DataType::FLOAT32);

  std::vector<int64_t> y_shape; 
#ifdef MOE_OPS_AUTO
  if (use_pad)
    y_shape = {num_experts_diff, capacity, x_shape[1]} ;
  else
    y_shape = {num_rows, k, x_shape[1]};
#else
  if (use_pad)
    y_shape = {num_experts_diff * capacity, x_shape[1]} ;
  else
    y_shape = {num_rows * k, x_shape[1]};
#endif

#ifdef DEBUG_MOE_OP
  std::cerr << "[DEBUG] infer-shape: k=" << k << " num_rows=" << num_rows << " use_pad="<< use_pad << " reverse_token_drop=" << reverse_token_drop
  <<" capacity="<< capacity << " num_experts="<<num_experts << " expert_start_index:"<<expert_start_index << " expert_end_index:" <<  expert_end_index
  <<" y_shape:"<< join_strings(y_shape, ',') << std::endl;
#endif
 
  auto place = x.place();
  paddle::Tensor y = paddle::zeros(y_shape, x.type(), place);
  paddle::Tensor scatter_index_rev = paddle::zeros({num_experts * capacity}, paddle::DataType::INT32, place);
  paddle::Tensor scatter_index = paddle::zeros({k, num_rows}, paddle::DataType::INT32, place);
  paddle::Tensor expert_offset = paddle::zeros({num_experts}, paddle::DataType::INT64, place);
  paddle::Tensor expert_nums_local = paddle::zeros({num_experts}, paddle::DataType::INT64, place);
  thrust::host_vector<int64_t> expert_offset_host(num_experts);

  moe_dispatch_fwd(x,
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
                   combine_weights,
                   scatter_index,
                   scatter_index_rev,
                   expert_offset, //global-offset
                   expert_nums_local,
                   expert_id,
                   use_pad
                  );
  if(use_pad){
    scatter_index_rev = scatter_index_rev.slice(0, num_experts_diff * capacity);
  }else{
    if (expert_offset_host.back() > 0){
      y = y.slice(0, expert_offset_host.back());
      scatter_index_rev = scatter_index_rev.slice(0, expert_offset_host.back());
    }else{
      y = paddle::zeros({1, x_shape[1]}, x.type(), place);
      scatter_index_rev = paddle::zeros({}, paddle::DataType::INT32, place); //special treatment
    }
  }
  return {y, combine_weights, scatter_index, scatter_index_rev, expert_offset, expert_nums_local};
}

std::vector<std::vector<int64_t>> MoEDispatchFwdInferShape(
    std::vector<int64_t> x_shape,
    std::vector<int64_t> gate_shape,
    int64_t k,
    int64_t capacity,
    bool use_pad)
{
  int64_t num_rows = x_shape[0];
  std::cerr << "infer-shape: k" << k << " num_rows" << num_rows << std::endl;
  int64_t dim = x_shape[1];
  int64_t num_experts = gate_shape[1];
  return {{num_rows * k, dim}, {num_rows, k}, {k, num_rows}, {num_experts}, {num_rows, k}};
}

std::vector<paddle::DataType> MoEDispatchFwdInferDtype(
    paddle::DataType x_dtype,
    paddle::DataType gate_dtype)
{
  return {x_dtype, paddle::DataType::FLOAT32, paddle::DataType::INT32, paddle::DataType::INT64, paddle::DataType::INT32};
}


template <typename T>
void apply_moe_dispatch_bwd(
    const T* y_grad,
    const float* combine_weights, // [s, k]
    const int* scatter_index,   // [s, k]
    const float* combine_weights_out_grad,
    float* combine_weights_in_grad,
    T* x_grad,
    int64_t num_rows,
    int64_t k,
    int64_t dim,
    int64_t num_experts,
    int64_t num_active,
    cudaStream_t stream){
#ifdef DEBUG_MOE_OP
    std::cerr << "[DEBUG-BWD] x3 launch kernel, num_active=" << num_active << std::endl;
#endif      
    gather_with_mask_launcher<T>(y_grad,
                                scatter_index, 
                                combine_weights,
                                x_grad, num_rows, k, dim, num_active, stream);
    auto out_grad_ptr = thrust::device_pointer_cast(combine_weights_out_grad);
    auto in_grad_ptr = thrust::device_pointer_cast(combine_weights_in_grad);
    auto combine_weight_ptr = thrust::device_pointer_cast(combine_weights);
    thrust::transform(
      thrust::cuda::par.on(stream),
      out_grad_ptr,
      out_grad_ptr + num_rows * k,
      combine_weight_ptr,
      in_grad_ptr,
      [] __device__ (float g, float w){
          return w > static_cast<float>(0) ? g : static_cast<float>(0);
      }
    );
    // topk_grad_with_mask_launcher<float>(combine_weights_grad,
    //                                     expert_id,
    //                                     combine_weights,
    //                                     gate_logtis_grad,
    //                                     num_rows, k, num_experts, stream);
}


void moe_dispatch_bwd(const paddle::Tensor &combine_weights, // [s, k]
                    const paddle::Tensor &scatter_index,   // [k, s]
                    const paddle::Tensor &y_grad, // [num_experts * capacity, h]
                    const paddle::Tensor &combine_weights_out_grad, // [s, k]
                    paddle::Tensor &x_grad, 
                    paddle::Tensor &combine_weights_in_grad,
                    int64_t num_experts){
      int64_t num_rows = combine_weights.shape()[0];
      int64_t k = combine_weights.shape()[1];
#ifdef MOE_OPS_AUTO
      int64_t hidden_size = y_grad.shape()[2];
#else
      int64_t hidden_size = y_grad.shape()[1];
#endif
      int64_t num_active = y_grad.shape()[0];

      DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
        y_grad.type(),
        "apply_moe_dispatch_bwd",
        apply_moe_dispatch_bwd(
            y_grad.data<scalar_t_in>(),
            combine_weights.data<float>(),
            scatter_index.data<int>(),
            combine_weights_out_grad.data<float>(),
            combine_weights_in_grad.data<float>(),
            x_grad.data<scalar_t_in>(),
            num_rows,
            k,
            hidden_size,
            num_experts,
            num_active,
            y_grad.stream()));
}


std::vector<paddle::Tensor> MoEDispatchBwd(const paddle::Tensor &combine_weights_out, // [s, k]
                                            const paddle::Tensor &scatter_index,   // [k, s]
                                            const paddle::Tensor &scatter_index_rev,
                                            const paddle::Tensor &expert_offset, // [num_experts]
                                            const paddle::Tensor &expert_offset_local,
                                            const paddle::Tensor &y_grad, // [num_experts * capacity, h]
                                            const paddle::Tensor &combine_weights_out_grad, // [s, k]
                                            int64_t k,
                                            int64_t capacity,
                                            bool use_pad,
                                            int64_t expert_start_index, 
                                            int64_t expert_end_index
                                          ){
#ifdef DEBUG_MOE_OP
  std::cerr << "[DEBUG-BWD] "<<std::endl;
#endif  
  int64_t num_experts = expert_offset.shape()[0];
#ifdef MOE_OPS_AUTO
  // y_grad shape is [num_experts, capacity, h]
  int64_t hidden_size = y_grad.shape()[2];
#else
  int64_t hidden_size = y_grad.shape()[1];
#endif
  int64_t num_rows = scatter_index.shape()[1];
#ifdef DEBUG_MOE_OP
  std::cerr << "[DEBUG-BWD] num_experts=" << num_experts << " capacity="<< capacity << std::endl;
#endif  
  PD_CHECK(num_experts > 0);
  PD_CHECK(expert_offset.type() == paddle::DataType::INT64);
  if (use_pad){
    PD_CHECK(num_experts >= y_grad.shape()[0]/capacity);
  }else{
    PD_CHECK(y_grad.shape()[0] > 0);
  }

  auto place = y_grad.place();

  paddle::Tensor combine_weights_in_grad  =
      paddle::zeros(combine_weights_out_grad.shape(), paddle::DataType::FLOAT32, place); // [s, num_experts]

  paddle::Tensor x_grad = 
      paddle::empty({num_rows, hidden_size}, y_grad.type(), place); // [s, h]

  paddle::Tensor t_scatter_index = paddle::experimental::transpose(scatter_index, {1, 0});

  fleety_utils::TensorTrans2Contiguous(&t_scatter_index); 
  moe_dispatch_bwd(combine_weights_out,
                  t_scatter_index,
                  y_grad,
                  combine_weights_out_grad,
                  x_grad,
                  combine_weights_in_grad,
                  num_experts);

  return {x_grad, combine_weights_in_grad};
}

std::vector<std::vector<int64_t>> MoEDispatchBwdInferShape(
    std::vector<int64_t> combine_weights_shape,
    std::vector<int64_t> scatter_index_shape,
    std::vector<int64_t> expert_id_shape,
    std::vector<int64_t> y_grad_shape,
    std::vector<int64_t> combine_weights_grad_shape,
    int64_t k,
    int64_t capacity,
    bool use_pad)
{
  int64_t num_rows = scatter_index_shape[1];
  int64_t num_experts = y_grad_shape[0] / capacity;
  int64_t hidden_size = y_grad_shape[1];
  return {{num_rows, hidden_size}, {num_rows, num_experts}};
}

std::vector<paddle::DataType> MoEDispatchBwdInferDtype(
    paddle::DataType combine_weights_dtype,
    paddle::DataType scatter_index_dtype,
    paddle::DataType expert_id_dtype,
    paddle::DataType y_grad_dtype,
    paddle::DataType combine_weights_grad_dtype)
{
  return {y_grad_dtype, paddle::DataType::FLOAT32};
}


PD_BUILD_OP(moe_gate_dispatch_partial_nosoftmaxtopk)
    .Inputs({"x", "combine_weights", "expert_id"})
    .Outputs({"y", "combine_weights_out", "scatter_index", "scatter_index_rev", "expert_offset", "expert_offset_local"})
    .Attrs({"k: int64_t", 
          "capacity: int64_t", 
          "num_experts: int64_t", 
          "use_pad: bool", 
          "expert_start_index:int64_t", 
          "expert_end_index:int64_t", 
          "reverse_token_drop: bool"})
    .SetKernelFn(PD_KERNEL(MoEDispatchFwd))
    .SetInferShapeFn(PD_INFER_SHAPE(MoEDispatchFwdInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MoEDispatchFwdInferDtype));

PD_BUILD_GRAD_OP(moe_gate_dispatch_partial_nosoftmaxtopk)
  .Inputs({"combine_weights_out", "scatter_index", "scatter_index_rev", "expert_offset", "expert_offset_local", paddle::Grad("y"), paddle::Grad("combine_weights_out")})
  .Outputs({paddle::Grad("x"), paddle::Grad("combine_weights")})
  .Attrs({"k: int64_t", "capacity: int64_t", "use_pad: bool",  "expert_start_index:int64_t", "expert_end_index:int64_t"})
  .SetKernelFn(PD_KERNEL(MoEDispatchBwd))
  .SetInferShapeFn(PD_INFER_SHAPE(MoEDispatchBwdInferShape))
  .SetInferDtypeFn(PD_INFER_DTYPE(MoEDispatchBwdInferDtype));
