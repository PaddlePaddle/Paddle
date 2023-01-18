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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/kernels/fusion/cutlass/fused_multi_head_attention/kernel_forward.h"

namespace phi {
namespace fusion {
namespace cutlass_internal {

struct LaunchParams {
    // meta params
    phi::DataType datatype; 

    // Input tensors
    const void* query_ptr; // [num_queries, num_heads, head_dim]
    const void* key_ptr; // [num_keys, num_heads, head_dim]
    const void* attn_bias_ptr = nullptr; // [num_heads, num_queries, num_keys]
    const void* value_ptr; // [num_keys, num_heads, head_dim_value]
    
    int32_t* cu_seqlens_q_ptr = nullptr;
    int32_t* cu_seqlens_k_ptr = nullptr;

    // Output tensors
    void* output_ptr; // [num_queries, num_heads, head_dim_value]
    void* output_accum_ptr; // [num_queries, num_heads, head_dim_value]
    void* logsumexp_ptr; // [num_heads, num_queries] - can be null

    // Scale
    // todo: 
    float scale;

    // Dimensions/strides
    int32_t num_batches;
    int32_t num_heads;
    int32_t query_seq_len; 
    int32_t key_value_seq_len; 
    int32_t head_size;
    int32_t value_head_size;
    bool causal;
    int32_t query_strideM;
    int32_t key_strideM;
    int32_t value_strideM;
    // Since bias can be broadcasted, we need to assign each stride 
    int32_t bias_strideM;
    int64_t bias_strideH;
    int64_t bias_strideB;
}; 

template<typename T, typename ArchTag, bool IsAligned, int QueriesPerBlock, int KeysPerBlock, bool SingleValueIteration> 
void LaunchMultiHeadAttentionKernel(LaunchParams params, cudaStream_t stream){
    using Attention = AttentionKernel<T, ArchTag, IsAligned, QueriesPerBlock, KeysPerBlock, SingleValueIteration>; 

    typename Attention::Params p;
    { // set parameters
      p.query_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.query_ptr));
      p.key_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.key_ptr));
      p.value_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.value_ptr));
      p.attn_bias_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.attn_bias_ptr));
      // TODO(zhengzekang): check this: 
      p.logsumexp_ptr = nullptr; // Only needed for bw

      p.output_accum_ptr = nullptr;
      // TODO(zhengzekang): Need to modify logic to use allocator
      if (Attention::kNeedsOutputAccumulatorBuffer) {
        // cudaMalloc(&p.output_accum_ptr, block_O.size() * sizeof(typename Attention::output_accum_t));
      }

      p.output_ptr = reinterpret_cast<T*>(params.output_ptr);

      // TODO: support arbitrary seq lengths
      // if (cu_seqlens_q.has_value()) {
      //   p.cu_seqlens_q_ptr = (int32_t*)cu_seqlens_q->data_ptr();
      //   p.cu_seqlens_k_ptr = (int32_t*)cu_seqlens_k->data_ptr();
      // }

      p.num_batches = params.num_batches;
      p.num_heads = params.num_heads;
      p.num_queries = params.query_seq_len;
      p.num_keys = params.key_value_seq_len;
      p.head_dim = params.head_size;
      p.head_dim_value = params.value_head_size;

      p.scale = params.scale; 
      p.causal = params.causal;

      // TODO: This might overflow for big tensors
      p.q_strideM = params.query_strideM;
      p.k_strideM = params.key_strideM;
      p.bias_strideM = params.bias_strideM;
      p.v_strideM = params.value_strideM;

      p.q_strideH = p.q_strideM * params.query_seq_len;
      p.k_strideH = p.k_strideM * params.key_value_seq_len;
      p.bias_strideH = params.bias_strideH;
      p.v_strideH = p.v_strideM * params.key_value_seq_len;
      p.o_strideH = params.value_head_size * params.query_seq_len;

      p.q_strideB = p.q_strideH * params.num_heads;
      p.k_strideB = p.k_strideH * params.num_heads;
      p.bias_strideB = params.bias_strideB;
      p.v_strideB = p.v_strideH * params.num_heads;
      p.o_strideB = params.value_head_size * params.query_seq_len * params.num_heads;
    }

    // launch kernel :)
    constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
    int smem_bytes = sizeof(typename Attention::SharedStorage);
    if (smem_bytes > 0xc000) {
      cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    if (!Attention::check_supported(p)) {
      // TODO(zhengzekang): use Paddle error
      printf("Wrong ======== \n"); 
      return; 
    //   std::cerr << "Kernel does not support these inputs" << std::endl;
    //   return result;
    }
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes, stream>>>(p);

}

// // TODO(zhengzekang) refine dispatch logic. 
void DispatchFusedMultiheadAttentionKernel(LaunchParams params, cudaStream_t stream){
    if(params.datatype == DataType::FLOAT32){
        // TODO(zhengzekang) check align, dispatch arch, dispatch query per blcok. 
        if (params.value_head_size > 64) {
            static int const kQueriesPerBlock = 32;
            static int const kKeysPerBlock = 128;
            if (params.value_head_size <= kKeysPerBlock) {
                LaunchMultiHeadAttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, kQueriesPerBlock, kKeysPerBlock, true>(params, stream); 
            } else {
                LaunchMultiHeadAttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, kQueriesPerBlock, kKeysPerBlock, false>(params, stream); 
            }
        } else {
            static int const kQueriesPerBlock = 64;
            static int const kKeysPerBlock = 64;
            LaunchMultiHeadAttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, kQueriesPerBlock, kKeysPerBlock, true>(params, stream); 
        }
    } else if (params.datatype == DataType::FLOAT16){
        printf("Enter here float16? \n"); 
        if (params.value_head_size > 64) {
            static int const kQueriesPerBlock = 32;
            static int const kKeysPerBlock = 128;
            if (params.value_head_size <= kKeysPerBlock) {
                LaunchMultiHeadAttentionKernel<cutlass::tfloat32_t, cutlass::arch::Sm80, true, kQueriesPerBlock, kKeysPerBlock, true>(params, stream); 
            } else {
                LaunchMultiHeadAttentionKernel<cutlass::tfloat32_t, cutlass::arch::Sm80, true, kQueriesPerBlock, kKeysPerBlock, false>(params, stream); 
            }
        } else {
            static int const kQueriesPerBlock = 64;
            static int const kKeysPerBlock = 64;
            LaunchMultiHeadAttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, kQueriesPerBlock, kKeysPerBlock, true>(params, stream); 
        }
    } else {
        // TODO(zhengzekang) raise error
        printf("===error===== \n"); 
    }
}

template <typename T, typename Context>
void MultiHeadAttentionForwardKernel(const Context& ctx,
                                     const DenseTensor& query,
                                     const DenseTensor& key,
                                     const DenseTensor& value,
                                    //  const paddle::optional<DenseTensor>& mask,
                                     const DenseTensor& mask, 
                                     const float scale, 
                                     const bool causal, 
                                     DenseTensor* output) {
    ctx.template Alloc<T>(output);
    LaunchParams params{}; 

    params.datatype = query.dtype(); 

    params.query_ptr = query.data(); 
    params.key_ptr = key.data(); 
    
    // TODO(zhengzekang): Check optional.  
    params.attn_bias_ptr = mask.data(); 
    // params.attn_bias_ptr = nullptr; 

    params.value_ptr = value.data(); 
    params.output_ptr = output->data();

    // TODO(zhengzekang): Use paddle allocator to refine.  
    params.output_accum_ptr = nullptr;

    // TODO(zhengzekang): currently we only used in inference. Maybe add a bool flag to save it ?
    params.logsumexp_ptr = nullptr;


    params.num_batches = query.dims()[0]; 
    params.num_heads = query.dims()[1]; 
    params.query_seq_len = query.dims()[2]; 
    params.key_value_seq_len = key.dims()[2]; 
    params.head_size = query.dims()[3]; 
    params.value_head_size = value.dims()[3];

    // TODO(zhengzekang): custom your qk scale. 
    float scale_value = sqrt(params.head_size); 
    if(scale != 0.0f){
        // assume 0.0f is default value. 
        scale_value = scale; 
    }
    params.scale = scale_value;  
    params.causal = causal; 

    params.query_strideM = query.dims()[3]; 
    params.key_strideM = key.dims()[3]; 
    params.value_strideM = value.dims()[3]; 

    // TODO(zhengzekang): fix it. 
    params.bias_strideM = mask.dims()[2] == 1 ? 0 : mask.dims()[3]; 
    params.bias_strideH = mask.dims()[1] == 1 ? 0 : params.bias_strideM * params.key_value_seq_len; 
    params.bias_strideB = mask.dims()[0] == 1 ? 0 : params.bias_strideH * params.num_heads; 

    printf("Bias stride M is: %d \n", int32_t(params.bias_strideM)); 
    printf("Bias stride H is: %d \n", int32_t(params.bias_strideH)); 
    printf("Bias stride B is: %d \n", int32_t(params.bias_strideB)); 

    // params.bias_strideM = 0; 
    // params.bias_strideH = 0; 
    // params.bias_strideB = 0; 

    auto stream = ctx.stream();
    DispatchFusedMultiheadAttentionKernel(params, stream); 
}
}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_multihead_attention,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::cutlass_internal::MultiHeadAttentionForwardKernel,
                   float,
                   phi::dtype::float16) {}
