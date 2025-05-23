#pragma once
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/common/exception.h"
#include "paddle/phi/kernels/gpu/moe_kernel_impl.h"

template<typename T, int TPB>
__launch_bounds__(TPB) __global__ void moe_top_k(const T*    inputs_after_softmax,
                                                 const T*    bias, //bias could be nullptr if not used
                                                 T*          output,
                                                 int*        indices,
                                                 int*        source_rows,
                                                 const int   num_experts,
                                                 const int   k){
    using cub_kvp     = cub::KeyValuePair<int, T>;
    using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    cub_kvp     thread_kvp;
    cub::ArgMax arg_max;

    const int num_rows  = gridDim.x;
    const int block_row = blockIdx.x;
    const int  thread_read_offset = blockIdx.x * num_experts;
    for (int k_idx = 0; k_idx < k; ++k_idx) {
        thread_kvp.key   = 0;
        thread_kvp.value = T(-1.f);  // This is OK because inputs are probabilities

        cub_kvp inp_kvp;
        for (int expert = threadIdx.x; expert < num_experts; expert += TPB) {
            const int idx = thread_read_offset + expert;
            inp_kvp.key   = expert;
            inp_kvp.value = bias ? inputs_after_softmax[idx] + bias[expert] : inputs_after_softmax[idx] ;

            for (int prior_k = 0; prior_k < k_idx; ++prior_k) {
                const int prior_winning_expert = indices[k * block_row + prior_k];

                if (prior_winning_expert == expert) {
                    inp_kvp = thread_kvp;
                }
            }

            thread_kvp = arg_max(inp_kvp, thread_kvp);
        }

        const cub_kvp result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
        if (threadIdx.x == 0) {
            const int idx    = k * block_row + k_idx;
            output[idx]      = bias ? inputs_after_softmax[thread_read_offset + result_kvp.key]: result_kvp.value;
            indices[idx]     = result_kvp.key;
            source_rows[idx] = k_idx * num_rows + block_row;
        }
        __syncthreads();
    }
}

template<typename T>
void topk_gating_softmax_kernelLauncher(const T*     input,
                                        const T*     bias,
                                        T*           output,
                                        T*           softmax, //no use
                                        int*         indices,
                                        int*         source_row,
                                        const int    num_rows,
                                        const int    num_experts,
                                        const int    k,
                                        cudaStream_t stream){
    static constexpr int WARPS_PER_TB = 4;
    static constexpr int TPB = 256;
    moe_top_k<T, TPB><<<num_rows, TPB, 0, stream>>>(
        input, bias, output, indices, source_row, num_experts, k);    
}

template<typename T>
__global__ void modify_expert_id(const T*  expert_id,
                                T*         expert_id_out,
                                const int k,
                                const int num_rows,
                                const int64_t num_experts){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k * num_rows)
        return;    
    int ik = idx % k;
    int irow = idx / k;
    // const T mask = (~0) >> (8*sizeof(T)-ik); // 最后 ik 位为 1 其他位为 0
    int mask = ik; // k => 2(11)
    // printf("before: idx=%d, expert-id:%d, ik=%d\n", idx, expert_id[idx], ik);
    int offset = log2(k) + 1;
    expert_id_out[idx] = (expert_id[idx]<<offset) | mask;
    // printf("after: idx=%d, expert-id:%d, ik=%d\n", idx, expert_id_out[idx], ik);
}

template<typename T>
void modify_expert_id_launcher(const T* expert_id, 
        T* expert_id_out,
        const int k,
        const int num_rows,
        const int64_t num_experts,
        const cudaStream_t& stream){
    int max = 1024;
    const int threads = std::min(max, num_rows * k);
    const int blocks = (num_rows * k + threads - 1) / threads;
    
    modify_expert_id<T><<<blocks, threads, 0, stream>>>(
        expert_id, 
        expert_id_out,
        k, 
        num_rows,
        num_experts
    );
}

template<typename T>
__global__ void 
unmodify_expert_id(const T*  expert_id,
                                T*         expert_id_out,
                                const int k,
                                const int num_rows,
                                const int64_t num_experts){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k * num_rows)
        return;    
    int ik = idx % k;
    int irow = idx / k;
    int offset = log2(k) + 1;
    expert_id_out[idx] = (expert_id[idx]>>offset);
}

template<typename T>
void unmodify_expert_id_launcher(const T* expert_id, 
        T* expert_id_out,
        const int k,
        const int num_rows,
        const int64_t num_experts,
        const cudaStream_t& stream){
    int max = 1024;
    const int threads = std::min(max, num_rows * k);
    const int blocks = (num_rows * k + threads - 1) / threads;
    
    unmodify_expert_id<T><<<blocks, threads, 0, stream>>>(
        expert_id, 
        expert_id_out,
        k, 
        num_rows,
        num_experts
    );
}

template<typename T>
__device__ inline int find_total_elts_leq_target(const T* sorted_indices, const int arr_length, const int target)
{
    int64_t low = 0, high = arr_length - 1, target_location = -1;
    while (low <= high) {
        int64_t mid = (low + high) / 2;

        if (sorted_indices[mid] > target) {
            high = mid - 1;
        }
        else {
            low             = mid + 1;
            target_location = mid;
        }
    }
    return target_location + 1;
}

template<typename T>
__global__ void compute_total_rows_before_expert_kernel(const T*    sorted_experts,
                                                        const int     sorted_experts_len,
                                                        const int64_t num_experts,
                                                        int64_t*      total_rows_before_expert)
{

    // First, compute the global tid. We only need 1 thread per expert.
    const int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= num_experts)
        return;
    

    // This should construct the last index where each expert occurs.
    total_rows_before_expert[expert] = find_total_elts_leq_target<T>(sorted_experts, sorted_experts_len, expert);
    // total_rows_before_expert[0] = 0;
    // total_rows_before_expert[1] = 1;
    // if (sorted_experts_len > 3) {
    //     for (int i=0; i<35;i++){
    //         total_rows_before_expert[i] = i;
    //     }
    // }


}

template<typename T>
void compute_total_rows_before_expert(const T*   sorted_indices,
                                      const int    total_indices,
                                      const int64_t num_experts,
                                      int64_t*     total_rows_before_expert,
                                      const cudaStream_t& stream)
{
    const int threads = std::min(static_cast<int64_t>(1024), num_experts);
    const int blocks  = (num_experts + threads - 1) / threads;
    
    
    compute_total_rows_before_expert_kernel<T><<<blocks, threads, 0, stream>>>(
        sorted_indices, total_indices, num_experts, total_rows_before_expert);
}

template<typename T, int VecSize>
__global__ void initialize_moe_routing_kernel(const T*   unpermuted_input,
                                              T*         permuted_output,
                                              const int* expanded_dest_row_to_expanded_source_row,
                                              int*       expanded_source_row_to_expanded_dest_row,
                                              const int* permuted_experts,
                                              const int64_t* expert_offset,
                                              float* combine_weights, //output
                                              const int  num_rows,
                                              const int  cols,
                                              const int  k,
                                              const int64_t capacity,
                                              bool use_pad
                                              )
{

    // Reverse permutation map.
    // I do this so that later, we can use the source -> dest map to do the k-way reduction and unpermuting. I need the
    // reverse map for that reduction to allow each threadblock to do 1 k-way reduce without atomics later in MoE. 1
    // thread block will be responsible for all k summations.
    using LoadT = phi::AlignedVector<T, VecSize>;
    LoadT src_vec;    
    const int expanded_dest_row   = blockIdx.x;
    const int expanded_source_row = expanded_dest_row_to_expanded_source_row[expanded_dest_row];
    const int64_t iexpert = permuted_experts[expanded_dest_row];
    const int64_t offset = iexpert == 0 ? 0 : (expert_offset[iexpert - 1]);
    const int64_t row_in_expert = expanded_dest_row - offset;
    if (row_in_expert >= capacity){
        if (threadIdx.x == 0) {
            expanded_source_row_to_expanded_dest_row[expanded_source_row] = 0; // unset scatter-idx
            auto ik = expanded_source_row / num_rows;
            auto isent = expanded_source_row % num_rows; // transpose
            combine_weights[isent * k + ik] = 0.f; //unset combine-weight            
        }
        return;
    }
    int64_t num_padded = 0;
    if (threadIdx.x == 0) {
        // printf("going through: capacity=%lld, num_active=%lld, row=[%d->%d], row-in-expert %lld\n",
        //     capacity,
        //     num_active,
        //     expanded_dest_row, expanded_source_row,
        //     row_in_expert
        // );
        if (use_pad)
            num_padded = iexpert * capacity - offset;
        expanded_source_row_to_expanded_dest_row[expanded_source_row] = expanded_dest_row + num_padded;
    }
    // Duplicate and permute rows
    const int source_row = expanded_source_row % num_rows;

    const T* source_row_ptr = unpermuted_input + source_row * cols;
    T* dest_row_ptr;
    if (use_pad){
        dest_row_ptr = permuted_output + 
                       iexpert * capacity * cols + 
                       row_in_expert * cols;
    }else{
        dest_row_ptr = permuted_output + expanded_dest_row * cols;
    }


    for (int tid = threadIdx.x * VecSize; tid < cols; tid += blockDim.x* VecSize) {
        phi::Load<T, VecSize>(&source_row_ptr[tid], &src_vec);
        phi::Store<T, VecSize>(src_vec, &dest_row_ptr[tid]);
    }
}

template<typename T>
void initialize_moe_routing_kernelLauncher(const T*     unpermuted_input,
                                           T*           permuted_output,
                                           const int*   expanded_dest_row_to_expanded_source_row,
                                           int*         expanded_source_row_to_expanded_dest_row,
                                           const int*   permuted_experts,
                                           const int64_t* expert_offset,
                                           float* combine_weights, //output
                                           const int    num_rows,
                                           const int    cols,
                                           const int    k,
                                           const int64_t  capacity,
                                           bool use_pad,
                                           cudaStream_t stream)
{
    const int blocks  = num_rows * k;
    const int threads = std::min(cols, 1024);
    constexpr int max_pack_size = 16 / sizeof(T);
    if (cols % max_pack_size == 0) {
        initialize_moe_routing_kernel<T, max_pack_size><<<blocks, threads, 0, stream>>>(unpermuted_input,
                                                                    permuted_output,
                                                                    expanded_dest_row_to_expanded_source_row,
                                                                    expanded_source_row_to_expanded_dest_row,
                                                                    permuted_experts,
                                                                    expert_offset,
                                                                    combine_weights,                                                                    
                                                                    num_rows,
                                                                    cols,
                                                                    k,
                                                                    capacity,
                                                                    use_pad
                                                                    );
    } else {
        initialize_moe_routing_kernel<T, 1><<<blocks, threads, 0, stream>>>(unpermuted_input,
                                                                    permuted_output,
                                                                    expanded_dest_row_to_expanded_source_row,
                                                                    expanded_source_row_to_expanded_dest_row,
                                                                    permuted_experts,
                                                                    expert_offset,
                                                                    combine_weights,                                                                    
                                                                    num_rows,
                                                                    cols,
                                                                    k,
                                                                    capacity,
                                                                    use_pad
                                                                );
    }
}

/**
 * 原逻辑的output:
 * R0E0
 * R0E1
 * R1E0
 * R1E1
 * 
 * 我们想对all2all和专家gemm做overlap, 所以需要将all2all拆成流水线, 为了便于后续计算, 此kernel的output:
 * R0E0
 * R1E0
 * R0E1
 * R1E1
*/
template<typename T, int VecSize, int LoopSize>
__global__ void initialize_moe_routing_permute_kernel(const T*   unpermuted_input,
                                                        T*         permuted_output,
                                                        const int* expanded_dest_row_to_expanded_source_row,
                                                        int*       expanded_source_row_to_expanded_dest_row,
                                                        const int* permuted_experts,
                                                        const int64_t* expert_offset,
                                                        float* combine_weights, //output
                                                        const int  num_rows,
                                                        const int  cols,
                                                        const int  k,
                                                        const int64_t capacity,
                                                        const int64_t world_size,
                                                        const int64_t num_local_experts
                                              )
{
    // Reverse permutation map.
    // I do this so that later, we can use the source -> dest map to do the k-way reduction and unpermuting. I need the
    // reverse map for that reduction to allow each threadblock to do 1 k-way reduce without atomics later in MoE. 1
    // thread block will be responsible for all k summations.
#pragma unroll
    for (int i = 0; i < LoopSize; i++) {
        using LoadT = phi::AlignedVector<T, VecSize>;
        LoadT src_vec;    
        const int expanded_dest_row   = blockIdx.x + i * gridDim.x;
        const int expanded_source_row = expanded_dest_row_to_expanded_source_row[expanded_dest_row];
        const int64_t iexpert = permuted_experts[expanded_dest_row];
        const int64_t offset = iexpert == 0 ? 0 : (expert_offset[iexpert - 1]);
        const int64_t row_in_expert = expanded_dest_row - offset;
        if (row_in_expert >= capacity){
            if (threadIdx.x == 0) {
                expanded_source_row_to_expanded_dest_row[expanded_source_row] = 0; // unset scatter-idx
                auto ik = expanded_source_row / num_rows;
                auto isent = expanded_source_row % num_rows; // transpose
                combine_weights[isent * k + ik] = 0.f; //unset combine-weight            
            }
            continue;
        }
        int64_t num_padded = 0;
        if (threadIdx.x == 0) {
            num_padded = iexpert * capacity - offset;
            expanded_source_row_to_expanded_dest_row[expanded_source_row] = expanded_dest_row + num_padded;
        }
        // Duplicate and permute rows
        const int source_row = expanded_source_row % num_rows;

        const T* source_row_ptr = unpermuted_input + source_row * cols;
        T* dest_row_ptr;

        const int64_t irank = iexpert / num_local_experts;
        const int64_t local_iexpert = iexpert % num_local_experts;
        dest_row_ptr = permuted_output + local_iexpert * world_size * capacity * cols + irank * capacity * cols + row_in_expert * cols;

        for (int tid = threadIdx.x * VecSize; tid < cols; tid += blockDim.x * VecSize) {
            phi::Load<T, VecSize>(&source_row_ptr[tid], &src_vec);
            phi::Store<T, VecSize>(src_vec, &dest_row_ptr[tid]);
        }
    }
}

template<typename T>
void initialize_moe_routing_permute_kernelLauncher(const T*     unpermuted_input,
                                                    T*           permuted_output,
                                                    const int*   expanded_dest_row_to_expanded_source_row,
                                                    int*         expanded_source_row_to_expanded_dest_row,
                                                    const int*   permuted_experts,
                                                    const int64_t* expert_offset,
                                                    float* combine_weights, //output
                                                    const int    num_rows,
                                                    const int    cols,
                                                    const int    k,
                                                    const int64_t  capacity,
                                                    const int64_t world_size,
                                                    const int64_t  num_local_experts,
                                                    cudaStream_t stream)
{
    const int loop_size = 2;
    const int blocks  = (num_rows * k) / loop_size;
    assert((num_rows * k) % loop_size == 0);
    const int threads = std::min(cols, 1024);
    constexpr int max_pack_size = 16 / sizeof(T);
    if (cols % max_pack_size == 0) {
        initialize_moe_routing_permute_kernel<T, max_pack_size, loop_size><<<blocks, threads, 0, stream>>>(unpermuted_input,
                                                                    permuted_output,
                                                                    expanded_dest_row_to_expanded_source_row,
                                                                    expanded_source_row_to_expanded_dest_row,
                                                                    permuted_experts,
                                                                    expert_offset,
                                                                    combine_weights,                                                                    
                                                                    num_rows,
                                                                    cols,
                                                                    k,
                                                                    capacity,
                                                                    world_size,
                                                                    num_local_experts
                                                                    );
    } else {
        initialize_moe_routing_permute_kernel<T, 1, loop_size><<<blocks, threads, 0, stream>>>(unpermuted_input,
                                                                    permuted_output,
                                                                    expanded_dest_row_to_expanded_source_row,
                                                                    expanded_source_row_to_expanded_dest_row,
                                                                    permuted_experts,
                                                                    expert_offset,
                                                                    combine_weights,                                                                    
                                                                    num_rows,
                                                                    cols,
                                                                    k,
                                                                    capacity,
                                                                    world_size,
                                                                    num_local_experts
                                                                );
    }
}

