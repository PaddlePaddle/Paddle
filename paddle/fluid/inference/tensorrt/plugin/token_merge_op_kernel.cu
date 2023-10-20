

#include "paddle/fluid/inference/tensorrt/plugin/token_merge_op_plugin.h"



#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include "paddle/phi/kernels/argsort_kernel.h"
#include "paddle/phi/kernels/matmul_kernel.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

template <typename T>
class tome_TypeTrait {
 public:
  using Type = T;
};

template <>
class tome_TypeTrait<half> {
 public:
  using Type = typename phi::dtype::float16;
};


/**
 * @brief  generate random indices to select dst_tokens
 *
 * @param rand_select_arr Randomly shuffled sequence of 0, 1, 3, and 5.
 * @param token_num token_number
 */

template<typename T>
__device__ void swap(T& a, T& b) {
    int temp = a;
    a = b;
    b = temp;
}

__global__ void token_merge_rand_select_kernel(int* rand_select_arr, int token_num, bool use_rand){

  int tid = blockIdx.x * blockDim.x + threadIdx.x;;
  int rand_arrry_begin = blockIdx.y * token_num +  tid * 4;
  if(tid < (token_num / SELECT_STRIDE)){
    rand_select_arr[rand_arrry_begin + 0] = 0;
    rand_select_arr[rand_arrry_begin + 1] = 1;
    rand_select_arr[rand_arrry_begin + 2] = 3;
    rand_select_arr[rand_arrry_begin + 3] = 5;
    if (use_rand == true){
      curandState localState;
      curand_init(clock64(), threadIdx.x, 0, &localState);
      for (int i = 3; i > 0; i--) {
        int randomIndex = curand(&localState) % (i + 1);
        swap(rand_select_arr[rand_arrry_begin + i], rand_select_arr[rand_arrry_begin + randomIndex]);
      }
    }
  }
}


/**
 * @brief  1.split origin token_tensor to src_token and dst_token, 
 *         2. get token's fro-norm and each element divisi that fro-norm
 *
 * @param origin_token_tensor origin token tensor
 * @param rand_select_arr rand select dst_token
 * @param rand_select select dst_token
 * @param hid_dim token's hidim
 * @param dst_token_num dst_token_number
 * @param width picture's width 

 * @return out_put_for_score for calculate the similarity score between src and dst 
 * @return rearranged_tensor origin token_tensor to src_token and dst_token 
 */
template <typename T>
__global__ void token_merge_split_kernel(
            const T* origin_token_tensor,
            int *rand_select_arr,
            int token_number,
            int hid_dim,
            int src_token_num,
            int dst_token_num, 
            int width,
            T* dst_tensor,
            T* src_tensor,
            T* dst_L2,
            T* src_L2) {
  typedef cub::WarpReduce<T> WarpReduce;
  int bid = gridDim.x * blockIdx.y + blockIdx.x; // bid in batch
  int block_num_pre_row = width / 16;
  //int bsz_begin_index = blockIdx.z * token_number * hid_dim;
  int bsz_begin_index = blockIdx.z * token_number;
  int block_begin_index = bsz_begin_index;
  if(bid % block_num_pre_row == 0)
    //block_begin_index = block_begin_index + (bid / block_num_pre_row) * 2 * width * hid_dim;
    block_begin_index = block_begin_index + (bid / block_num_pre_row) * 2 * width;
  else
    //block_begin_index = block_begin_index + ((bid / block_num_pre_row) * 2 * width + (bid % block_num_pre_row) * 16) * hid_dim;
    block_begin_index = block_begin_index + ((bid / block_num_pre_row) * 2 * width + (bid % block_num_pre_row) * 16);
  int tid = blockDim.x * threadIdx.y + threadIdx.x; // tid in block
  
  int wrap_id_in_block = tid / WARP_SIZE; //wrap_id in block，用来定位每个warp要处理的token的初始位置

  int thread_id_in_wrap = threadIdx.x; //
  int gloable_wrap_id = bid * 8 + wrap_id_in_block; // for write out_put  
  int rand_select_begin = blockIdx.z * token_number + gloable_wrap_id * 4;
  __shared__ T sum[8];

  //loop0
  int token0_id = block_begin_index +  wrap_id_in_block * 2;
  int token0_begin = token0_id * hid_dim;
  int data_index0 = token0_begin + thread_id_in_wrap;   //
  T squared_sum;
  __shared__ typename WarpReduce::TempStorage temp_storage_loop0;
  for(int cnt = 0, idx = data_index0; idx < token0_begin + hid_dim; idx += WARP_SIZE, cnt++){
     squared_sum = squared_sum + origin_token_tensor[idx] * origin_token_tensor[idx];
  } 
  T aggregate = WarpReduce(temp_storage_loop0).Sum(squared_sum);
  if(thread_id_in_wrap == 0)
    sum[wrap_id_in_block] = sqrtf(aggregate);
  __syncthreads();
  T aggregate0=static_cast<T>(1.0f)/sum[wrap_id_in_block];
  for(int cnt = 0, idx = data_index0; idx < token0_begin + hid_dim; idx += WARP_SIZE, cnt++){
    if(rand_select_arr[rand_select_begin] == 0){
      dst_L2[(dst_token_num * blockIdx.z + gloable_wrap_id) * hid_dim + threadIdx.x + cnt * WARP_SIZE] = origin_token_tensor[idx]*aggregate0;
      dst_tensor[(dst_token_num * blockIdx.z + gloable_wrap_id) * hid_dim + threadIdx.x + cnt * WARP_SIZE] = origin_token_tensor[idx];
    }else{
      src_L2[(src_token_num * blockIdx.z + gloable_wrap_id * 3 + rand_select_arr[rand_select_begin] / 2) * hid_dim + threadIdx.x + cnt * WARP_SIZE] = origin_token_tensor[idx]*aggregate0;
      src_tensor[(src_token_num * blockIdx.z + gloable_wrap_id * 3 + rand_select_arr[rand_select_begin] / 2) * hid_dim + threadIdx.x + cnt * WARP_SIZE] = origin_token_tensor[idx];
    }

  } 

  //loop1
  squared_sum =0.0f;
  int token1_begin = token0_begin + hid_dim;
  int data_index1 = data_index0 + hid_dim;
  __shared__ typename WarpReduce::TempStorage temp_storage_loop1;
  for(int cnt = 0, idx =  data_index1; idx < token1_begin + hid_dim; idx += WARP_SIZE, cnt++){
    squared_sum = squared_sum + origin_token_tensor[idx] * origin_token_tensor[idx];
  }
  aggregate = WarpReduce(temp_storage_loop1).Sum(squared_sum);
  if(thread_id_in_wrap == 0)
    sum[wrap_id_in_block] = sqrtf(aggregate);
  __syncthreads();
  T aggregate1 = static_cast<T>(1.0f)/sum[wrap_id_in_block];
  for(int cnt = 0, idx = data_index1; idx < token1_begin + hid_dim; idx += WARP_SIZE, cnt++){
   if(rand_select_arr[rand_select_begin + 1] == 0){
      dst_L2[(dst_token_num * blockIdx.z + gloable_wrap_id) * hid_dim + threadIdx.x + cnt * WARP_SIZE] = origin_token_tensor[idx]*aggregate1;
      dst_tensor[(dst_token_num * blockIdx.z + gloable_wrap_id) * hid_dim + threadIdx.x + cnt * WARP_SIZE] = origin_token_tensor[idx];
    }else{
      src_L2[(src_token_num * blockIdx.z + gloable_wrap_id * 3 + rand_select_arr[rand_select_begin + 1] / 2) * hid_dim + threadIdx.x + cnt * WARP_SIZE] = origin_token_tensor[idx]*aggregate1;
      src_tensor[(src_token_num * blockIdx.z + gloable_wrap_id * 3 + rand_select_arr[rand_select_begin + 1] / 2) * hid_dim + threadIdx.x + cnt * WARP_SIZE] = origin_token_tensor[idx];
    }
  }


  //loop2
  squared_sum = 0.0f;
  int token2_begin = token0_begin + hid_dim * width;
  int data_index2 = data_index0 + hid_dim * width;
  __shared__ typename WarpReduce::TempStorage temp_storage_loop2;
  for(int cnt = 0, idx = data_index2; idx < token2_begin + hid_dim; idx += WARP_SIZE, cnt++){
    squared_sum = squared_sum + origin_token_tensor[idx] * origin_token_tensor[idx];;
    
  }
  aggregate = WarpReduce(temp_storage_loop2).Sum(squared_sum);
  if(thread_id_in_wrap % 32 == 0)
    sum[wrap_id_in_block] = sqrtf(aggregate);
  __syncthreads();
  T aggregate2 = static_cast<T>(1.0f)/sum[wrap_id_in_block];
  for(int cnt = 0, idx = data_index2; idx < token2_begin + hid_dim; idx += WARP_SIZE, cnt++){
   if(rand_select_arr[rand_select_begin + 2] == 0){
      dst_L2[(dst_token_num * blockIdx.z + gloable_wrap_id) * hid_dim + threadIdx.x  +cnt * WARP_SIZE] = origin_token_tensor[idx]*aggregate2;
      dst_tensor[(dst_token_num * blockIdx.z + gloable_wrap_id) * hid_dim + threadIdx.x  + cnt * WARP_SIZE] = origin_token_tensor[idx];
    }else{
      src_L2[(src_token_num * blockIdx.z + gloable_wrap_id * 3 + rand_select_arr[rand_select_begin + 2] / 2) * hid_dim + threadIdx.x + cnt * WARP_SIZE] = origin_token_tensor[idx]*aggregate2;
      src_tensor[(src_token_num * blockIdx.z + gloable_wrap_id * 3 + rand_select_arr[rand_select_begin + 2] / 2) * hid_dim + threadIdx.x + cnt * WARP_SIZE] = origin_token_tensor[idx];
    }
  } 

  //loop3
  squared_sum =0.0f;
  int token3_begin = token2_begin + hid_dim;
  int data_index3 = data_index2 + hid_dim;
  __shared__ typename WarpReduce::TempStorage temp_storage_loop3;
  for(int cnt = 0, idx = data_index3; idx < token3_begin + hid_dim; idx += WARP_SIZE, cnt++){
    squared_sum = squared_sum + origin_token_tensor[idx] * origin_token_tensor[idx];;
  }
  aggregate = WarpReduce(temp_storage_loop3).Sum(squared_sum);
  if(thread_id_in_wrap % 32 == 0)
    sum[wrap_id_in_block] = sqrtf(aggregate);
  __syncthreads();
  T aggregate3 = static_cast<T>(1.0f)/sum[wrap_id_in_block];
  for(int cnt = 0, idx = data_index3; idx < token3_begin + hid_dim; idx += WARP_SIZE, cnt++){
   if(rand_select_arr[rand_select_begin + 3] == 0){
      dst_L2[(dst_token_num * blockIdx.z + gloable_wrap_id) * hid_dim + threadIdx.x  +cnt * WARP_SIZE] = origin_token_tensor[idx]*aggregate0;
      dst_tensor[(dst_token_num * blockIdx.z + gloable_wrap_id) * hid_dim + threadIdx.x  + cnt * WARP_SIZE] = origin_token_tensor[idx];
    }else{
      src_L2[(src_token_num * blockIdx.z + gloable_wrap_id * 3 + rand_select_arr[rand_select_begin + 3] / 2) * hid_dim + threadIdx.x  + cnt * WARP_SIZE] = origin_token_tensor[idx]*aggregate0;
      src_tensor[(src_token_num * blockIdx.z + gloable_wrap_id * 3 + rand_select_arr[rand_select_begin + 3] / 2) * hid_dim + threadIdx.x  + cnt * WARP_SIZE] = origin_token_tensor[idx];
    }
  } 
}

template <typename T>
struct maxScoreAndIndex
{
  T score;
  int idx;
  __device__ bool operator > (const maxScoreAndIndex &s) const {
    return score > s.score;
  }
};

/**
 * @brief Select the dst_token that is most similar to src_token by score
 *
 * @param input similarity_score
 * @param dst_token_num dst_token_number
 * @param  src_token_num src_token number

 * @return max_score  represent the similarity between each src_token and the it's most similar dst_token
 * @return max_score_index max_score_index[i] represent index of the most similar dst_token to i-th src_token
 */
template <typename T> 
__global__ void token_merge__max_similarity_kernel(T* input,
                  int src_token_num,
                  int dst_token_num, 
                  T* max_score, 
                  int* max_score_index) {
  typedef cub::WarpReduce<maxScoreAndIndex<T>> WarpReduce;
  int bsz_begin_index = blockIdx.y * dst_token_num * src_token_num;
  int bid = blockIdx.x;
  int tid =  blockDim.x * threadIdx.y + threadIdx.x;
  int gloable_warp_id = bid * 8 + tid / WARP_SIZE;
  maxScoreAndIndex<T> score_ = {INT_MIN, INT_MIN};
  int data_begin = bsz_begin_index + gloable_warp_id * dst_token_num + threadIdx.x;
  if(gloable_warp_id > src_token_num)return;
  for(int score_idx = data_begin; score_idx < data_begin + dst_token_num; score_idx += WARP_SIZE ){
    if (input[score_idx] > score_.score ){
      score_.score = input[score_idx];
      score_.idx = score_idx;
    }
  }
  __shared__ typename WarpReduce::TempStorage temp_storage;
  maxScoreAndIndex<T> aggregate = WarpReduce(temp_storage).Reduce(
      score_, cub::Max()); 
  __syncthreads();
  if (threadIdx.x == 0){
    max_score[blockIdx.y * src_token_num + gloable_warp_id] = aggregate.score;
    max_score_index[blockIdx.y * src_token_num + gloable_warp_id] = aggregate.idx % dst_token_num;
  }
}



/**
 * @brief merge src_token which need to be merged, one wrap process one needmerged src_token 
 *
 * @param input rearranged tensor
 * @param hid_dim hidim
 * @param dst_num dst_number
 * @param src_need_to_be_merged_idx The serial number of the token to be merged
 * @param src_token_need_merged_num how many src_token need to be merge
 * @param merged_to_dst_index merged_to_dst_index[i] indicates which dst_token should fused to if the i-th src_token needs to be fused
 * 
 * @return divied_rank The number of times dst_token[i] needs to be reduced
 * @return whether_tobe_merge indicate if token need to be merged
 */
template <typename T> 
__global__ void token_merge_reduce_kernel(T* dst_token, T* src_token, 
                                          int token_number, const int hid_dim, 
                                          int src_token_num, const int dst_token_num, 
                                          int* src_need_tobe_merged_idx, const int src_token_need_merged_num,
                                          int* merged_to_dst_index,  int* divied_rank, int* whether_tobe_merge){
  int bid = blockIdx.x;
  int tid =  blockDim.x * threadIdx.y + threadIdx.x;
  int gloable_warp_id = bid * 8 + tid / WARP_SIZE; 
  int dst_tensor_batch_begin = blockIdx.y * dst_token_num * hid_dim;
  int src_tensor_batch_begin = blockIdx.y * src_token_num * hid_dim;
  int divied_rank_batch_begin = blockIdx.y * dst_token_num;
  int src_need_tobe_merged_idx_batch_begin = blockIdx.y * src_token_num;
  int merged_to_dst_index_batch_begin = blockIdx.y * src_token_num;
  int whether_tobe_merge_batch_begin = blockIdx.y * src_token_num;
  if (gloable_warp_id < src_token_need_merged_num){
    int src_token_need_merged = src_need_tobe_merged_idx[src_need_tobe_merged_idx_batch_begin + gloable_warp_id];
    int dst_token_id_merged_to_in_batch = merged_to_dst_index[merged_to_dst_index_batch_begin + src_token_need_merged];
    int token_idx = src_tensor_batch_begin + src_token_need_merged * hid_dim ;
    for(int data_idx = token_idx + threadIdx.x, cnt = 0; data_idx < token_idx + hid_dim; data_idx += WARP_SIZE, cnt++ ){
      atomicAdd(&dst_token[dst_tensor_batch_begin + dst_token_id_merged_to_in_batch * hid_dim + threadIdx.x + cnt * WARP_SIZE], src_token[data_idx]);
    }
    if(threadIdx.x == 0){
      atomicAdd(&divied_rank[divied_rank_batch_begin + dst_token_id_merged_to_in_batch], 1);
      whether_tobe_merge[whether_tobe_merge_batch_begin + src_token_need_merged] = dst_token_id_merged_to_in_batch;
    }
  }
}

/**
 * @brief concat dst_token and remained src_toen
 *
 * @param input reduced tensor
 * @param src_token_number srctoken number
 * @param dst_token_num dsttoken number
 * @param hid_dim hidim
 * @param divide_rank 
 * 
 * @param src_need_to_be_merged_idx The serial number of the token to be merged
 * @param src_token_need_merged_num how many src_token need to be merged
 * @param divied_rank The number of times dst_token[i] needs to be reduced
 
 * @return output finianl res
 */
template <typename T> 
__global__ void token_merge_concat_kernel(T* dst_token, T* src_token, int token_num, int hid_dim, int src_token_num, int dst_token_num, int* divied_rank, 
                                          int* src_token_need_tobe_merged_idx, int final_token_num, int src_token_need_merged_num, 
                                          int* whether_tobe_merge,
                                          T* out_put){
  int bid = blockIdx.x;
  int tid =  blockDim.x * threadIdx.y + threadIdx.x;
  int dst_tensor_batch_begin = blockIdx.y * dst_token_num * hid_dim;
  int src_tensor_batch_begin = blockIdx.y * src_token_num * hid_dim;
  int out_put_batch_begin = blockIdx.y * final_token_num * hid_dim;
  int divied_rank_batch_begin = blockIdx.y * dst_token_num;
  int src_token_need_tobe_merged_idx_batch_begin = blockIdx.y * src_token_num;
  int gloable_warp_id = bid * 8 + tid / WARP_SIZE;
  if(gloable_warp_id > final_token_num)return;
  if (gloable_warp_id < dst_token_num){
      int token_begin_idx = dst_tensor_batch_begin + gloable_warp_id * hid_dim;
      for(int data_idx = token_begin_idx + threadIdx.x, cnt = 0; data_idx < token_begin_idx + hid_dim; data_idx += WARP_SIZE, cnt++ ){
        if (divied_rank[divied_rank_batch_begin + gloable_warp_id] > 0){
          out_put[out_put_batch_begin + gloable_warp_id * hid_dim + threadIdx.x + cnt * WARP_SIZE] = dst_token[data_idx] / static_cast<T>(divied_rank[divied_rank_batch_begin + gloable_warp_id] + 1);
        }else{
          out_put[out_put_batch_begin + gloable_warp_id * hid_dim + threadIdx.x + cnt * WARP_SIZE] = dst_token[data_idx];
        }
      }
  }else{
      int remained_token_id_in_batch = src_token_need_tobe_merged_idx[src_token_need_tobe_merged_idx_batch_begin + src_token_need_merged_num + (gloable_warp_id - dst_token_num)];
      int remained_src_token_begin_idx = src_tensor_batch_begin + remained_token_id_in_batch * hid_dim;
      for(int data_idx = remained_src_token_begin_idx + threadIdx.x, cnt = 0; data_idx < remained_src_token_begin_idx +  hid_dim; data_idx += WARP_SIZE, cnt++)
          out_put[out_put_batch_begin + gloable_warp_id * hid_dim + threadIdx.x + cnt * WARP_SIZE] = src_token[data_idx];
      if (threadIdx.x == 0)
      {
        int whether_tobe_merge_batch_begin = src_token_num * blockIdx.y;
        whether_tobe_merge[whether_tobe_merge_batch_begin + remained_token_id_in_batch] = gloable_warp_id;
      }
  }
}




template<typename T>
int32_t tokenMerge<T>::operator()(const phi::GPUContext &dev_ctx,
                   bool use_rand,
                   int bsz,
                   int token_number,
                   int src_token_number,
                   int dst_token_number,
                   int final_token_number,
                   int src_token_need_merged_num,
                   int hid_dim,
                   int height,
                   int width,
                   const T *origined_tensor,
                   phi::DenseTensor &src_token_tensor,
                   phi::DenseTensor &dst_token_tensor,
                   phi::DenseTensor &src_L2_tensor,
                   phi::DenseTensor &dst_L2_tensor,
                   phi::DenseTensor &similarity_tensor,
                   phi::DenseTensor &max_similarity_tensor,
                   phi::DenseTensor &max_similarity_idx_tensor,
                   phi::DenseTensor &argsort_res0_tensor,
                   phi::DenseTensor &argsort_res1_tensor,
                   phi::DenseTensor &divided_rank_tensor,
                   T *merged_tensor,
                   int *rand_select,
                   int *whether_to_be_merged){
  using PD_T = typename tome_TypeTrait<T>::Type;
  T* src_L2 = reinterpret_cast<T*>(src_L2_tensor.data<PD_T>());
  T* src_token = reinterpret_cast<T*>(src_token_tensor.data<PD_T>());

  T* dst_L2 = reinterpret_cast<T*>(dst_L2_tensor.data<PD_T>());
  T* dst_token = reinterpret_cast<T*>(dst_token_tensor.data<PD_T>());

  T* similarity = reinterpret_cast<T*>(similarity_tensor.data<PD_T>());
  T* max_similarity =reinterpret_cast<T*>(max_similarity_tensor.data<PD_T>());
  int* max_similarity_idx = max_similarity_idx_tensor.data<int>();
  int* divied_rank = divided_rank_tensor.data<int>();
  //0: gen randomly select array，and init whether_tobe_merge_arrry
  dim3 grid_dim_for_rand_select(token_number / (4 * WARP_SIZE), bsz);
  dim3 block_dim_for_rand_select(WARP_SIZE);
  token_merge_rand_select_kernel<<<grid_dim_for_rand_select, block_dim_for_rand_select, 0, dev_ctx.stream()>>>(rand_select, token_number, use_rand);
    
  //1.split origined_tensor
  dim3 grid_dim_for_split(height / 16, width / 2, bsz);
  dim3 block_dim_for_split(WARP_SIZE, 8);
  token_merge_split_kernel<<<grid_dim_for_split, block_dim_for_split>>>(
    origined_tensor,
    rand_select,
    token_number,
    hid_dim,
    dst_token_number,
    src_token_number,
    width,
    dst_token,
    src_token,
    dst_L2,
    src_L2);

  //step2: calculate the similarity between each src_token and each dst_token
  phi::MatmulKernel<typename tome_TypeTrait<T>::Type, phi::GPUContext>(dev_ctx,
                                        src_L2_tensor,
                                        dst_L2_tensor,
                                        false,
                                        true,
                                        &similarity_tensor);
  

  //step3: get dst_token that is most similar to src_token
  dim3 grid_for_max_similarity(src_token_number / 8, bsz);
  dim3 block_for_max_similarity(32, 8);
  token_merge__max_similarity_kernel<<<grid_for_max_similarity, block_for_max_similarity>>>(similarity,
                                                                                            src_token_number,
                                                                                            dst_token_number,
                                                                                            max_similarity,
                                                                                            max_similarity_idx);
  

  //step4: argsort by echo src_token's max similarity
  phi::ArgsortKernel<typename tome_TypeTrait<T>::Type, phi::GPUContext>(dev_ctx,
                                        max_similarity_tensor,
                                        -1,
                                        false,
                                        &argsort_res0_tensor,
                                        &argsort_res1_tensor);

  //step5: do reduce
  auto src_token_need_tobe_merged_id = argsort_res1_tensor.data<int>();
  dim3 grid_dim_for_reduce(src_token_number / 8, bsz);
  dim3 block_dim_for_reduce(32, 8);
  token_merge_reduce_kernel<<<grid_dim_for_reduce, block_dim_for_reduce>>>(
    dst_token,
    src_token,
    token_number,
    hid_dim,
    src_token_number,
    dst_token_number,
    src_token_need_tobe_merged_id,
    src_token_need_merged_num,
    max_similarity_idx,
    divied_rank,
    whether_to_be_merged);

  dim3 grid_dim_for_concat(final_token_number / 8, bsz);
  dim3 block_dim_for_concat(32, 8);
  token_merge_concat_kernel<<<grid_dim_for_concat, block_dim_for_concat>>>(
    dst_token,
    src_token,
    token_number,
    hid_dim,
    src_token_number,
    dst_token_number,
    divied_rank,
    src_token_need_tobe_merged_id,
    final_token_number,
    src_token_need_merged_num,
    whether_to_be_merged,
    merged_tensor);
  return cudaGetLastError() != cudaSuccess;
}
template struct tokenMerge<float>;
template struct tokenMerge<half>;
}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
