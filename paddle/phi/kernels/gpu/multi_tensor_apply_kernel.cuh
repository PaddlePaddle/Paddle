#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/core/tensor_utils.h"
#ifdef PADDLE_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include <assert.h>

namespace phi {

#define MAX_CHUNK_SIZE 65535

typedef enum{
  ADAM_MODE_0   =0, // L2 regularization mode
  ADAM_MODE_1   =1  // Decoupled weight decay mode(AdamW)
} adamMode_t;

constexpr int depth_to_max_tensors[6] = {110, 64, 48, 36, 30, 24};
constexpr int depth_to_max_blocks[6] = {320, 320, 320, 320, 320, 320};

template<int n> struct TensorListMetadata
{
  const void* addresses[n][depth_to_max_tensors[n-1]];
  int sizes[depth_to_max_tensors[n-1]];
  unsigned char block_to_tensor[depth_to_max_blocks[n-1]];
  //int16
  unsigned short int block_to_chunk[depth_to_max_blocks[n-1]];
  int start_tensor_this_launch;
  int start_chunk_this_tensor;
};


template<typename MT, typename T, typename U, typename... ArgTypes>
__global__ void multi_tensor_apply_kernel(
    int chunk_size,
    T tl,
    U callable,
    ArgTypes... args)
{
  callable(chunk_size, tl, args...);
}

template<int depth, typename MT, typename FT, typename Context, typename... ArgTypes>
void multi_tensor_apply(
  const Context &dev_ctx,
  int block_size, //512
  int chunk_size, //2048*32
  const std::vector<std::vector<DenseTensor *>> &tensor_lists,
  const std::vector<const DenseTensor *> &g,
  FT callable,
  ArgTypes... args)
{
  PADDLE_ENFORCE_EQ(
        tensor_lists.size(),
        depth - 1,
        errors::InvalidArgument("ensor_lists.size() != depth - 1"));
  int len0 = tensor_lists[0].size();
  PADDLE_ENFORCE_GT(
      len0,
      0,
      errors::InvalidArgument(
          "tensor_lists[0].size() is not > 0"));
  auto ref_device = tensor_lists[0][0]->place();
  PADDLE_ENFORCE_NE(
        ref_device,
        CPUPlace(),
        errors::InvalidArgument("expected input to be on gpu"));
  for (int l = 0; l < tensor_lists.size(); l++) // No range-based for because I need indices
  {
    PADDLE_ENFORCE_EQ(
        tensor_lists[l].size(),
        len0,
        errors::InvalidArgument("Size mismatch among tensor lists"));
    for(int t = 0; t < tensor_lists[l].size(); t++)
    {
      PADDLE_ENFORCE_EQ(
        tensor_lists[l][t]->place(),
        ref_device,
        errors::InvalidArgument("A tensor was not on the same device as the first tensor"));
      PADDLE_ENFORCE_EQ(
        tensor_lists[l][t]->numel(),
        tensor_lists[0][t]->numel(),
        errors::InvalidArgument("The number of elements of Inputs msut be equal"));
    }
  }

  int ntensors = tensor_lists[0].size();

  TensorListMetadata<depth> tl;

  auto stream = dev_ctx.stream();
  tl.start_tensor_this_launch = 0;
  int loc_block_info = 0;
  int loc_tensor_info = 0;
  for(int t = 0; t < ntensors; t++)
  {
    tl.sizes[loc_tensor_info] = tensor_lists[0][t]->numel();
    tl.addresses[0][loc_tensor_info] = g[t]->data();
    for(int d = 1; d < depth; d++)
      tl.addresses[d][loc_tensor_info] = tensor_lists[d - 1][t]->data();
    loc_tensor_info++;
    int chunks_this_tensor = (tensor_lists[0][t]->numel() + chunk_size - 1)/chunk_size;
    tl.start_chunk_this_tensor = 0;
    int local_chunk = 0;

    cudaError_t error;

    for(int chunk = 0; chunk < chunks_this_tensor; chunk++)
    {
      tl.block_to_tensor[loc_block_info] = loc_tensor_info - 1;
      if(local_chunk > MAX_CHUNK_SIZE){
        tl.start_chunk_this_tensor += MAX_CHUNK_SIZE;
        local_chunk = 1;
      }
      tl.block_to_chunk[loc_block_info] = local_chunk;
      local_chunk++;
      loc_block_info++;
      bool tensors_full = (loc_tensor_info == depth_to_max_tensors[depth-1] &&
                           chunk == chunks_this_tensor - 1);
      bool blocks_full = (loc_block_info == depth_to_max_blocks[depth-1]);
      bool last_chunk = (t == ntensors - 1 && chunk == chunks_this_tensor - 1);
      if(tensors_full || blocks_full || last_chunk)
      {
        multi_tensor_apply_kernel<MT><<<loc_block_info, block_size, 0, stream>>>(
          chunk_size, //2048*32
          tl,
          callable,
          args...);
        
        loc_block_info = 0;
        if(chunk == chunks_this_tensor - 1)
        {
          loc_tensor_info = 0;
          tl.start_tensor_this_launch = t + 1;
        }
        else
        {
          tl.sizes[0] = tl.sizes[loc_tensor_info-1];
          for(int d = 0; d < depth; d++)
            tl.addresses[d][0] = tl.addresses[d][loc_tensor_info-1];
          loc_tensor_info = 1;
          tl.start_tensor_this_launch = t;
        }
      }
    }
  }
}

}  // namespace phi
