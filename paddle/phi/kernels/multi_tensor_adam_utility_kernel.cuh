#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/core/tensor_utils.h"

#include <assert.h>

namespace phi {

#define MAX_COMPUTE_GROUP_SIZE 65535

constexpr int max_tensors_size[6] = {110, 64, 48, 36, 30, 24};
constexpr int max_blocks_size[6] = {320, 320, 320, 320, 320, 320};

template<int n> struct TensorAndBlockInf
{
  void* tensors_addr[n-1][max_tensors_size[n-1]];
  const void* grad[max_tensors_size[n-1]];
  int sizes[max_tensors_size[n-1]];
  unsigned char tenosr_for_this_block[max_blocks_size[n-1]];
  //int16
  unsigned short int compute_group_for_this_block[max_blocks_size[n-1]];
  int start_compute_group_this_tensor;
};


template<typename MT, typename T, typename U, typename... ArgTypes>
__global__ void multi_tensor_adam_utility_kernel(
    int compute_group_size,
    T tabi,
    U cuda_kernel_functor_op,
    ArgTypes... args)
{
  cuda_kernel_functor_op(compute_group_size, tabi, args...);
}

template<int input_num, typename MT, typename T, typename Context, typename... ArgTypes>
void  multi_tensor_adam_utility(
  const Context &dev_ctx,
  int block_size, //512
  int compute_group_size, //2048*32
  const std::vector<std::vector<DenseTensor *>> & tensor_and_block_inf,
  const std::vector<const DenseTensor *> &g,
  T cuda_kernel_functor_op,
  ArgTypes... args)
{
  PADDLE_ENFORCE_EQ(
         tensor_and_block_inf.size(),
        input_num - 1,
        errors::InvalidArgument("ensor_lists.size() != input_num - 1"));
  int length =  tensor_and_block_inf[0].size();
  PADDLE_ENFORCE_GT(
      length,
      0,
      errors::InvalidArgument(
          " tensor_and_block_inf[0].size() is not > 0"));
  auto place =  tensor_and_block_inf[0][0]->place();
  PADDLE_ENFORCE_NE(
        place,
        CPUPlace(),
        errors::InvalidArgument("expected input to be on gpu"));
  for (int i = 0; i <  tensor_and_block_inf.size(); i++) // No range-based for because I need indices
  {
    PADDLE_ENFORCE_EQ(
         tensor_and_block_inf[i].size(),
        length,
        errors::InvalidArgument("Size mismatch among tensor lists"));
    for(int j = 0; j <  tensor_and_block_inf[i].size(); j++)
    {
      PADDLE_ENFORCE_EQ(
         tensor_and_block_inf[i][j]->place(),
        place,
        errors::InvalidArgument("A tensor was not on the same device as the first tensor"));
      PADDLE_ENFORCE_EQ(
         tensor_and_block_inf[i][j]->numel(),
         tensor_and_block_inf[0][j]->numel(),
        errors::InvalidArgument("The number of elements of Inputs msut be equal"));
    }
  }

  int tensors_size =  tensor_and_block_inf[0].size();

  TensorAndBlockInf<input_num> tabi;

  auto stream = dev_ctx.stream();
  int block_id = 0;
  int tensor_id = 0;
  for(int t = 0; t < tensors_size; t++)
  {
    tabi.sizes[tensor_id] =  tensor_and_block_inf[0][t]->numel();
    tabi.grad[tensor_id] = g[t]->data();
    for(int d = 0; d < input_num - 1; d++)
      tabi.tensors_addr[d][tensor_id] =  tensor_and_block_inf[d][t]->data();
    tensor_id++;
    int compute_groups_this_tensor = ( tensor_and_block_inf[0][t]->numel() + compute_group_size - 1)/compute_group_size;
    tabi.start_compute_group_this_tensor = 0;
    int local_compute_group = 0;

    for(int compute_group = 0; compute_group < compute_groups_this_tensor; compute_group++)
    {
      tabi.tenosr_for_this_block[block_id] = tensor_id - 1;
      if(local_compute_group > MAX_COMPUTE_GROUP_SIZE){
        tabi.start_compute_group_this_tensor += MAX_COMPUTE_GROUP_SIZE;
        local_compute_group = 1;
      }
      tabi.compute_group_for_this_block[block_id] = local_compute_group;
      local_compute_group++;
      block_id++;
      bool reach_tesnors_limit = (tensor_id == max_tensors_size[input_num-1] &&
                           compute_group == compute_groups_this_tensor - 1);
      bool reach_blocks_limit = (block_id == max_blocks_size[input_num-1]);
      bool finish_compute = (t == tensors_size - 1 && compute_group == compute_groups_this_tensor - 1);
      if(reach_tesnors_limit || reach_blocks_limit || finish_compute)
      {
        multi_tensor_adam_utility_kernel<MT><<<block_id, block_size, 0, stream>>>(
          compute_group_size, //2048*32
          tabi,
          cuda_kernel_functor_op,
          args...);
        
        block_id = 0;
        if(compute_group == compute_groups_this_tensor - 1)
        {
          tensor_id = 0;
        }
        else
        {
          tabi.sizes[0] = tabi.sizes[tensor_id-1];
          tabi.grad[0] = tabi.grad[tensor_id-1];
          for(int d = 0; d < input_num - 1; d++)
            tabi.tensors_addr[d][0] = tabi.tensors_addr[d][tensor_id-1];
          tensor_id = 1;
        }
      }
    }
  }
}

}  // namespace phi
