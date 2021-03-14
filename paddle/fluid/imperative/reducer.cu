// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/imperative/reducer.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace imperative {

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)

// template <typename T>
// __global__ void SpiltGroupToTensors(
//     const T* input_data,
//     T** output_data,
//     const int* out_len,
//     const int* global_used_var,
//     int total_len,
//     int out_num) {
//   CUDA_KERNEL_LOOP(i, total_len) {
//     int low = 0;
//     int high = out_num - 1;
//     while (low < high) {
//       int mid = (low + high) / 2;
//       if (i < out_len[mid]){
//         high = mid;
//       }else{
//         low = mid + 1;
//       }
//     }

//     int x = low;
//     int y = i - (x ? out_len[x - 1] : 0);

//     if(global_used_var[x] )

//     if (*(keys[x] + y) == 0) {
//       *(dest[x] + y * hidden) = 0;
//       *(dest[x] + y * hidden + 1) = 0;
//       *(dest[x] + y * hidden + 2) = 0;
//     } else {
//       *(dest[x] + y * hidden) = (src + i)->show;
//       *(dest[x] + y * hidden + 1) = (src + i)->clk;
//       *(dest[x] + y * hidden + 2) = (src + i)->embed_w;
//     }
//     if ((src + i)->embedding_size == 0 || *(keys[x] + y) == 0) {
//       for (int j = 0; j < hidden - 3; j++) {
//         *(dest[x] + y * hidden + 3 + j) = 0;
//       }
//     } else {
//       for (int j = 0; j < hidden - 3; j++) {
//         *(dest[x] + y * hidden + 3 + j) = (src + i)->embedx[1 + j];
//       }
//     }
//     // process embed_expand
//     if (expand_dim > 0) {
//       int z = x + slot_num;
//       if ((src + i)->embed_expand_size[0] == 0 || *(keys[x] + y) == 0) {
//         for (int j = 0; j < expand_dim; j++) {
//           *(dest[z] + y * expand_dim + j) = 0;
//         }
//       } else {
//         for (int j = 0; j < expand_dim; j++) {
//           *(dest[z] + y * expand_dim + j) = (src + i)->embed_expand[1 + j];
//         }
//       }
//     }
//   }  // end kernel loop
// }

void Group::DivNRanks(framework::Tensor *tensor, int nranks,
                      const platform::DeviceContext &context) {
  framework::VisitDataTypeSmall(
      dtype_, DivNRanksForAllReduce<platform::CUDADeviceContext>(tensor, nranks,
                                                                 context));
}

// According to a bitmap, choose whether to copy back the current tensor
// template <typename T>
// class SplitFunctorWithSelect<platform::CUDADeviceContext, T> {
//  public:
//   void operator()(const platform::CUDADeviceContext& context,
//                   const framework::Tensor& input,
//                   std::vector<framework::Tensor>* outputs,
//                   const framework::Tensor& global_used_var) {

//     int o_num = outputs->size();

//     std::vector<T*> outputs_data(o_num);
//     std::vector<int> outputs_length_lod(o_num);

//     outputs_length_lod[0] = outputs[0].numel();

//     for(int i = 1; i < o_num; ++i){
//         int o_numel = outputs[i].numel();
//         outputs_length_lod[i] = outputs_length_lod[i - 1] + o_numel;
//     }

//     // put outputs_length_lod to gpu
//     auto tmp_dev_ins_col_data = memory::Alloc(context,
//                                 outputs_length_lod.size() * sizeof(int));
//     memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, context.GetPlace()),
//                 tmp_dev_ins_col_data->ptr(), platform::CPUPlace(),
//                 reinterpret_cast<void*>(outputs_length_lod.data()),
//                 outputs_length_lod.size() * sizeof(int), context.stream());
//     int* dev_outs_len_lod =
//         reinterpret_cast<int*>(tmp_dev_ins_col_data->ptr());

//     const int *global_used_var_data = global_used_var.data<int>();

//     memory::allocation::AllocationPtr tmp_dev_outs_data;
//     T** dev_out_gpu_data = nullptr;
//     tmp_dev_outs_data =
//         memory::Alloc(context, outputs_data.size() * sizeof(T*));
//     memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, context.GetPlace()),
//                 tmp_dev_outs_data->ptr(), platform::CPUPlace(),
//                 reinterpret_cast<void*>(outputs_data.data()),
//                 outputs_data.size() * sizeof(T*), context.stream());
//     dev_out_gpu_data = reinterpret_cast<T**>(tmp_dev_outs_data->ptr());

//     SplitKernel<<<grid_dims, block_dims, 0, context.stream()>>>(
//         input.data<T>(), in_row, in_col, dev_outs_col_data,
//         static_cast<int>(outputs_length_lod.size()), dev_out_gpu_data);

//     // dim3 block_dims;
//     // dim3 grid_dims;
//     // GetBlockDims(context, out_row, in_col, &block_dims, &grid_dims);

//     int o_num = outputs->size();
//     int out_row = 1;
//     auto dim_0 = ref_inputs[0]->dims();
//     for (int i = 0; i < axis; ++i) {
//       out_row *= dim_0[i];
//     }

//     int out0_col = ref_inputs[0]->numel() / out_row;
//     int in_col = 0, in_row = out_row;
//     bool has_same_shape = true;

//     std::vector<T*> outputs_data(o_num);
//     std::vector<int> outputs_cols(o_num + 1);

//     outputs_cols[0] = 0;
//     for (int i = 0; i < o_num; ++i) {
//       int t_col = ref_inputs.at(i)->numel() / out_row;
//       if (has_same_shape) {
//         if (t_col != out0_col) has_same_shape = false;
//       }
//       in_col += t_col;
//       outputs_cols[i + 1] = in_col;
//       if (outputs->at(i) != nullptr) {
//         outputs_data[i] = outputs->at(i)->data<T>();
//       } else {
//         outputs_data[i] = nullptr;
//       }
//     }

//     dim3 block_dims;
//     dim3 grid_dims;
//     GetBlockDims(context, out_row, in_col, &block_dims, &grid_dims);

//     memory::allocation::AllocationPtr tmp_dev_outs_data;
//     T** dev_out_gpu_data = nullptr;
//     if (!has_same_shape || o_num < 2 || o_num > 4) {
//       tmp_dev_outs_data =
//           memory::Alloc(context, outputs_data.size() * sizeof(T*));
//       memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, context.GetPlace()),
//                    tmp_dev_outs_data->ptr(), platform::CPUPlace(),
//                    reinterpret_cast<void*>(outputs_data.data()),
//                    outputs_data.size() * sizeof(T*), context.stream());
//       dev_out_gpu_data = reinterpret_cast<T**>(tmp_dev_outs_data->ptr());
//     }

//     if (has_same_shape) {
//       if (o_num == 2) {
//         SplitKernel<<<grid_dims, block_dims, 0, context.stream()>>>(
//             input.data<T>(), in_row, in_col, out0_col, outputs_data[0],
//             outputs_data[1]);
//       } else if (o_num == 3) {
//         SplitKernel<<<grid_dims, block_dims, 0, context.stream()>>>(
//             input.data<T>(), in_row, in_col, out0_col, outputs_data[0],
//             outputs_data[1], outputs_data[2]);
//       } else if (o_num == 4) {
//         SplitKernel<<<grid_dims, block_dims, 0, context.stream()>>>(
//             input.data<T>(), in_row, in_col, out0_col, outputs_data[0],
//             outputs_data[1], outputs_data[2], outputs_data[3]);
//       } else {
//         SplitKernel<<<grid_dims, block_dims, 0, context.stream()>>>(
//             input.data<T>(), in_row, in_col, out0_col, dev_out_gpu_data);
//       }
//     } else {
//       auto tmp_dev_ins_col_data =
//           memory::Alloc(context,

//                         outputs_cols.size() * sizeof(int));
//       memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, context.GetPlace()),
//                    tmp_dev_ins_col_data->ptr(), platform::CPUPlace(),
//                    reinterpret_cast<void*>(outputs_cols.data()),
//                    outputs_cols.size() * sizeof(int), context.stream());
//       int* dev_outs_col_data =
//           reinterpret_cast<int*>(tmp_dev_ins_col_data->ptr());

//       SplitKernel<<<grid_dims, block_dims, 0, context.stream()>>>(
//           input.data<T>(), in_row, in_col, dev_outs_col_data,
//           static_cast<int>(outputs_cols.size()), dev_out_gpu_data);
//     }
//   }
// };

#endif

}  // namespace imperative
}  // namespace paddle
