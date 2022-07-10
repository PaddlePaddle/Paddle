// // Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //     http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.

// #include <vector>
// #include "paddle/fluid/framework/tensor.h"
// #include "paddle/fluid/framework/tensor_util.h"
// #include "paddle/fluid/inference/tensorrt/plugin/fused_token_prune_op_plugin.h"
// #include "paddle/fluid/operators/fused_token_prune_op.cu.h"
// #include "paddle/fluid/platform/device_context.h"

// #include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
// #include "paddle/phi/kernels/funcs/math_function.h"

// namespace paddle {
// namespace inference {
// namespace tensorrt {
// namespace plugin {

// #if IS_TRT_VERSION_GE(6000)

// template <typename T>
// __global__ void ElementwiseMask(const T* a, const T* b, T* res, int num_raws,
//                                 int num_cols) {
//   auto tid = threadIdx.x + blockIdx.x * blockDim.x;
//   if (tid >= num_raws * num_cols) return;
//   const T zero = 0;
//   res[tid] = b[tid] >= zero ? a[tid] : zero;
// }

// template <typename T>
// __global__ void FillZero(T* data, int len) {
//   auto tid = threadIdx.x + blockIdx.x * blockDim.x;
//   if (tid >= len) return;
//   const T zero = 0;
//   data[tid] = zero;
// }

// __global__ void FillIndex(int* indices, int num_rows, int num_cols) {
//   int tid = threadIdx.x + blockIdx.x * blockDim.x;
//   if (tid >= num_rows * num_cols) return;

//   int col = tid % num_cols;
//   int row = tid / num_cols;

//   indices[tid] = col;
// }

// template <typename T>
// __global__ void ReduceSum2(const T* src, T* dst, int bsz, int nb_head,
//                            int max_seq_len) {
//   int tid = threadIdx.x;
//   int bid = blockIdx.x;
//   int batch = bid / (nb_head * max_seq_len);
//   int col = bid % max_seq_len;
//   int head = (bid / max_seq_len) % nb_head;

//   extern __shared__ T res_float[];
//   res_float[tid] =
//       src[batch * (nb_head * max_seq_len * max_seq_len) +
//           head * (max_seq_len * max_seq_len) + col + tid * max_seq_len];
//   __syncthreads();

//   for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
//     if (tid < offset) {
//       res_float[tid] += res_float[tid + offset];
//     }
//     __syncthreads();
//   }

//   if (tid == 0) {
//     auto* dst_addr = dst + batch * max_seq_len + col;
//     atomicAdd(dst_addr, res_float[0]);
//   }
// }

// template <>
// __global__ void ReduceSum2<half>(const half* src, half* dst, int bsz,
//                                  int nb_head, int max_seq_len) {
//   int tid = threadIdx.x;
//   int bid = blockIdx.x;
//   int batch = bid / (nb_head * max_seq_len);
//   int col = bid % max_seq_len;
//   int head = (bid / max_seq_len) % nb_head;
//   extern __shared__ half res_half[];
//   res_half[tid] =
//       src[batch * (nb_head * max_seq_len * max_seq_len) +
//           head * (max_seq_len * max_seq_len) + col + tid * max_seq_len];
//   __syncthreads();

//   for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
//     if (tid < offset) {
//       res_half[tid] += res_half[tid + offset];
//     }
//     __syncthreads();
//   }

//   if (tid == 0) {
//     platform::fastAtomicAdd<platform::float16>(
//         reinterpret_cast<platform::float16*>(dst),
//         static_cast<size_t>(batch * max_seq_len + col),
//         static_cast<size_t>(bsz * max_seq_len),
//         static_cast<platform::float16>(res_half[0]));
//   }
// }

// template <typename T>
// __global__ void SlicedArgsort(T* data, int* indices, int num_rows,
//                               int num_cols) {
//   auto raw = blockIdx.x * blockDim.x + threadIdx.x;
//   if (raw >= num_rows) return;
//   thrust::sort_by_key(thrust::seq, data + raw * num_cols + 1,
//                       data + (raw + 1) * num_cols, indices + raw * num_cols + 1,
//                       thrust::greater<T>());
// }

// template <typename T>
// __global__ void TakeAlongLastAxis2D(const T* src, T* dst, int* indices,
//                                     int num_rows, int src_num_cols,
//                                     int dst_num_cols, int num_elements) {
//   int tid = threadIdx.x + blockIdx.x * blockDim.x;
//   if (tid >= num_rows * dst_num_cols) return;

//   int raw = tid / dst_num_cols;
//   int col = tid % dst_num_cols;
//   for (int i = 0; i < num_elements; ++i) {
//     dst[tid * num_elements + i] =
//         *(src +
//           (raw * src_num_cols + indices[raw * src_num_cols + col]) *
//               num_elements +
//           i);
//   }
// }

// nvinfer1::DimsExprs FusedTokenPrunePluginDynamic::getOutputDimensions(
//     int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
//     nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT {
//   PADDLE_ENFORCE_EQ(
//       output_index, 0,
//       platform::errors::InvalidArgument(
//           "The FusedTokenPrune Plugin only has one output, so the "
//           "index value should be 0, but get %d.",
//           output_index));
//   auto x_dims = inputs[1], new_mask_dims = inputs[3];
//   nvinfer1::DimsExprs ret = x_dims;
//   ret.d[1] = new_mask_dims.d[2];
//   return ret;
// }

// bool FusedTokenPrunePluginDynamic::supportsFormatCombination(
//     int pos, const nvinfer1::PluginTensorDesc* in_out, int nb_inputs,
//     int nb_outputs) TRT_NOEXCEPT {
//   PADDLE_ENFORCE_NOT_NULL(
//       in_out, platform::errors::InvalidArgument(
//                   "The input of swish plugin shoule not be nullptr."));

//   PADDLE_ENFORCE_LT(
//       pos, nb_inputs + nb_outputs,
//       platform::errors::InvalidArgument("The pos(%d) should be less than the "
//                                         "num(%d) of the input and the output.",
//                                         pos, nb_inputs + nb_outputs));

//   const nvinfer1::PluginTensorDesc& in = in_out[pos];
//   if (pos < 4) {
//     if (with_fp16_) {
// #ifdef TRT_PLUGIN_FP16_AVALIABLE
//       return (in.type == nvinfer1::DataType::kFLOAT ||
//               in.type == nvinfer1::DataType::kHALF) &&
//              (in.format == nvinfer1::TensorFormat::kLINEAR);
// #else
//       return (in.type == nvinfer1::DataType::kFLOAT) &&
//              (in.format == nvinfer1::TensorFormat::kLINEAR);
// #endif
//     } else {
//       return (in.type == nvinfer1::DataType::kFLOAT) &&
//              (in.format == nvinfer1::TensorFormat::kLINEAR);
//     }
//   }
//   const nvinfer1::PluginTensorDesc& prev = in_out[pos - 1];

//   // output
//   return in.type == prev.type && in.format == prev.format;
// }

// nvinfer1::DataType FusedTokenPrunePluginDynamic::getOutputDataType(
//     int index, const nvinfer1::DataType* input_types,
//     int nb_inputs) const TRT_NOEXCEPT {
//   PADDLE_ENFORCE_EQ(
//       index, 0,
//       platform::errors::InvalidArgument(
//           "The EmbEltwiseLayernorm Plugin only has one output, so the "
//           "index value should be 0, but get %d.",
//           index));
//   return input_types[1];
// }

// template <typename T>
// int FusedTokenPrunePluginDynamic::enqueueImpl(
//     const nvinfer1::PluginTensorDesc* input_desc,
//     const nvinfer1::PluginTensorDesc* output_desc, const void* const* inputs,
//     void* const* outputs, void* workspace, cudaStream_t stream,
//     T* attn_tmp_data, T* attn_accu_data, int device_id) {
//   auto attn_dims = input_desc[0].dims;
//   auto x_dims = input_desc[1].dims;
//   auto new_mask_dims = input_desc[3].dims;
//   auto bsz = attn_dims.d[0], nb_head = attn_dims.d[1],
//        max_seq_len = attn_dims.d[2];
//   auto c = x_dims.d[2];

//   const T* attn_data = static_cast<const T*>(inputs[0]);
//   const T* x_data = static_cast<const T*>(inputs[1]);
//   const T* mask_data = static_cast<const T*>(inputs[2]);
//   T* output_data = static_cast<T*>(outputs[0]);
//   int total = bsz * nb_head * max_seq_len * max_seq_len;
//   int block = operators::ComputeBlockSize(max_seq_len);
//   int grid = operators::CeilDivide(total, block);

//   ElementwiseMask<T><<<grid, block, 0, stream>>>(
//       attn_data, mask_data, attn_tmp_data, grid, max_seq_len);

//   total = bsz * max_seq_len;
//   block = max_seq_len;
//   grid = operators::CeilDivide(total, block);
//   FillZero<T><<<grid, block, 0, stream>>>(attn_accu_data, total);

//   total = bsz * nb_head * max_seq_len * max_seq_len;
//   block = max_seq_len;
//   grid = operators::CeilDivide(total, block);
//   ReduceSum2<T><<<grid, block, block * sizeof(T), stream>>>(
//       attn_tmp_data, attn_accu_data, bsz, nb_head, max_seq_len);
//   framework::Tensor attn_accu_indices;
//   attn_accu_indices.Resize({bsz, max_seq_len});
//   auto* attn_accu_indices_data =
//       attn_accu_indices.mutable_data<int>(platform::CUDAPlace(device_id));

//   FillIndex<<<grid, block, 0, stream>>>(attn_accu_indices_data, bsz,
//                                         max_seq_len);

//   SlicedArgsort<T><<<bsz, 1, 0, stream>>>(
//       attn_accu_data, attn_accu_indices_data, bsz, max_seq_len);
//   int slimmed_x_len = new_mask_dims.d[2];
//   block = operators::ComputeBlockSize(slimmed_x_len);
//   TakeAlongLastAxis2D<T><<<grid, block, 0, stream>>>(
//       x_data, output_data, attn_accu_indices_data, bsz, max_seq_len,
//       slimmed_x_len, c);
//   return cudaGetLastError() != cudaSuccess;
// }

// int FusedTokenPrunePluginDynamic::enqueue(
//     const nvinfer1::PluginTensorDesc* input_desc,
//     const nvinfer1::PluginTensorDesc* output_desc, const void* const* inputs,
//     void* const* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT {
//   auto input_type = input_desc[0].type;
//   auto attn_dims = input_desc[0].dims;
//   auto bsz = attn_dims.d[0], nb_head = attn_dims.d[1],
//        max_seq_len = attn_dims.d[2];
//   int device_id;
//   cudaGetDevice(&device_id);

//   if (input_type == nvinfer1::DataType::kFLOAT) {
//     VLOG(1) << "TRT Plugin DataType selected. FusedTokenPrune-->fp32";

//     framework::Tensor attn_tmp;
//     attn_tmp.Resize({bsz, nb_head, max_seq_len, max_seq_len});
//     auto* attn_tmp_data =
//         attn_tmp.mutable_data<float>(platform::CUDAPlace(device_id));

//     framework::Tensor attn_accu;
//     attn_accu.Resize({bsz, max_seq_len});
//     auto* attn_accu_data =
//         attn_accu.mutable_data<float>(platform::CUDAPlace(device_id));

//     return enqueueImpl<float>(input_desc, output_desc, inputs, outputs,
//                               workspace, stream, attn_tmp_data, attn_accu_data,
//                               device_id);

//   } else if (input_type == nvinfer1::DataType::kHALF) {
// #ifdef TRT_PLUGIN_FP16_AVALIABLE
//     VLOG(1) << "TRT Plugin DataType selected. FusedTokenPrune-->fp16";

//     framework::Tensor attn_tmp;
//     attn_tmp.Resize({bsz, nb_head, max_seq_len, max_seq_len});
//     auto* attn_tmp_data_tmp = attn_tmp.mutable_data<int16_t>(
//         platform::CUDAPlace(device_id));  // NOLINT
//     auto* attn_tmp_data = reinterpret_cast<half*>(attn_tmp_data_tmp);

//     framework::Tensor attn_accu;
//     attn_accu.Resize({bsz, max_seq_len});
//     auto* attn_accu_data_tmp =
//         attn_accu.mutable_data<int16_t>(platform::CUDAPlace(device_id));
//     auto* attn_accu_data = reinterpret_cast<half*>(attn_accu_data_tmp);

//     return enqueueImpl<half>(input_desc, output_desc, inputs, outputs,
//                              workspace, stream, attn_tmp_data, attn_accu_data,
//                              device_id);

// #else
//     PADDLE_THROW(platform::errors::Fatal(
//         "The Ernie(Bert) TensorRT Plugin should be "
//         "complied with CUDA version >= 10.0 when running with fp16. "
//         "Please recomplie it or try to use fp32 by set "
//         "config.SetTRTDynamicShapeInfo(min_input_shape, "
//         "max_input_shape, opt_input_shape, true"));
// #endif
//   } else {
//     PADDLE_THROW(
//         platform::errors::Fatal("The FusedTokenPrune TRT Plugin's input type "
//                                 "should be float or half."));
//   }
// }

// #endif
// }  // namespace plugin
// }  // namespace tensorrt
// }  // namespace inference
// }  // namespace paddle
