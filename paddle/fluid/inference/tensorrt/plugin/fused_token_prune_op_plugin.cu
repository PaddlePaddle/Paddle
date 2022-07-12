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

#include <vector>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/inference/tensorrt/plugin/fused_token_prune_op_plugin.h"
#include "paddle/fluid/operators/fused_token_prune_op.cu.h"
#include "paddle/fluid/platform/device_context.h"

#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

#if IS_TRT_VERSION_GE(6000)

template <typename T>
__global__ void ElementwiseMask(const T* a, const T* b, T* res, int num_elements) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_elements) return;
  const T zero = 0;
  res[tid] = b[tid] >= zero ? a[tid] : zero;
}

template <typename T>
__global__ void FillZero(T* data, int len) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= len) return;
  const T zero = 0;
  data[tid] = zero;
}

__global__ void FillIndex(int32_t* indices, int num_raws, int num_cols) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_raws * num_cols) return;

  int col = tid % num_cols;
  int raw = tid / num_cols;

  indices[tid] = col;
}

template <typename T>
__global__ void ReduceSum2(const T* src, T* dst, int bsz, int nb_head,
                           int max_seq_len) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int num_blocks_per_head = ((max_seq_len / blockDim.x) * max_seq_len);
  int batch = bid / (nb_head * num_blocks_per_head);
  int col = bid % max_seq_len;
  int head = (bid / num_blocks_per_head) % nb_head;

  extern __shared__ T res_float[];
  res_float[tid] =
      src[batch * (nb_head * max_seq_len * max_seq_len) +
          head * (max_seq_len * max_seq_len) + col + tid * max_seq_len];
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      res_float[tid] += res_float[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    auto* dst_addr = dst + batch * max_seq_len + col;
    atomicAdd(dst_addr, res_float[0]);
  }
}

template <>
__global__ void ReduceSum2<half>(const half* src, half* dst, int bsz,
                                 int nb_head, int max_seq_len) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int num_blocks_per_head = ((max_seq_len / blockDim.x) * max_seq_len);
  int batch = bid / (nb_head * num_blocks_per_head);
  int col = bid % max_seq_len;
  int head = (bid / num_blocks_per_head) % nb_head;
  
  extern __shared__ half res_half[];
  res_half[tid] =
      src[batch * (nb_head * max_seq_len * max_seq_len) +
          head * (max_seq_len * max_seq_len) + col + tid * max_seq_len];
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      res_half[tid] += res_half[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    platform::fastAtomicAdd<platform::float16>(
        reinterpret_cast<platform::float16*>(dst),
        static_cast<size_t>(batch * max_seq_len + col),
        static_cast<size_t>(bsz * max_seq_len),
        static_cast<platform::float16>(res_half[0]));
  }
}

// template <typename T>
// __global__ void SlicedArgsort(T* data, int* indices, int num_raws,
//                               int num_cols) {
//   auto raw = blockIdx.x * blockDim.x + threadIdx.x;
//   if (raw >= num_raws) return;
//   thrust::sort_by_key(thrust::seq, data + raw * num_cols + 1,
//                       data + (raw + 1) * num_cols, indices + raw * num_cols + 1,
//                       thrust::greater<T>());
// }

// template <typename T>
// __global__ void TakeAlongLastAxis2D(const T* src, T* dst, int* indices,
//                                     int num_raws, int src_num_cols,
//                                     int dst_num_cols, int num_elements) {
//   int tid = threadIdx.x + blockIdx.x * blockDim.x;
//   if (tid >= num_raws * dst_num_cols) return;

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

template <typename T>
__global__ void Slice(const T* src, T* dst, int num_raws,
    int src_num_cols, int dst_num_cols) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_raws * dst_num_cols) return;
    int raw = tid / dst_num_cols;
    int col = tid % dst_num_cols;
    dst[tid] = src[raw * src_num_cols + col];
    
}

template <typename T>
__global__ void TakeAlongAxis(const T* src, T* dst, int32_t* indices, int num_raws,
                              int src_num_cols, int dst_num_cols,
                              int num_elements) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_raws * dst_num_cols) return;

  int raw = tid / dst_num_cols;
  int col = tid % dst_num_cols;
  for (int i = 0; i < num_elements; ++i) {
    dst[tid * num_elements + i] =
        *(src + (raw * src_num_cols + (int)indices[tid]) * num_elements + i);
  }
}

template <typename T>
__global__ void MaximumFirst(T* mat, int num_raws, int num_cols, T max_value) {
  auto raw = blockIdx.x * blockDim.x + threadIdx.x;
  if (raw >= num_raws) return;
  mat[raw * num_cols] = max_value;
}

__global__ void FillOffsets(int* offsets, int num_raws, int num_cols) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid > num_raws) return;

  offsets[tid] = tid * num_cols;
}


nvinfer1::DimsExprs FusedTokenPrunePluginDynamic::getOutputDimensions(
    int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT {
  
  auto x_dims = inputs[1], new_mask_dims = inputs[3];
  if (output_index == 0) {
    nvinfer1::DimsExprs ret = x_dims;
    ret.d[1] = new_mask_dims.d[2];
    return ret;
  } else {
    nvinfer1::DimsExprs ret;
    ret.nbDims = 2;
    ret.d[0] = new_mask_dims.d[0];
    ret.d[1] = new_mask_dims.d[2];
    return ret;
  }
}

bool FusedTokenPrunePluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* in_out, int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_NOT_NULL(
      in_out, platform::errors::InvalidArgument(
                  "The input of swish plugin shoule not be nullptr."));

  PADDLE_ENFORCE_LT(
      pos, nb_inputs + nb_outputs,
      platform::errors::InvalidArgument("The pos(%d) should be less than the "
                                        "num(%d) of the input and the output.",
                                        pos, nb_inputs + nb_outputs));

  const nvinfer1::PluginTensorDesc& in = in_out[pos];
  if (pos < 4) {
    if (with_fp16_) {
#ifdef TRT_PLUGIN_FP16_AVALIABLE
      return (in.type == nvinfer1::DataType::kFLOAT ||
              in.type == nvinfer1::DataType::kHALF) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
#else
      return (in.type == nvinfer1::DataType::kFLOAT) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
#endif
    } else {
      return (in.type == nvinfer1::DataType::kFLOAT) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
    }
  } else if (pos == 4) {
    const nvinfer1::PluginTensorDesc& prev = in_out[pos - 1];
    return in.type == prev.type && in.format == prev.format;
  } else {
    const nvinfer1::PluginTensorDesc& prev = in_out[pos - 1];
    return in.type == nvinfer1::DataType::kINT32 && in.format == prev.format;
  }
}

nvinfer1::DataType FusedTokenPrunePluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType* input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  if (index == 0) {
    return input_types[1];
  } else if (index == 1) {
    return nvinfer1::DataType::kINT32;
  }
}

size_t FusedTokenPrunePluginDynamic::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
    int nb_inputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int nb_outputs) const TRT_NOEXCEPT {
        auto attn_dims = inputs[0].dims;
        auto x_dims = inputs[1].dims;
        auto new_mask_dims = inputs[3].dims;
        auto bsz = attn_dims.d[0], nb_head = attn_dims.d[1],
            max_seq_len = attn_dims.d[2];
        
        int slimmed_x_len = new_mask_dims.d[2];
        int total = bsz * nb_head * max_seq_len * max_seq_len;
        size_t size = total * sizeof(float);
        size += bsz * max_seq_len * sizeof(float);
        size += bsz * max_seq_len * sizeof(int32_t);
        size += bsz * max_seq_len * sizeof(float);
        size += bsz * max_seq_len * sizeof(int32_t);
        size += (bsz+1) * sizeof(int);
        size += bsz * slimmed_x_len * sizeof(int32_t);
        size += bsz * slimmed_x_len * sizeof(int32_t);
        return size;

}

template <typename T>
int FusedTokenPrunePluginDynamic::enqueueImpl(
    const nvinfer1::PluginTensorDesc* input_desc,
    const nvinfer1::PluginTensorDesc* output_desc, const void* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream/*,
T* attn_tmp_data, T* attn_accu_data*/, int device_id, T max_value) {
  auto attn_dims = input_desc[0].dims;
  auto x_dims = input_desc[1].dims;
  auto new_mask_dims = input_desc[3].dims;
  auto bsz = attn_dims.d[0], nb_head = attn_dims.d[1],
       max_seq_len = attn_dims.d[2];
  auto c = x_dims.d[2];
  int slimmed_x_len = new_mask_dims.d[2];

  const T* attn_data = static_cast<const T*>(inputs[0]);
  const T* x_data = static_cast<const T*>(inputs[1]);
  const T* mask_data = static_cast<const T*>(inputs[2]);
  T* output_data = static_cast<T*>(outputs[0]);
  int32_t* output_indices_data = static_cast<int32_t*>(outputs[1]);
  

  int total = bsz * nb_head * max_seq_len * max_seq_len;
  int block = operators::ComputeBlockSize(max_seq_len);
  int grid = operators::CeilDivide(total, block);

  T* attn_tmp_data = static_cast<T*>(workspace);
  size_t offset = total * sizeof(T);
  T* attn_accu_data = static_cast<T*>(workspace + offset);
  offset += bsz * max_seq_len * sizeof(T);
  int32_t* attn_accu_indices_data = static_cast<int32_t*>(workspace + offset);
  offset += bsz * max_seq_len * sizeof(int32_t);
  T* sort_attn_accu_data = static_cast<T*>(workspace + offset);
  offset += bsz * max_seq_len * sizeof(T);
  int32_t* sort_attn_accu_indices_data = static_cast<int32_t*>(workspace + offset);
  offset += bsz * max_seq_len * sizeof(int32_t);
  int* offsets_data = static_cast<int*>(workspace + offset);
  offset += (bsz+1) * sizeof(int);
  int32_t* slimmed_sort_attn_accu_indices_data = static_cast<int32_t*>(workspace + offset);
  offset += bsz * slimmed_x_len * sizeof(int32_t);
  int32_t* slimmed_resort_attn_accu_indices_data = static_cast<int32_t*>(workspace + offset);

    //debugggg
    if (max_seq_len < 10) {
      std::vector<T> attn_data_h(bsz * nb_head * max_seq_len * max_seq_len);
      cudaMemcpy(attn_data_h.data(), attn_data, bsz * nb_head * max_seq_len * max_seq_len*sizeof(T), cudaMemcpyDeviceToHost);
      VLOG(1) << "attn_data_h";
      for (auto k : attn_data_h) std::cout << static_cast<float>(k) << " ";
      std::cout << std::endl;
    }
  
    //end debug 

  ElementwiseMask<T><<<grid, block, 0, stream>>>(
      attn_data, mask_data, attn_tmp_data, total);

  total = bsz * max_seq_len;
  block = max_seq_len;
  grid = operators::CeilDivide(total, block);
  FillZero<T><<<grid, block, 0, stream>>>(attn_accu_data, total);

   //debugggg
   if (max_seq_len < 10) {
    std::vector<T> attn_tmp_data_h(bsz * nb_head * max_seq_len * max_seq_len);
    cudaMemcpy(attn_tmp_data_h.data(), attn_tmp_data, bsz * nb_head * max_seq_len * max_seq_len*sizeof(T), cudaMemcpyDeviceToHost);
    VLOG(1) << "attn_tmp_data_h";
    for (auto k : attn_tmp_data_h) std::cout << static_cast<float>(k) << " ";
    std::cout << std::endl;
   }
 
   //end debug 
  total = bsz * nb_head * max_seq_len * max_seq_len;
  int block_tmp = max_seq_len;
  while (block_tmp > 1024) block_tmp /= 2; // if max seq len > 1024, it must be 2^n
  block = block_tmp;// make sure max_seq_len is an integral multiple of block
  grid = operators::CeilDivide(total, block);
  ReduceSum2<T><<<grid, block, block * sizeof(T), stream>>>(
      attn_tmp_data, attn_accu_data, bsz, nb_head, max_seq_len);

  total = bsz * max_seq_len;
  block = operators::ComputeBlockSize(max_seq_len);
  grid = operators::CeilDivide(total, block);

  FillIndex<<<grid, block, 0, stream>>>(attn_accu_indices_data, bsz,
                                        max_seq_len);

//   SlicedArgsort<T><<<bsz, 1, 0, stream>>>(
//       attn_accu_data, attn_accu_indices_data, bsz, max_seq_len);

  
  size_t temp_storage_bytes = -1;

  if (keep_first_token_) {
    VLOG(1) << "keep_first_token_";
    // T max = std::numeric_limits<T>::max();
    MaximumFirst<T><<<bsz, 1, 0, stream>>>(attn_accu_data, bsz, max_seq_len, max_value);
  }
   //debugggg
   if (max_seq_len < 10) {

    std::vector<T> attn_accu_data_h(bsz*max_seq_len);
    cudaMemcpy(attn_accu_data_h.data(), attn_accu_data, bsz*max_seq_len*sizeof(T), cudaMemcpyDeviceToHost);
    VLOG(1) << "attn_accu_data_h";
    for (auto k : attn_accu_data_h) std::cout << static_cast<float>(k) << " ";
    std::cout << std::endl;
   }
 
   //end debug 
  int num_items = bsz * max_seq_len;
  int num_segments = bsz;
  FillOffsets<<<bsz+1, 1, 0, stream>>>(offsets_data, bsz, max_seq_len);

  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr,
    temp_storage_bytes,
    attn_accu_data,
    sort_attn_accu_data,
    attn_accu_indices_data,
    sort_attn_accu_indices_data,
    num_items,
    num_segments,
    offsets_data,
    offsets_data+1,
    0,
    sizeof(T) * 8,
    stream));
  
  int64_t temp_size = temp_storage_bytes;
  framework::Tensor temp_storage;
  auto* temp_storage_data = temp_storage.mutable_data<uint8_t>({temp_size}, platform::CUDAPlace(device_id));

  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceSegmentedRadixSort::SortPairsDescending(temp_storage_data,
    temp_storage_bytes,
    attn_accu_data,
    sort_attn_accu_data,
    attn_accu_indices_data,
    sort_attn_accu_indices_data,
    num_items,
    num_segments,
    offsets_data,
    offsets_data+1,
    0,
    sizeof(T) * 8,
    stream));

  //debugggg
  if (max_seq_len < 10) {

    std::vector<int32_t> sort_attn_accu_indices_data_h(bsz*max_seq_len);
    cudaMemcpy(sort_attn_accu_indices_data_h.data(), sort_attn_accu_indices_data, bsz*max_seq_len*sizeof(int32_t), cudaMemcpyDeviceToHost);
    VLOG(1) << "sort_attn_accu_indices_data_h";
    for (auto k : sort_attn_accu_indices_data_h) std::cout << k << " ";
    std::cout << std::endl;
  }

  //end debug

  total = bsz * slimmed_x_len;
  block = operators::ComputeBlockSize(slimmed_x_len);
  grid = operators::CeilDivide(total, block);

//   Tensor slimmed_sort_attn_accu_indices;
//   slimmed_sort_attn_accu_indices.Resize(sort_attn_accu.dims());
//   int32_t* slimmed_sort_attn_accu_indices_data =
//         slimmed_sort_attn_accu_indices.mutable_data<int32_t>(context.GetPlace());
  Slice<int32_t><<<grid,block, 0, stream>>>(sort_attn_accu_indices_data, slimmed_sort_attn_accu_indices_data, bsz, max_seq_len, slimmed_x_len);
  
  if (keep_order_) {
    VLOG(1) << "keep_order_";
    num_items = bsz * slimmed_x_len;
    FillOffsets<<<bsz+1, 1, 0, stream>>>(offsets_data, bsz, slimmed_x_len);
    temp_storage_bytes = -1;
    PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceSegmentedRadixSort::SortKeys(nullptr,
      temp_storage_bytes,
      slimmed_sort_attn_accu_indices_data,
      output_indices_data,
      num_items,
      num_segments,
      offsets_data,
      offsets_data+1,
      0,
      sizeof(int32_t) * 8,
      stream));

    temp_size = temp_storage_bytes;
    temp_storage.Resize({temp_size});
    temp_storage_data = temp_storage.mutable_data<uint8_t>(platform::CUDAPlace(device_id));
    PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceSegmentedRadixSort::SortKeys(temp_storage_data,
      temp_storage_bytes,
      slimmed_sort_attn_accu_indices_data,
      output_indices_data,
      num_items,
      num_segments,
      offsets_data,
      offsets_data+1,
      0,
      sizeof(int32_t) * 8,
      stream));

    TakeAlongAxis<T><<<grid, block, 0, stream>>>(
      x_data, output_data, output_indices_data, bsz, max_seq_len,
      slimmed_x_len, c);
    
  } else {
      PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(output_indices_data, 
                slimmed_sort_attn_accu_indices_data, 
                bsz * slimmed_x_len * sizeof(int32_t), 
                cudaMemcpyDeviceToDevice));
      TakeAlongAxis<T><<<grid, block, 0, stream>>>(
        x_data, output_data, slimmed_sort_attn_accu_indices_data, bsz, max_seq_len,
        slimmed_x_len, c);
  }

  return cudaGetLastError() != cudaSuccess;
}

int FusedTokenPrunePluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc* input_desc,
    const nvinfer1::PluginTensorDesc* output_desc, const void* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT {
  auto input_type = input_desc[0].type;
  auto attn_dims = input_desc[0].dims;
  auto bsz = attn_dims.d[0], nb_head = attn_dims.d[1],
       max_seq_len = attn_dims.d[2];
  int device_id;
  cudaGetDevice(&device_id);

  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(1) << "TRT Plugin DataType selected. FusedTokenPrune-->fp32";

    framework::Tensor attn_tmp;
    attn_tmp.Resize({bsz, nb_head, max_seq_len, max_seq_len});
    auto* attn_tmp_data =
        attn_tmp.mutable_data<float>(platform::CUDAPlace(device_id));

    framework::Tensor attn_accu;
    attn_accu.Resize({bsz, max_seq_len});
    auto* attn_accu_data =
        attn_accu.mutable_data<float>(platform::CUDAPlace(device_id));

    float max = std::numeric_limits<float>::max();

    return enqueueImpl<float>(input_desc, output_desc, inputs, outputs,
                              workspace, stream/*, attn_tmp_data, attn_accu_data*/,
                              device_id, max);

  } else if (input_type == nvinfer1::DataType::kHALF) {
#ifdef TRT_PLUGIN_FP16_AVALIABLE
    VLOG(1) << "TRT Plugin DataType selected. FusedTokenPrune-->fp16";

    framework::Tensor attn_tmp;
    attn_tmp.Resize({bsz, nb_head, max_seq_len, max_seq_len});
    auto* attn_tmp_data_tmp = attn_tmp.mutable_data<int16_t>(
        platform::CUDAPlace(device_id));  // NOLINT
    auto* attn_tmp_data = reinterpret_cast<half*>(attn_tmp_data_tmp);

    framework::Tensor attn_accu;
    attn_accu.Resize({bsz, max_seq_len});
    auto* attn_accu_data_tmp =
        attn_accu.mutable_data<int16_t>(platform::CUDAPlace(device_id));
    auto* attn_accu_data = reinterpret_cast<half*>(attn_accu_data_tmp);

    half max = 65504.0;

    return enqueueImpl<half>(input_desc, output_desc, inputs, outputs,
                             workspace, stream/*, attn_tmp_data, attn_accu_data*/,
                             device_id, max);

#else
    PADDLE_THROW(platform::errors::Fatal(
        "The Ernie(Bert) TensorRT Plugin should be "
        "complied with CUDA version >= 10.0 when running with fp16. "
        "Please recomplie it or try to use fp32 by set "
        "config.SetTRTDynamicShapeInfo(min_input_shape, "
        "max_input_shape, opt_input_shape, true"));
#endif
  } else {
    PADDLE_THROW(
        platform::errors::Fatal("The FusedTokenPrune TRT Plugin's input type "
                                "should be float or half."));
  }
}

#endif
}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
