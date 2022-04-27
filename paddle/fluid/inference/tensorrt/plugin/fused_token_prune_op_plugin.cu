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

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

#if IS_TRT_VERSION_GE(6000)

template <typename T>
__global__ void ElementwiseMul(const T* a, const T* b, T* res, int num_raws,
                               int num_cols) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_raws * num_cols) return;
  const T zero = 0;
  res[tid] = b[tid] >= zero ? a[tid] : zero;
}

nvinfer1::DimsExprs FusedTokenPrunePluginDynamic::getOutputDimensions(
    int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(
      output_index, 0,
      platform::errors::InvalidArgument(
          "The FusedTokenPrune Plugin only has one output, so the "
          "index value should be 0, but get %d.",
          output_index));
  auto attn_dims = inputs[0], x_dims = inputs[1];
  int max_seq_len = attn_dims.d[2]->getConstantValue();
  int slimmed_seq_len = max_seq_len * factor_;
  nvinfer1::DimsExprs ret = x_dims;
  ret.d[1] = expr_builder.constant(slimmed_seq_len);
  return ret;
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
  if (pos < 3) {
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
  }
  const nvinfer1::PluginTensorDesc& prev = in_out[pos - 1];

  // output
  return in.type == prev.type && in.format == prev.format;
}

nvinfer1::DataType FusedTokenPrunePluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType* input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(
      index, 0,
      platform::errors::InvalidArgument(
          "The EmbEltwiseLayernorm Plugin only has one output, so the "
          "index value should be 0, but get %d.",
          index));
  return input_types[1];
}

template <typename T>
int FusedTokenPrunePluginDynamic::enqueueImpl(
    const nvinfer1::PluginTensorDesc* input_desc,
    const nvinfer1::PluginTensorDesc* output_desc, const void* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) {
  int device_id;
  cudaGetDevice(&device_id);

  auto attn_dims = input_desc[0].dims;
  auto x_dims = input_desc[1].dims;
  auto bsz = attn_dims.d[0], nb_head = attn_dims.d[1],
       max_seq_len = attn_dims.d[2];
  auto c = x_dims.d[2];

  const T* attn_data = static_cast<const T*>(inputs[0]);
  const T* x_data = static_cast<const T*>(inputs[1]);
  const T* mask_data = static_cast<const T*>(inputs[2]);
  T* output_data = static_cast<T*>(outputs[0]);

  framework::Tensor attn_tmp;
  attn_tmp.Resize({bsz, nb_head, max_seq_len, max_seq_len});
  auto* attn_tmp_data =
      attn_tmp.mutable_data<T>(platform::CUDAPlace(device_id));

  int grid = bsz * nb_head * max_seq_len;
  int block = operators::ComputeBlockSize(max_seq_len);
  ElementwiseMul<T><<<grid, block, 0, stream>>>(
      attn_data, mask_data, attn_tmp_data, grid, max_seq_len);

  auto* device_ctx = static_cast<platform::CUDADeviceContext*>(
      platform::DeviceContextPool::Instance().Get(
          platform::CUDAPlace(device_id)));
  const platform::CUDADeviceContext& dev_ctx = *device_ctx;
  framework::Tensor attn_by;
  attn_by.Resize({bsz, max_seq_len});
  auto* attn_by_data = attn_by.mutable_data<T>(platform::CUDAPlace(device_id));
  const std::vector<int64_t> reduce_dims{1, 2};
  phi::Reduce<T, kps::AddFunctor, kps::IdentityFunctor>(
      dev_ctx, attn_tmp, false, reduce_dims, false, attn_by.dtype(), &attn_by);

  framework::Tensor attn_by_indices;
  attn_by_indices.Resize({bsz, max_seq_len});
  auto* attn_by_indices_data =
      attn_by_indices.mutable_data<int>(platform::CUDAPlace(device_id));
  grid = bsz;
  operators::FillIndex<<<grid, block, 0, stream>>>(attn_by_indices_data, bsz,
                                                   max_seq_len);
  operators::SlicedArgsort<T><<<grid, 1, 0, stream>>>(
      attn_by_data, attn_by_indices_data, bsz, max_seq_len);

  int slimmed_x_len = max_seq_len * factor_;
  auto slimmed_indices = phi::funcs::Slice<int>(dev_ctx, attn_by_indices, {1},
                                                {0}, {slimmed_x_len});
  block = operators::ComputeBlockSize(slimmed_x_len);
  operators::TakeAlongAxis<T><<<grid, block, 0, stream>>>(
      x_data, output_data, slimmed_indices.data<int>(), bsz, max_seq_len,
      slimmed_x_len, c);
}

int FusedTokenPrunePluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc* input_desc,
    const nvinfer1::PluginTensorDesc* output_desc, const void* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT {
  auto input_type = input_desc[0].type;
  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(1) << "TRT Plugin DataType selected. FusedTokenPrune-->fp32";
    enqueueImpl<float>(input_desc, output_desc, inputs, outputs, workspace,
                       stream);
  } else if (input_type == nvinfer1::DataType::kHALF) {
#ifdef TRT_PLUGIN_FP16_AVALIABLE
    VLOG(1) << "TRT Plugin DataType selected. FusedTokenPrune-->fp16";
    enqueueImpl<half>(input_desc, output_desc, inputs, outputs, workspace,
                      stream);
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
  return cudaGetLastError() != cudaSuccess;
}

#endif
}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
