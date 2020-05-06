// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <stdio.h>
#include <cassert>
#include <cub/cub.cuh>  // NOLINT
#include <vector>
#include "glog/logging.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/inference/tensorrt/plugin/emb_eltwise_layernorm_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_factory.h"
#include "paddle/fluid/operators/math/bert_encoder_functor.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

// Dynamic shape plugin requires TRT version greater than 6.0.
#if IS_TRT_VERSION_GE(6000)

template <typename T>
int EmbEltwiseLayernormPluginDynamic<T>::initialize() {
  int nb_emb = embs_.size();
  std::vector<void *> ptr_vector(nb_emb);
  std::vector<std::vector<half>> emb_fp16(nb_emb);

  if (sizeof(T) == sizeof(float)) {
    // FP32
    for (int i = 0; i < nb_emb; ++i) {
      ptr_vector[i] = embs_[i];
    }
  } else {
    // FP16
    for (int i = 0; i < nb_emb; ++i) {
      auto emb_size = emb_sizes_[i];
      auto &tmp = emb_fp16[i];
      tmp.resize(emb_size);

      for (int j = 0; j < emb_size; ++j) {
        tmp[j] = static_cast<half>(embs_[i][j]);
      }
      ptr_vector[i] = tmp.data();
    }
  }
  embs_gpu_.resize(embs_.size());
  for (int i = 0; i < embs_.size(); i++) {
    cudaMalloc(&embs_gpu_[i], sizeof(T) * emb_sizes_[i]);
    cudaMemcpy(embs_gpu_[i], ptr_vector[i], emb_sizes_[i] * sizeof(T),
               cudaMemcpyHostToDevice);
  }

  cudaMalloc(&bias_gpu_, sizeof(float) * bias_size_);
  cudaMemcpy(bias_gpu_, bias_, bias_size_ * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMalloc(&scale_gpu_, sizeof(float) * scale_size_);
  cudaMemcpy(scale_gpu_, scale_, scale_size_ * sizeof(float),
             cudaMemcpyHostToDevice);

  return 0;
}

template <typename T>
size_t EmbEltwiseLayernormPluginDynamic<T>::getSerializationSize() const {
  return 0;
}

template <typename T>
void EmbEltwiseLayernormPluginDynamic<T>::serialize(void *buffer) const {}

template <typename T>
nvinfer1::DimsExprs EmbEltwiseLayernormPluginDynamic<T>::getOutputDimensions(
    int output_index, const nvinfer1::DimsExprs *inputs, int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) {  // NOLINT
  PADDLE_ENFORCE_EQ(output_index, 0,
                    platform::errors::InvalidArgument(
                        "There is only one output of the EmbEltwiseLayernorm, "
                        "so the index should be zero,"
                        "but it's (%d)",
                        output_index));
  PADDLE_ENFORCE_EQ(
      nb_inputs, 3,
      platform::errors::InvalidArgument(
          "The Input of the EmbEltwiseLayernorm should be 3, but we found "
          "it has (%d) inputs",
          nb_inputs));
  nvinfer1::DimsExprs ret;
  ret.nbDims = 5;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = inputs[0].d[1];
  ret.d[2] = expr_builder.constant(hidden_size_);
  ret.d[3] = expr_builder.constant(1);
  ret.d[4] = expr_builder.constant(1);
  return ret;
}

template <typename T>
bool EmbEltwiseLayernormPluginDynamic<T>::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *in_out, int nb_inputs,
    int nb_outputs) {
  PADDLE_ENFORCE_NOT_NULL(
      in_out, platform::errors::InvalidArgument(
                  "The input of swish plugin shoule not be nullptr."));

  PADDLE_ENFORCE_LT(
      pos, nb_inputs + nb_outputs,
      platform::errors::InvalidArgument("The pos(%d) should be less than the "
                                        "num(%d) of the input and the output.",
                                        pos, nb_inputs + nb_outputs));
  (in_out && pos < (nb_inputs + nb_outputs));

  const nvinfer1::PluginTensorDesc &desc = in_out[pos];
  if (desc.format != nvinfer1::TensorFormat::kLINEAR) {
    return false;
  }

  if (pos == 0) {
    return desc.type == nvinfer1::DataType::kINT32;
  }

  const nvinfer1::PluginTensorDesc &prev = in_out[pos - 1];
  if (pos == 1 || pos == 2) {
    return desc.type == nvinfer1::DataType::kINT32 &&
           desc.dims.d[0] == prev.dims.d[0] && desc.dims.d[1] == prev.dims.d[1];
  }

  if (pos == 3) {
    if (sizeof(T) == sizeof(float)) {
      return desc.type == nvinfer1::DataType::kFLOAT;
    } else {
      return desc.type == nvinfer1::DataType::kHALF;
    }
  }
}

template <typename T>
nvinfer1::DataType EmbEltwiseLayernormPluginDynamic<T>::getOutputDataType(
    int index, const nvinfer1::DataType *input_types, int nb_inputs) const {
  PADDLE_ENFORCE_EQ(
      index, 0, platform::errors::InvalidArgument(
                    "The EmbEltwiseLayernorm Plugin only has one input, so the "
                    "index value should be 0, but get %d.",
                    index));
  return nvinfer1::DataType::kFLOAT;
}

template <typename T>
int EmbEltwiseLayernormPluginDynamic<T>::enqueue(
    const nvinfer1::PluginTensorDesc *input_desc,
    const nvinfer1::PluginTensorDesc *output_desc, const void *const *inputs,
    void *const *outputs, void *workspace, cudaStream_t stream) {
  auto id_dims = input_desc[0].dims;
  int batch = id_dims.d[0];
  int seq_len = id_dims.d[1];
  int input_num = embs_.size();

  framework::Tensor in_ptr_tensor, emb_ptr_tensor;
  int device_id;
  cudaGetDevice(&device_id);

  in_ptr_tensor.Resize({input_num});
  emb_ptr_tensor.Resize({input_num});
  int64_t *in_ptr_gpu_d =
      in_ptr_tensor.mutable_data<int64_t>(platform::CUDAPlace(device_id));
  int64_t *emb_ptr_gpu_d =
      emb_ptr_tensor.mutable_data<int64_t>(platform::CUDAPlace(device_id));

  std::vector<int64_t> in_ptr, emb_ptr;
  for (int i = 0; i < input_num; i++) {
    in_ptr.push_back(reinterpret_cast<uintptr_t>(inputs[i]));
    emb_ptr.push_back(reinterpret_cast<uintptr_t>(embs_gpu_[i]));
  }

  cudaMemcpyAsync(in_ptr_gpu_d, in_ptr.data(), sizeof(int64_t) * input_num,
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(emb_ptr_gpu_d, emb_ptr.data(), sizeof(int64_t) * input_num,
                  cudaMemcpyHostToDevice, stream);

  auto out_type = output_desc[0].type;

  const unsigned tpb = 256;
  const dim3 grid(seq_len, batch, 1);
  const dim3 block(tpb, 1, 1);
  if (sizeof(T) == sizeof(float)) {
    PADDLE_ENFORCE_EQ(
        out_type == nvinfer1::DataType::kFLOAT, true,
        platform::errors::InvalidArgument(
            "The EmbEltwiseLayernorm Plugin only support fp32 input."));
  } else if (sizeof(T) == sizeof(int16_t)) {
    PADDLE_ENFORCE_EQ(
        out_type == nvinfer1::DataType::kHALF, true,
        platform::errors::InvalidArgument(
            "The EmbEltwiseLayernorm Plugin only support fp16 input."));
  } else {
    PADDLE_THROW(platform::errors::Fatal(
        "Unsupport data type, the out type of EmbEltwiseLayernorm should be "
        "float or half."));
  }

  T *output_d = static_cast<T *>(outputs[0]);

  operators::math::EmbEltwiseLayerNormFunctor<T> emb_eltwise_layernorm_func;
  emb_eltwise_layernorm_func(batch, seq_len, hidden_size_, in_ptr_gpu_d,
                             scale_gpu_, bias_gpu_, emb_ptr_gpu_d, output_d,
                             eps_, input_num, stream);
  return cudaGetLastError() != cudaSuccess;
}

template class EmbEltwiseLayernormPluginDynamic<float>;
#ifdef SUPPORTS_CUDA_FP16
template class EmbEltwiseLayernormPluginDynamic<half>;
#endif  // SUPPORTS_CUDA_FP16

#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
