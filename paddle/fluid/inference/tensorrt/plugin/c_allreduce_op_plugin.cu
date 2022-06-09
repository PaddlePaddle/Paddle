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

#include <cstring>

#include "glog/logging.h"
#include "paddle/fluid/inference/tensorrt/plugin/c_allreduce_op_plugin.h"
#include "paddle/fluid/platform/collective_helper.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

inline ncclDataType_t NvInferDtypeToNCCLDType(nvinfer1::DataType type) {
  if (type == nvinfer1::DataType::kFLOAT) {
    return ncclFloat;
  } else if (type == nvinfer1::DataType::kHALF) {
    return ncclFloat16;
  } else if (type == nvinfer1::DataType::kINT8) {
    return ncclInt8;
  } else if (type == nvinfer1::DataType::kINT32) {
    return ncclInt32;
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "This datatype in nccl is not supported."));
  }
}

int CAllReducePlugin::initialize() TRT_NOEXCEPT { return 0; }

bool CAllReducePlugin::supportsFormat(
    nvinfer1::DataType type, nvinfer1::PluginFormat format) const TRT_NOEXCEPT {
  if (with_fp16_) {
    return ((type == nvinfer1::DataType::kFLOAT ||
             type == nvinfer1::DataType::kHALF) &&
            (format == nvinfer1::PluginFormat::kLINEAR));
  } else {
    return ((type == nvinfer1::DataType::kFLOAT) &&
            (format == nvinfer1::PluginFormat::kLINEAR));
  }
}

nvinfer1::Dims CAllReducePlugin::getOutputDimensions(
    int index, const nvinfer1::Dims* in_dims, int nb_inputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(
      nb_inputs, 1,
      platform::errors::InvalidArgument("We expect [number of inputs] == 1"
                                        "in TRT CAllReduce op plugin, but got "
                                        "[number of inputs] = %d.",
                                        nb_inputs));
  PADDLE_ENFORCE_LT(index, this->getNbOutputs(),
                    platform::errors::InvalidArgument(
                        "We expect [index] < [number of outputs]"
                        "in TRT CAllReduce op plugin, but got "
                        "[index] = %d, [number of outputs] = %d.",
                        index, this->getNbOutputs()));
  nvinfer1::Dims const& input_dims = in_dims[0];
  nvinfer1::Dims output_dims = input_dims;
  return output_dims;
}

#if IS_TRT_VERSION_LT(8000)
int CAllReducePlugin::enqueue(int batchSize, const void* const* inputs,
                              void** outputs,
#else
int CAllReducePlugin::enqueue(int batchSize, const void* const* inputs,
                              void* const* outputs,
#endif
                              void* workspace,
                              cudaStream_t stream) TRT_NOEXCEPT {
  VLOG(3) << "-----------------CAllReducePlugin--------------";
  const auto& input_dims = this->getInputDims(0);
  size_t numel = 1;
  for (int i = 0; i < input_dims.nbDims; i++) {
    numel *= input_dims.d[i];
  }
  auto input_type = getDataType();
  void* sendbuff;
  void* recvbuff = outputs[0];
  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(1) << "TRT Plugin DataType selected. CAllReduce-->fp32";
    sendbuff = const_cast<void*>(inputs[0]);
  } else if (input_type == nvinfer1::DataType::kHALF) {
    VLOG(1) << "TRT Plugin DataType selected. CAllReduce-->fp16";
    sendbuff = const_cast<void*>(inputs[0]);
  }
  ncclDataType_t dtype = NvInferDtypeToNCCLDType(input_type);
  ncclRedOp_t nccl_red_type = ncclSum;
  switch (red_type_) {
    case kRedSum:
      nccl_red_type = ncclSum;
      break;

    case kRedMax:
      nccl_red_type = ncclMax;
      break;

    case kRedMin:
      nccl_red_type = ncclMin;
      break;

    case kRedProd:
      nccl_red_type = ncclProd;
      break;

    default:
      PADDLE_THROW(platform::errors::InvalidArgument("Invalid reduce type: %d",
                                                     red_type_));
  }

  auto comm = platform::NCCLCommContext::Instance().Get(ring_id_);
  cudaStream_t custream = use_calc_stream_ ? stream : custream = comm->stream();
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(
      sendbuff, recvbuff, numel, dtype, nccl_red_type, comm->comm(), stream));
  return (cudaGetLastError() != cudaSuccess);
}

// Dynamic Plugin below.
// int CAllReducePluginDynamic::initialize() TRT_NOEXCEPT {
//  getPluginNamespace();
//  return 0;
//}

size_t CAllReducePluginDynamic::getSerializationSize() const TRT_NOEXCEPT {
  return SerializedSize(ring_id_) + SerializedSize(use_calc_stream_) +
         SerializedSize(red_type_);
  +SerializedSize(with_fp16_);
}

void CAllReducePluginDynamic::serialize(void* buffer) const TRT_NOEXCEPT {
  SerializeValue(&buffer, ring_id_);
  SerializeValue(&buffer, use_calc_stream_);
  SerializeValue(&buffer, red_type_);
  SerializeValue(&buffer, with_fp16_);
}

nvinfer1::DimsExprs CAllReducePluginDynamic::getOutputDimensions(
    int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT {
  return inputs[0];
}

bool CAllReducePluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* in_out, int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_NOT_NULL(
      in_out, platform::errors::InvalidArgument(
                  "The input of CAllReduce plugin shoule not be nullptr."));

  PADDLE_ENFORCE_LT(
      pos, nb_inputs + nb_outputs,
      platform::errors::InvalidArgument("The pos(%d) should be less than the "
                                        "num(%d) of the input and the output.",
                                        pos, nb_inputs + nb_outputs));

  const nvinfer1::PluginTensorDesc& in = in_out[pos];
  if (pos == 0 || pos == 1) {
    if (with_fp16_) {
      return (in.type == nvinfer1::DataType::kHALF) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
    } else {
      return (in.type == nvinfer1::DataType::kFLOAT) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
    }
  }
}

nvinfer1::DataType CAllReducePluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType* input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(index, 0,
                    platform::errors::InvalidArgument(
                        "The CAllReduce Plugin only has one input, so the "
                        "index value should be 0, but get %d.",
                        index));
  return input_types[0];
}

int CAllReducePluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc* input_desc,
    const nvinfer1::PluginTensorDesc* output_desc, const void* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT {
  auto input_dims = input_desc[0].dims;
  size_t numel = ProductDim(input_dims);

  auto input_type = input_desc[0].type;
  void* sendbuff = const_cast<void*>(inputs[0]);
  void* recvbuff = outputs[0];
  ncclDataType_t dtype = NvInferDtypeToNCCLDType(input_type);
  ncclRedOp_t nccl_red_type = ncclSum;
  switch (red_type_) {
    case kRedSum:
      nccl_red_type = ncclSum;
      break;

    case kRedMax:
      nccl_red_type = ncclMax;
      break;

    case kRedMin:
      nccl_red_type = ncclMin;
      break;

    case kRedProd:
      nccl_red_type = ncclProd;
      break;

    default:
      PADDLE_THROW(platform::errors::InvalidArgument("Invalid reduce type: %d",
                                                     red_type_));
  }

  auto comm = platform::NCCLCommContext::Instance().Get(ring_id_);
  cudaStream_t custream = use_calc_stream_ ? stream : comm->stream();
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(
      sendbuff, recvbuff, numel, dtype, nccl_red_type, comm->comm(), stream));
  return (cudaGetLastError() != cudaSuccess);
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
