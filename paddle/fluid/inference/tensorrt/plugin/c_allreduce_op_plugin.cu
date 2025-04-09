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
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/platform/collective_helper.h"
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#endif

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {
#if defined(PADDLE_WITH_NCCL)
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
    PADDLE_THROW(common::errors::Unimplemented(
        "This datatype in nccl is not supported."));
  }
}
#endif

CAllReducePluginDynamic::CAllReducePluginDynamic(void const* serialData,
                                                 size_t serialLength) {
  DeserializeValue(&serialData, &serialLength, &ring_id_);
  DeserializeValue(&serialData, &serialLength, &use_calc_stream_);
  DeserializeValue(&serialData, &serialLength, &red_type_);
  DeserializeValue(&serialData, &serialLength, &with_fp16_);
}
nvinfer1::IPluginV2DynamicExt* CAllReducePluginDynamic::clone() const
    TRT_NOEXCEPT {
  return new CAllReducePluginDynamic(
      ring_id_, use_calc_stream_, red_type_, with_fp16_);
}

const char* CAllReducePluginDynamic::getPluginType() const TRT_NOEXCEPT {
  return "c_allreduce_plugin_dynamic";
}
int CAllReducePluginDynamic::getNbOutputs() const TRT_NOEXCEPT { return 1; }
int CAllReducePluginDynamic::initialize() TRT_NOEXCEPT { return 0; };

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
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT {
  return inputs[0];
}

bool CAllReducePluginDynamic::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc* in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_NOT_NULL(
      in_out,
      common::errors::InvalidArgument(
          "The input of CAllReduce plugin should not be nullptr."));

  PADDLE_ENFORCE_LT(
      pos,
      nb_inputs + nb_outputs,
      common::errors::InvalidArgument("The pos(%d) should be less than the "
                                      "num(%d) of the input and the output.",
                                      pos,
                                      nb_inputs + nb_outputs));

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

void CAllReducePluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in,
    int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int nbOutputs) TRT_NOEXCEPT {}

size_t CAllReducePluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs,
    int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int nbOutputs) const TRT_NOEXCEPT {
  return 0;
}

void CAllReducePluginDynamic::destroy() TRT_NOEXCEPT { delete this; }

nvinfer1::DataType CAllReducePluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType* input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(index,
                    0,
                    common::errors::InvalidArgument(
                        "The CAllReduce Plugin only has one input, so the "
                        "index value should be 0, but get %d.",
                        index));
  return input_types[0];
}

int CAllReducePluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc* input_desc,
    const nvinfer1::PluginTensorDesc* output_desc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
#if defined(PADDLE_WITH_NCCL)
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
      PADDLE_THROW(common::errors::InvalidArgument("Invalid reduce type: %d",
                                                   red_type_));
  }
  const auto& comm_context_manager =
      phi::distributed::CommContextManager::GetInstance();
  PADDLE_ENFORCE_EQ(comm_context_manager.Has(std::to_string(ring_id_)),
                    true,
                    common::errors::InvalidArgument(
                        "You choose to use new communication library. "
                        "But ring_id(%d) is "
                        "not found in comm_context_manager.",
                        std::to_string(ring_id_)));
  auto comm_ctx = static_cast<phi::distributed::NCCLCommContext*>(
      comm_context_manager.Get(std::to_string(ring_id_)));
  PADDLE_ENFORCE_NE(comm_ctx,
                    nullptr,
                    common::errors::Unavailable(
                        "NCCLCommContext is nullptr, collective op should "
                        "has ring_id attr."));
  auto stream2 = comm_ctx->GetStream();
  // ncclRedOp_t nccl_red_type = ncclSum;
  // comm_ctx->AllReduce(&inputs[0], inputs[0], nccl_red_type, stream);
  phi::dynload::ncclAllReduce(sendbuff,
                              recvbuff,
                              numel,
                              dtype,
                              nccl_red_type,
                              comm_ctx->GetNcclComm(),
                              stream2);
  VLOG(3) << "new NCCLCommContext has ring_id_ " << ring_id_;
#endif
  return (cudaGetLastError() != cudaSuccess);
}

const char* CAllReducePluginDynamicCreator::getPluginName() const TRT_NOEXCEPT {
  return "c_allreduce_plugin_dynamic";
}

const char* CAllReducePluginDynamicCreator::getPluginVersion() const
    TRT_NOEXCEPT {
  return "1";
}

nvinfer1::IPluginV2* CAllReducePluginDynamicCreator::deserializePlugin(
    const char* name,
    const void* serial_data,
    size_t serial_length) TRT_NOEXCEPT {
  auto plugin = new CAllReducePluginDynamic(serial_data, serial_length);
  return plugin;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
