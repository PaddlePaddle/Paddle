// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/capi/include/c_infer_meta_context.h"

#include "paddle/phi/capi/include/common.h"
#include "paddle/phi/capi/include/type_utils.h"
#include "paddle/phi/core/infermeta_utils.h"

PD_MetaTensor* PD_InferMetaContextInputAt(PD_InferMetaContext* ctx,
                                          size_t index) {
  auto* meta_ctx = reinterpret_cast<phi::InferMetaContext*>(ctx);
  const std::pair<int, int> range = meta_ctx->InputRangeAt(index);
  const phi::MetaTensor& arg = meta_ctx->InputAt(range.first);
  return reinterpret_cast<PD_MetaTensor*>(const_cast<phi::MetaTensor*>(&arg));
}

PD_List PD_InferMetaContextMultiInputAt(PD_InferMetaContext* ctx,
                                        size_t index) {
  auto* meta_ctx = reinterpret_cast<phi::InferMetaContext*>(ctx);
  const std::pair<int, int> range = meta_ctx->InputRangeAt(index);
  std::vector<const phi::MetaTensor*> tensor_vec =
      meta_ctx->InputsBetween(range.first, range.second);
  PD_List list;
  list.size = tensor_vec.size();
  list.data = new void*[list.size];
  for (size_t i = 0; i < list.size; ++i) {
    (reinterpret_cast<void**>(list.data))[i] =
        reinterpret_cast<void*>(const_cast<phi::MetaTensor*>(tensor_vec[i]));
  }
  return list;
}

PD_MetaTensor* PD_InferMetaContextOutputAt(PD_InferMetaContext* ctx,
                                           size_t index) {
  auto* meta_ctx = reinterpret_cast<phi::InferMetaContext*>(ctx);
  const std::pair<int, int> range = meta_ctx->OutputRangeAt(index);
  phi::MetaTensor* arg = meta_ctx->MutableOutputAt(range.first);
  return reinterpret_cast<PD_MetaTensor*>(arg);
}

PD_List PD_InferMetaContextMultiOutputAt(PD_InferMetaContext* ctx,
                                         size_t index) {
  auto* meta_ctx = reinterpret_cast<phi::InferMetaContext*>(ctx);
  const std::pair<int, int> range = meta_ctx->OutputRangeAt(index);
  std::vector<phi::MetaTensor*> tensor_vec =
      meta_ctx->MutableOutputBetween(range.first, range.second);
  PD_List list;
  list.size = tensor_vec.size();
  list.data = new void*[list.size];
  for (size_t i = 0; i < list.size; ++i) {
    (reinterpret_cast<void**>(list.data))[i] =
        reinterpret_cast<void*>(tensor_vec[i]);
  }
  return list;
}

bool PD_InferMetaContextBoolAttrAt(PD_InferMetaContext* ctx, size_t index) {
  auto meta_ctx = reinterpret_cast<phi::InferMetaContext*>(ctx);
  return meta_ctx->AttrAt<bool>(index);
}

int32_t PD_InferMetaContextInt32AttrAt(PD_InferMetaContext* ctx, size_t index) {
  auto meta_ctx = reinterpret_cast<phi::InferMetaContext*>(ctx);
  return meta_ctx->AttrAt<int32_t>(index);
}

int64_t PD_InferMetaContextInt64AttrAt(PD_InferMetaContext* ctx, size_t index) {
  auto meta_ctx = reinterpret_cast<phi::InferMetaContext*>(ctx);
  return meta_ctx->AttrAt<int64_t>(index);
}

float PD_InferMetaContextFloatAttrAt(PD_InferMetaContext* ctx, size_t index) {
  auto meta_ctx = reinterpret_cast<phi::InferMetaContext*>(ctx);
  return meta_ctx->AttrAt<float>(index);
}

double PD_InferMetaContextDoubleAttrAt(PD_InferMetaContext* ctx, size_t index) {
  auto meta_ctx = reinterpret_cast<phi::InferMetaContext*>(ctx);
  return meta_ctx->AttrAt<double>(index);
}

PD_Scalar* PD_InferMetaContextScalarAttrAt(PD_InferMetaContext* ctx,
                                           size_t index) {
  auto meta_ctx = reinterpret_cast<phi::InferMetaContext*>(ctx);
  return reinterpret_cast<PD_Scalar*>(
      const_cast<phi::Scalar*>(&meta_ctx->AttrAt<phi::Scalar>(index)));
}

PD_IntArray* PD_InferMetaContextIntArrayAttrAt(PD_InferMetaContext* ctx,
                                               size_t index) {
  auto meta_ctx = reinterpret_cast<phi::InferMetaContext*>(ctx);
  return reinterpret_cast<PD_IntArray*>(
      const_cast<phi::IntArray*>(&meta_ctx->AttrAt<phi::IntArray>(index)));
}

PD_List PD_InferMetaContextListBoolAttrAt(PD_InferMetaContext* ctx,
                                          size_t index) {
  PD_List list;
  auto meta_ctx = reinterpret_cast<phi::InferMetaContext*>(ctx);
  const auto& cc_list = meta_ctx->AttrAt<std::vector<bool>>(index);
  list.size = cc_list.size();
  auto data = reinterpret_cast<uint8_t*>(new uint8_t[cc_list.size()]);
  for (size_t i = 0; i < cc_list.size(); ++i) {
    data[i] = static_cast<uint8_t>(cc_list[i]);
  }
  list.data = data;
  return list;
}

PD_List PD_InferMetaContextListInt32AttrAt(PD_InferMetaContext* ctx,
                                           size_t index) {
  PD_List list;
  auto meta_ctx = reinterpret_cast<phi::InferMetaContext*>(ctx);
  const auto& cc_list = meta_ctx->AttrAt<std::vector<int32_t>>(index);
  list.size = cc_list.size();
  list.data = const_cast<int32_t*>(cc_list.data());
  return list;
}

PD_List PD_InferMetaContextListInt64AttrAt(PD_InferMetaContext* ctx,
                                           size_t index) {
  PD_List list;
  auto meta_ctx = reinterpret_cast<phi::InferMetaContext*>(ctx);
  const auto& cc_list = meta_ctx->AttrAt<std::vector<int64_t>>(index);
  list.size = cc_list.size();
  list.data = const_cast<int64_t*>(cc_list.data());
  return list;
}

PD_List PD_InferMetaContextListFloatAttrAt(PD_InferMetaContext* ctx,
                                           size_t index) {
  PD_List list;
  auto meta_ctx = reinterpret_cast<phi::InferMetaContext*>(ctx);
  const auto& cc_list = meta_ctx->AttrAt<std::vector<float>>(index);
  list.size = cc_list.size();
  list.data = const_cast<float*>(cc_list.data());
  return list;
}

PD_List PD_InferMetaContextListDoubleAttrAt(PD_InferMetaContext* ctx,
                                            size_t index) {
  PD_List list;
  auto meta_ctx = reinterpret_cast<phi::InferMetaContext*>(ctx);
  const auto& cc_list = meta_ctx->AttrAt<std::vector<double>>(index);
  list.size = cc_list.size();
  list.data = const_cast<double*>(cc_list.data());
  return list;
}

char* PD_InferMetaContextStringAttrAt(PD_InferMetaContext* ctx, size_t index) {
  auto meta_ctx = reinterpret_cast<phi::InferMetaContext*>(ctx);
  return const_cast<char*>(meta_ctx->AttrAt<std::string>(index).data());
}

PD_List PD_InferMetaContextListStringAttrAt(PD_InferMetaContext* ctx,
                                            size_t index) {
  PD_List list;
  auto meta_ctx = reinterpret_cast<phi::InferMetaContext*>(ctx);
  const auto& cc_list = meta_ctx->AttrAt<std::vector<std::string>>(index);
  list.size = cc_list.size();
  auto data = new char*[list.size];
  for (size_t i = 0; i < list.size; ++i) {
    data[i] = const_cast<char*>(cc_list[i].data());
  }
  list.data = reinterpret_cast<void*>(data);
  return list;
}

PD_List PD_InferMetaContextListScalarAttrAt(PD_InferMetaContext* ctx,
                                            size_t index) {
  PD_List list;
  auto meta_ctx = reinterpret_cast<phi::InferMetaContext*>(ctx);
  const auto& cc_list = meta_ctx->AttrAt<std::vector<phi::Scalar>>(index);
  list.size = cc_list.size();
  auto data = new PD_Scalar*[list.size];
  for (size_t i = 0; i < list.size; ++i) {
    data[i] =
        const_cast<PD_Scalar*>(reinterpret_cast<const PD_Scalar*>(&cc_list[i]));
  }
  list.data = data;
  return list;
}

PD_Place* PD_InferMetaContextPlaceAttrAt(PD_InferMetaContext* ctx,
                                         size_t index) {
  auto meta_ctx = reinterpret_cast<phi::InferMetaContext*>(ctx);
  return reinterpret_cast<PD_Place*>(
      const_cast<phi::Place*>(&meta_ctx->AttrAt<phi::Place>(index)));
}

PD_DataType PD_InferMetaContextDataTypeAttrAt(PD_InferMetaContext* ctx,
                                              size_t index) {
  auto meta_ctx = reinterpret_cast<phi::InferMetaContext*>(ctx);
  return phi::capi::ToPDDataType(meta_ctx->AttrAt<phi::DataType>(index));
}

PD_DataLayout PD_InferMetaContextDataLayoutAttrAt(PD_InferMetaContext* ctx,
                                                  size_t index) {
  auto meta_ctx = reinterpret_cast<phi::InferMetaContext*>(ctx);
  return phi::capi::ToPDDataLayout(meta_ctx->AttrAt<phi::DataLayout>(index));
}

PD_REGISTER_CAPI(infer_meta_context);
