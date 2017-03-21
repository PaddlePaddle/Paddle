/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "PaddleCAPI.h"
#include "PaddleCAPIPrivate.h"

using paddle::capi::cast;

#define castArg(v) cast<paddle::capi::CArguments>(v)
#define castIVec(v) cast<paddle::capi::CIVector>(v)

extern "C" {
paddle_arguments paddle_arguments_create_none() {
  return new paddle::capi::CArguments();
}

paddle_error paddle_arguments_destroy(paddle_arguments args) {
  if (args == nullptr) return kPD_NULLPTR;
  delete castArg(args);
  return kPD_NO_ERROR;
}

paddle_error paddle_arguments_size(paddle_arguments args, uint64_t* size) {
  if (args == nullptr || size == nullptr) return kPD_NULLPTR;
  *size = castArg(args)->args.size();
  return kPD_NO_ERROR;
}

paddle_error paddle_arguments_resize(paddle_arguments args, uint64_t size) {
  if (args == nullptr) return kPD_NULLPTR;
  castArg(args)->args.resize(size);
  return kPD_NO_ERROR;
}

paddle_error paddle_arguments_set_value(paddle_arguments args,
                                        uint64_t ID,
                                        paddle_matrix mat) {
  if (args == nullptr || mat == nullptr) return kPD_NULLPTR;
  auto m = paddle::capi::cast<paddle::capi::CMatrix>(mat);
  if (m->mat == nullptr) return kPD_NULLPTR;
  auto a = castArg(args);
  if (ID >= a->args.size()) return kPD_OUT_OF_RANGE;
  a->args[ID].value = m->mat;
  return kPD_NO_ERROR;
}

paddle_error paddle_arguments_value(paddle_arguments args,
                                    uint64_t ID,
                                    paddle_matrix mat) {
  if (args == nullptr || mat == nullptr) return kPD_NULLPTR;
  auto m = paddle::capi::cast<paddle::capi::CMatrix>(mat);
  auto a = castArg(args);
  if (ID >= a->args.size()) return kPD_OUT_OF_RANGE;
  m->mat = a->args[ID].value;
  return kPD_NO_ERROR;
}

paddle_error paddle_arguments_ids(paddle_arguments args,
                                  uint64_t ID,
                                  paddle_ivector ids) {
  if (args == nullptr || ids == nullptr) return kPD_NULLPTR;
  auto iv = castIVec(ids);
  auto a = castArg(args);
  if (ID >= a->args.size()) return kPD_OUT_OF_RANGE;
  iv->vec = a->args[ID].ids;
  return kPD_NO_ERROR;
}

paddle_error paddle_arguments_set_ids(paddle_arguments args,
                                      uint64_t ID,
                                      paddle_ivector ids) {
  //! TODO(lizhao): Complete this method.
  if (args == nullptr || ids == nullptr) return kPD_NULLPTR;
  auto iv = paddle::capi::cast<paddle::capi::CIVector>(ids);
  if (iv->vec == nullptr) return kPD_NULLPTR;
  auto a = castArg(args);
  if (ID >= a->args.size()) return kPD_OUT_OF_RANGE;
  a->args[ID].ids = iv->vec;
  return kPD_NO_ERROR;
}

paddle_error paddle_arguments_set_sequence_start_pos(paddle_arguments args,
                                                     uint64_t ID,
                                                     paddle_ivector seqPos) {
  if (args == nullptr || seqPos == nullptr) return kPD_NULLPTR;
  auto iv = paddle::capi::cast<paddle::capi::CIVector>(seqPos);
  if (iv->vec == nullptr) return kPD_NULLPTR;
  auto a = castArg(args);
  if (ID >= a->args.size()) return kPD_OUT_OF_RANGE;
  a->args[ID].sequenceStartPositions =
      std::make_shared<paddle::ICpuGpuVector>(iv->vec);
  return kPD_NO_ERROR;
}

paddle_error paddle_arguments_set_sub_sequence_start_pos(
    paddle_arguments args, uint64_t ID, paddle_ivector subSeqPos) {
  if (args == nullptr || subSeqPos == nullptr) return kPD_NULLPTR;
  auto iv = paddle::capi::cast<paddle::capi::CIVector>(subSeqPos);
  if (iv->vec == nullptr) return kPD_NULLPTR;
  auto a = castArg(args);
  if (ID >= a->args.size()) return kPD_OUT_OF_RANGE;
  a->args[ID].subSequenceStartPositions =
      std::make_shared<paddle::ICpuGpuVector>(iv->vec);
  return kPD_NO_ERROR;
}

paddle_error paddle_arguments_sequence_start_pos(paddle_arguments args,
                                                 uint64_t ID,
                                                 paddle_ivector seqPos) {
  if (args == nullptr || seqPos == nullptr) return kPD_NULLPTR;
  auto iv = castIVec(seqPos);
  auto a = castArg(args);
  if (ID >= a->args.size()) return kPD_OUT_OF_RANGE;
  paddle::Argument& arg = a->args[ID];
  iv->vec = arg.sequenceStartPositions->getMutableVector(false);
  return kPD_NO_ERROR;
}

paddle_error paddle_arguments_sub_sequence_start_pos(paddle_arguments args,
                                                     uint64_t ID,
                                                     paddle_ivector subSeqPos) {
  if (args == nullptr || subSeqPos == nullptr) return kPD_NULLPTR;
  auto iv = castIVec(subSeqPos);
  auto a = castArg(args);
  if (ID >= a->args.size()) return kPD_OUT_OF_RANGE;
  paddle::Argument& arg = a->args[ID];
  iv->vec = arg.subSequenceStartPositions->getMutableVector(false);
  return kPD_NO_ERROR;
}
}
