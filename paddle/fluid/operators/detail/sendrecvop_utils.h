/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <sys/time.h>
#include <iostream>
#include <string>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/var_type.h"

#include "paddle/fluid/operators/detail/send_recv.grpc.pb.h"
#include "paddle/fluid/operators/detail/send_recv.pb.h"

namespace paddle {
namespace operators {
namespace detail {

#define LISTEN_TERMINATE_MESSAGE "TERMINATE@RECV"
#define BATCH_BARRIER_MESSAGE "BATCH_BARRIER@RECV"
#define FETCH_BARRIER_MESSAGE "FETCH_BARRIER@RECV"

static int64_t GetTimestamp() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return tp.tv_sec * 1000 + tp.tv_usec / 1000;
}

typedef void (*DestroyCallback)(void*);

void SerializeToByteBuffer(const std::string& name, framework::Variable* var,
                           const platform::DeviceContext& ctx,
                           ::grpc::ByteBuffer* msg,
                           const std::string& out_varname = std::string());

void DeserializeFromByteBuffer(const ::grpc::ByteBuffer& msg,
                               const platform::DeviceContext& ctx,
                               const framework::Scope* scope,
                               framework::Variable** var);

inline std::type_index ToTypeIndex(sendrecv::VariableMessage::Type type) {
  switch (type) {
    case sendrecv::VariableMessage::FP32:
      return typeid(float);  // NOLINT
    case sendrecv::VariableMessage::FP64:
      return typeid(double);  // NOLINT
    case sendrecv::VariableMessage::INT32:
      return typeid(int);  // NOLINT
    case sendrecv::VariableMessage::INT64:
      return typeid(int64_t);  // NOLINT
    case sendrecv::VariableMessage::BOOL:
      return typeid(bool);  // NOLINT
    default:
      PADDLE_THROW("Not support type %d", type);
  }
}

}  // namespace detail
}  // namespace operators
}  // namespace paddle
