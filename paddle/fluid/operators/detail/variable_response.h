//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/var_type.h"

#include "paddle/fluid/operators/detail/send_recv.grpc.pb.h"
#include "paddle/fluid/operators/detail/send_recv.pb.h"

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/detail/bytebuffer_stream.h"

namespace paddle {
namespace operators {
namespace detail {

class VariableResponse {
 public:
  VariableResponse(const framework::Scope* scope,
                   const platform::DeviceContext* dev_ctx)
      : scope_(scope), dev_ctx_(dev_ctx){};

  virtual ~VariableResponse(){};

  // return:
  // 0:ok.
  // -1: unkown error.
  // other: number of error field.
  int Parse(Source* source);

  // return:
  // 0:ok.
  // -1: unkown error.
  // other: number of error field.
  int Parse(const ::grpc::ByteBuffer& byte_buffer);

  inline std::string Varname() { return meta_.varname(); }

  // should call parse first.
  framework::Variable* GetVar() { return scope_->FindVar(meta_.varname()); }

 private:
  bool CopySelectRowsTensorData(::google::protobuf::io::CodedInputStream* input,
                                const platform::DeviceContext& ctx,
                                framework::DDim& dims, int length);

  bool CopySelectRowsData(::google::protobuf::io::CodedInputStream* input,
                          const platform::DeviceContext& ctx, int length);

  bool CopyLodTensorData(::google::protobuf::io::CodedInputStream* input,
                         const platform::DeviceContext& ctx,
                         framework::DDim& dims, int length);

 private:
  const framework::Scope* scope_;
  const platform::DeviceContext* dev_ctx_;
  // only Skeleton
  sendrecv::VariableMessage meta_;
};

};  // namespace detail
};  // namespace operators
};  // namespace paddle
