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

#include <string>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/var_type.h"

#include "paddle/fluid/operators/distributed/send_recv.grpc.pb.h"
#include "paddle/fluid/operators/distributed/send_recv.pb.h"

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/distributed/bytebuffer_stream.h"

namespace paddle {
namespace operators {
namespace distributed {

class VariableResponse {
 public:
  VariableResponse(const framework::Scope* scope,
                   const platform::DeviceContext* dev_ctx,
                   bool create_scope = false)
      : scope_(scope), dev_ctx_(dev_ctx), create_scope_(create_scope) {
    if (create_scope) {
      local_scope_ = &scope->NewScope();
    }
  }

  virtual ~VariableResponse() {
    if (create_scope_) {
      scope_->DeleteScope(local_scope_);
    }
  }

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

  const framework::Scope& GetLocalScope() const { return *local_scope_; }

  framework::Scope* GetMutableLocalScope() const { return local_scope_; }

  inline std::string Varname() const { return meta_.varname(); }
  inline std::string OutVarname() const { return meta_.out_varname(); }

  // should call parse first.
  framework::Variable* GetVar() {
    if (create_scope_) {
      return local_scope_->Var(meta_.varname());
    }
    return scope_->FindVar(meta_.varname());
  }

 private:
  bool CopySelectRowsTensorData(::google::protobuf::io::CodedInputStream* input,
                                const platform::DeviceContext& ctx,
                                const framework::DDim& dims, int length);

  bool CopySelectRowsData(::google::protobuf::io::CodedInputStream* input,
                          const platform::DeviceContext& ctx, int length);

  bool CopyLodTensorData(::google::protobuf::io::CodedInputStream* input,
                         const platform::DeviceContext& ctx,
                         const framework::DDim& dims, int length);

 private:
  const framework::Scope* scope_;
  const platform::DeviceContext* dev_ctx_;
  bool create_scope_ = false;
  framework::Scope* local_scope_ = nullptr;
  // only Skeleton
  sendrecv::VariableMessage meta_;
};

};  // namespace distributed
};  // namespace operators
};  // namespace paddle
