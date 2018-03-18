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

namespace paddle {
namespace operators {
namespace detail {

class TensorResponse {
 public:
  TensorResponse(framework::Scope* scope) : scope_(scope){};

  virtual ~TensorResponse(){};

  // return:
  // 0:ok.
  // -1: unkown error.
  // other: number of error field.
  int TensorResponse::Parse(::grpc::ByteBuffer& byte_buffer);

  // should call parse first.
  framework::Variable* GetVar() { return scope_.FindVar(meta_.varname()); }

 private:
  /*
  int TensorResponse::ParseTensorSubmessage(
      protobuf::io::CodedInputStream* input,
      TensorProto* tensor_meta);
      */

  framework::Scope* scope_;
  // only Skeleton
  sendrecv::VariableMessage meta_;
}

};  // namespace detail
};  // namespace operators
};  // namespace paddle
