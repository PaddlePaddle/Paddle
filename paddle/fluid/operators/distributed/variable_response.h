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

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream.h"

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/operators/distributed/distributed_pb.h"

DECLARE_string(rpc_server_profile_path);

namespace paddle {
namespace operators {
namespace distributed {

// Source provides a way for a particular RPC implementation to provide
// received data to ParseFrom.
class Source {
 public:
  virtual ~Source() {}

  // Return the stream that contains the data to be parsed.
  // Note that this method might be invoked more than once if
  // ParseFrom needs to fall back to a more expensive parsing method.
  // Every call must return a stream pointing at the beginning of
  // the serialized RecvTensorResponse.
  //
  // Note that a subsequent call to contents() invalidates previous
  // results of contents().
  //
  // Ownership of the returned stream is retained by the Source and
  // should not be deleted by the caller.
  virtual ::google::protobuf::io::ZeroCopyInputStream* contents() = 0;
};

typedef std::function<framework::Variable*(const std::string& varname)>
    GetVarCallback;

class VariableResponse {
 public:
  VariableResponse(framework::Variable* var,
                   const platform::DeviceContext* dev_ctx)
      : dev_ctx_(dev_ctx), var_cache_(var) {}
  // for cases that can not determin varname before head, use a callback
  VariableResponse(GetVarCallback get_var_callback,
                   const platform::DeviceContext* dev_ctx)
      : dev_ctx_(dev_ctx), get_var_callback_(get_var_callback) {}

  virtual ~VariableResponse() {}

  int Parse(Source* source, const sendrecv::VariableMessage& meta) {
    meta_ = meta;
    return Parse(source);
  }

  // return:
  // 0:ok.
  // -1: unkown error.
  // other: number of error field.
  virtual int Parse(Source* source) = 0;

  inline std::string Varname() const { return meta_.varname(); }
  inline std::string OutVarname() const { return meta_.out_varname(); }
  inline std::string TableName() const { return meta_.table_name(); }

  // should call parse first.
  framework::Variable* GetVar() {
    if (LIKELY(var_cache_)) {
      return var_cache_;
    }
    PADDLE_ENFORCE_NOT_NULL(get_var_callback_);
    var_cache_ = get_var_callback_(meta_.varname());
    return var_cache_;
  }

  int GetTrainerId() { return static_cast<int>(meta_.trainer_id()); }

 protected:
  bool ReadRaw(::google::protobuf::io::CodedInputStream* input,
               const platform::DeviceContext& dev_ctx, platform::Place place,
               void* dest, int64_t size);

  bool CopySelectRowsTensorData(::google::protobuf::io::CodedInputStream* input,
                                const platform::DeviceContext& ctx,
                                const framework::DDim& dims, int length);

  bool CopySelectRowsData(::google::protobuf::io::CodedInputStream* input,
                          const platform::DeviceContext& ctx, int length);

  bool CopyLodTensorData(::google::protobuf::io::CodedInputStream* input,
                         const platform::DeviceContext& ctx,
                         const framework::DDim& dims, int length);

  bool ProcSerializedField(int tag,
                           ::google::protobuf::io::CodedInputStream* input,
                           int64_t num_bytes);

 protected:
  const framework::Scope* scope_;
  const platform::DeviceContext* dev_ctx_;

  framework::Variable* var_cache_ = nullptr;
  GetVarCallback get_var_callback_;

  sendrecv::VariableMessage meta_;
};

};  // namespace distributed
};  // namespace operators
};  // namespace paddle
