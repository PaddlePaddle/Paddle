/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include <netdb.h>
#include <iostream>
#include <string>
#include <vector>

#include "brpc/channel.h"
#include "paddle/fluid/distributed/ps/service/sendrecv.pb.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows_utils.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/phi/backends/dynload/port.h"

namespace butil {
class IOBuf;
class IOBufBytesIterator;
}  // namespace butil

namespace grpc {
class ByteBuffer;
}  // namespace grpc
namespace paddle {
namespace framework {
class Scope;
class Variable;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace distributed {

using MultiVarMsg = ::paddle::distributed::MultiVariableMessage;
using VarMsg = ::paddle::distributed::VariableMessage;

void SerializeToMultiVarMsgAndIOBuf(
    const std::string& message_name,
    const std::vector<std::string>& send_var_name_val,
    const std::vector<std::string>& recv_var_name_val,
    const platform::DeviceContext& ctx, const framework::Scope* scope,
    MultiVarMsg* var_msg, butil::IOBuf* iobuf);

void SerializeLodTensor(framework::Variable* var,
                        const platform::DeviceContext& ctx, VarMsg* var_msg,
                        butil::IOBuf* iobuf);

void SerializeSelectedRows(framework::Variable* var,
                           const platform::DeviceContext& ctx, VarMsg* request,
                           butil::IOBuf* iobuf);

// Deserialize for Server
void DeserializeFromMultiVarMsgAndIOBuf(const MultiVarMsg& multi_msg,
                                        const butil::IOBuf* iobuf,
                                        const platform::DeviceContext& ctx,
                                        framework::Scope* scope);

// Deserialize for Client
void DeserializeFromMultiVarMsgAndIOBuf(const MultiVarMsg& multi_msg,
                                        const butil::IOBuf* iobuf,
                                        const platform::DeviceContext& ctx,
                                        const framework::Scope* scope);

void DeserializeLodTensor(framework::Variable* var, const VarMsg& msg,
                          butil::IOBufBytesIterator& iobuf,  // NOLINT
                          const platform::DeviceContext& ctx);

void DeserializeSelectedRows(framework::Variable* var, const VarMsg& msg,
                             butil::IOBufBytesIterator& iobuf,  // NOLINT
                             const platform::DeviceContext& ctx);

std::string GetIntTypeEndpoint(const std::string& ip, const uint32_t& port);

}  // namespace distributed
}  // namespace paddle
