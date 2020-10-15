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

#ifdef PADDLE_WITH_NCCL
#include <nccl.h>
#endif
#include <limits>
#include <memory>
#include "grpcpp/impl/codegen/byte_buffer.h"
#include "grpcpp/impl/codegen/slice.h"
#include "paddle/fluid/operators/distributed/grpc/grpc_serde.h"
#include "paddle/fluid/operators/distributed/grpc/grpc_variable_response.h"
#include "paddle/fluid/operators/distributed/proto_encoder_helper.h"
#include "paddle/fluid/operators/distributed/send_recv.pb.h"
#include "paddle/fluid/operators/distributed/sendrecvop_utils.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace framework {
class Scope;
class Variable;
}  // namespace framework
namespace platform {
class DeviceContext;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {
namespace distributed {

void SerializeMultiVarToByteBuffer(
    const std::string& message_name,
    const std::vector<std::string>& send_var_name,
    const framework::Scope& scope, const platform::DeviceContext& ctx,
    const std::vector<std::string>& out_var_name, const int trainer_id,
    const std::string& table_name, const MultiVarMsg* request) {
  request.set_message_name = message_name;
  request.set_var_num = send_var_name.shape();

  for (size_t var_index = 0; var_index < send_var_name.shape(); var_index++) {
    auto* var_msg = request.add_vars();
    SerializeLodTensorToVarMsg(send_var_name[var_index], scope, ctx, trainer_id,
                               var_msg);
  }
}

void SerializeLodTensorToVarMsg(const std::string& var_name,
                                const framework::Scope& scope,
                                const platform::DeviceContext& ctx,
                                const int trainer_id,
                                VariableMessage* var_msg) {
  Variable* var = scope->FindVar(varname);
  if (var == nullptr) {
    // throw error
    return;
  }

  LoDTensor* tensor = var->GetMutable<LoDTensor>();
  var_msg->set_varname(var_name);
  var_msg->set_trainer_id(trainer_id);
  var_msg->set_type(::sendrecv::LOD_TENSOR);

  for (auto& dim : framework::vectorize(tensor->dims())) {
    var_msg->add_dims(dim);
  }
  const framework::LoD lod = tensor->lod();
  if (lod.size() > 0) {
    var_msg->set_lod_level(lod.size());
    for (auto& each : lod) {
      VariableMessage::LodData* lod_inner = var_msg->add_lod();
      for (auto& d : each) {
        lod_inner->add_lod_data(d);
      }
    }
  }

  auto* req_data = var_msg->mutable_data();
  req_data->clear();
  req_data->resize(tensor->numel() * SizeOfType(tensor->type()));
  char* data_ptr = const_cast<char*>(req_data->data());

  if (platform::is_cpu_place(tensor->place())) {
    memcpy(data_ptr, tensor->data<void>(),
           tensor->numel() * SizeOfType(tensor->type()));
  } else {
#ifdef PADDLE_WITH_CUDA
    memory::Copy(platform::CPUPlace(), data_ptr,
                 BOOST_GET_CONST(platform::CUDAPlace, tensor->place()),
                 tensor->data<void>(),
                 tensor->numel() * SizeOfType(tensor->type()), nullptr);
#endif
#ifdef PADDLE_WITH_XPU
    memory::Copy(platform::CPUPlace(), data_ptr,
                 BOOST_GET_CONST(platform::XPUPlace, tensor->place()),
                 tensor->data<void>(),
                 tensor->numel() * SizeOfType(tensor->type()));
#endif
  }
}

void SerializeToByteBuffer(const std::string& name, framework::Variable* var,
                           const platform::DeviceContext& ctx,
                           ::grpc::ByteBuffer* msg, const std::string& out_name,
                           const int trainer_id,
                           const std::string& table_name) {
  platform::RecordRPCEvent record_event("serial");
  VarMsg request;
  TensorPayload* payload = nullptr;

  request.set_varname(name);
  request.set_trainer_id(trainer_id);
  // Note: normally the profiler is enabled in 1 trainer, hence only
  // 1 trainer returns true for ShouldSendProfileState(). It tells PS
  // servers the trainer's profiling state so that PS can follow the
  // trainer.
  if (platform::ShouldSendProfileState()) {
    if (platform::IsProfileEnabled()) {
      request.set_profile(platform::kEnableProfiler);
    } else {
      request.set_profile(platform::kDisableProfiler);
    }
  }
  if (!out_name.empty()) {
    request.set_out_varname(out_name);
  }
  if (!table_name.empty()) {
    request.set_table_name(table_name);
  }
  if (var->IsType<framework::LoDTensor>()) {
    request.set_type(::sendrecv::LOD_TENSOR);
    payload = new TensorPayload(GetTensorPayload(var, ctx, &request));
  } else if (var->IsType<framework::SelectedRows>()) {
    request.set_type(::sendrecv::SELECTED_ROWS);
    payload = new TensorPayload(GetSelectedRowsPayload(var, ctx, &request));
#ifdef PADDLE_WITH_NCCL
  } else if (var->IsType<ncclUniqueId>()) {
    request.set_type(::sendrecv::NCCL_ID);
#endif
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Serialize does not support type: %s", typeid(var->Type()).name()));
  }
  std::string header;
  request.AppendToString(&header);
  auto buffer = std::unique_ptr<char[]>(new char[1024]);
  void* buf = buffer.get();
  ProtoEncodeHelper e(static_cast<char*>(buf), 1024);
  e.WriteRawBytes(std::string(header.data(), header.size()));
// NCCLID is copied directly to the message, return bytebuffer
// with only one slice if serializing NCCLID.
#ifdef PADDLE_WITH_NCCL
  if (var->IsType<ncclUniqueId>()) {
    e.WriteVarlengthBeginning(VarMsg::kSerializedFieldNumber,
                              NCCL_UNIQUE_ID_BYTES);
    const ncclUniqueId& uid = var->Get<ncclUniqueId>();
    e.WriteRawBytes(std::string(uid.internal, NCCL_UNIQUE_ID_BYTES));

    // for serialize NCCL_ID
    ::grpc::Slice slices(e.size());
    memcpy(const_cast<uint8_t*>(slices.begin()), e.data(), e.size());
    ::grpc::ByteBuffer tmp(&slices, 1);
    msg->Swap(&tmp);
    return;
  }
#endif
  PADDLE_ENFORCE_NOT_NULL(
      payload,
      platform::errors::InvalidArgument(
          "Not support type: %s, need to be LOD_TENSOR or SELECTED_ROWS",
          var->Type()));
  e.WriteVarlengthBeginning(VarMsg::kSerializedFieldNumber,
                            payload->memory_size());
  if (payload->memory_size() >= std::numeric_limits<int>::max()) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Variable %s length %d should less than %d.", name,
        payload->memory_size(), std::numeric_limits<int>::max()));
  }
  // steal reference of tensor data
  ::grpc::Slice slices[4];  // metadata, tensor, rows meta, rows
  int num_slices = 2;       // only SelectedRows have rows buffer
  slices[0] = ::grpc::Slice(e.size());
  memcpy(const_cast<uint8_t*>(slices[0].begin()), e.data(), e.size());
  slices[1] = ::grpc::Slice(
      grpc_slice_new_with_user_data(payload->ptr(), payload->memory_size(),
                                    SerializeDestroyCallback, payload),
      ::grpc::Slice::STEAL_REF);

  if (var->IsType<framework::SelectedRows>()) {
    auto* slr = var->GetMutable<framework::SelectedRows>();
    ProtoEncodeHelper e2(static_cast<char*>(buf), 128);

    PADDLE_ENFORCE_EQ(VectorElemName(slr->rows()), typeid(int64_t).name(),
                      platform::errors::InvalidArgument(
                          "Got wrong type %s, expect type: int64_t",
                          VectorElemName(slr->rows())));
    size_t rows_memory_size = slr->rows().size() * sizeof(int64_t);

    e2.WriteVarlengthBeginning(VarMsg::kRowsFieldNumber, rows_memory_size);
    slices[2] = ::grpc::Slice(e2.size());
    memcpy(const_cast<uint8_t*>(slices[2].begin()), e2.data(), e2.size());

    slices[3] = ::grpc::Slice(
        grpc_slice_new_with_user_data(
            const_cast<void*>(
                reinterpret_cast<const void*>(slr->rows().data())),
            rows_memory_size, [](void* backing) {},
            const_cast<char*>(
                reinterpret_cast<const char*>(slr->rows().data()))),
        ::grpc::Slice::STEAL_REF);
    num_slices = 4;
  }
  ::grpc::ByteBuffer tmp(&slices[0], num_slices);
  msg->Swap(&tmp);
}

void DeserialMultiVarFromByteBuffer() { return; }

void DeserializeFromByteBuffer(const ::grpc::ByteBuffer& msg,
                               const platform::DeviceContext& ctx,
                               const framework::Scope* scope,
                               framework::Variable** var, int* trainer_id) {
  platform::RecordRPCEvent record_event("deserial");
  operators::distributed::GRPCVariableResponse resp(scope, &ctx);
  PADDLE_ENFORCE_EQ(
      resp.Parse(msg), 0,
      platform::errors::InvalidArgument("parse bytebuffer to tensor error!"));
  *var = resp.GetVar();
  *trainer_id = resp.GetTrainerId();
}

void DeserializeRecvFromByteBuffer(const ::grpc::ByteBuffer& msg,
                                   const platform::DeviceContext& ctx,
                                   const framework::Scope* scope,
                                   framework::Variable** var, int* trainer_id) {
  platform::RecordRPCEvent record_event("deserial");
  operators::distributed::GRPCVariableResponse resp(scope, &ctx);
  PADDLE_ENFORCE_EQ(
      resp.Parse(msg), 0,
      platform::errors::InvalidArgument("parse bytebuffer to tensor error!"));
  *var = resp.GetRecvVar();
  *trainer_id = resp.GetTrainerId();
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
