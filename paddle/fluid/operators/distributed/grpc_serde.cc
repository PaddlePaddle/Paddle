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

#ifdef PADDLE_WITH_CUDA
#include <nccl.h>
#endif
#include <sys/time.h>
#include <thread>  // NOLINT

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/operators/distributed/grpc_bytebuffer_stream.h"
#include "paddle/fluid/operators/distributed/grpc_serde.h"
#include "paddle/fluid/operators/distributed/grpc_variable_response.h"
#include "paddle/fluid/operators/distributed/proto_encoder_helper.h"
#include "paddle/fluid/operators/distributed/sendrecvop_utils.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace operators {
namespace distributed {

void SerializeToByteBuffer(const std::string& name, framework::Variable* var,
                           const platform::DeviceContext& ctx,
                           ::grpc::ByteBuffer* msg, const std::string& out_name,
                           const int trainer_id) {
  platform::RecordRPCEvent record_event("serial", &ctx);
  // Default DestroyCallback does nothing, When using GPU
  // the CPU buffer need to be freed.
  DestroyCallback destroy_callback = [](void* backing) {};
  VarMsg request;
  void* payload = nullptr;
  size_t payload_size;

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
  if (var->IsType<framework::LoDTensor>()) {
    request.set_type(::sendrecv::LOD_TENSOR);
    GetTensorPayload(var, ctx, &request, &payload, &payload_size);
  } else if (var->IsType<framework::SelectedRows>()) {
    request.set_type(::sendrecv::SELECTED_ROWS);
    GetSelectedRowsPayload(var, ctx, &request, &payload, &payload_size);
#ifdef PADDLE_WITH_CUDA
  } else if (var->IsType<ncclUniqueId>()) {
    request.set_type(::sendrecv::NCCL_ID);
#endif
  } else {
    PADDLE_THROW("Serialize does not support type: %s",
                 typeid(var->Type()).name());
  }

  if (platform::is_gpu_place(ctx.GetPlace())) {
#ifdef PADDLE_WITH_CUDA
    // GPU data is copied to CPU buffer when sending,
    // free the buffer when possible.
    destroy_callback = [](void* backing) {
      platform::CUDAPinnedPlace cuda_pinned;
      memory::Free(cuda_pinned, backing);
    };
#endif
  }

  std::string header;
  request.AppendToString(&header);
  auto buffer = std::unique_ptr<char[]>(new char[1024]);
  void* buf = buffer.get();
  ProtoEncodeHelper e(static_cast<char*>(buf), 1024);
  e.WriteRawBytes(std::string(header.data(), header.size()));
// NCCLID is copied directly to the message, return bytebuffer
// with only one slice if serializing NCCLID.
#ifdef PADDLE_WITH_CUDA
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

  e.WriteVarlengthBeginning(VarMsg::kSerializedFieldNumber, payload_size);
  // steal reference of tensor data
  ::grpc::Slice slices[4];  // metadata, tensor, rows meta, rows
  int num_slices = 2;       // only SelectedRows have rows buffer
  slices[0] = ::grpc::Slice(e.size());
  memcpy(const_cast<uint8_t*>(slices[0].begin()), e.data(), e.size());
  slices[1] = ::grpc::Slice(
      grpc_slice_new_with_user_data(payload, payload_size, destroy_callback,
                                    static_cast<char*>(payload)),
      ::grpc::Slice::STEAL_REF);

  if (var->IsType<framework::SelectedRows>()) {
    auto* slr = var->GetMutable<framework::SelectedRows>();
    ProtoEncodeHelper e2(static_cast<char*>(buf), 128);
    size_t rows_memory_size =
        slr->rows().size() * framework::SizeOfType(typeid(int64_t));
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

void DeserializeFromByteBuffer(const ::grpc::ByteBuffer& msg,
                               const platform::DeviceContext& ctx,
                               const framework::Scope* scope,
                               framework::Variable** var, int* trainer_id) {
  platform::RecordRPCEvent record_event("deserial", &ctx);
  operators::distributed::GRPCVariableResponse resp(scope, &ctx);
  PADDLE_ENFORCE(resp.Parse(msg) == 0, "parse bytebuffer to tensor error!");
  *var = resp.GetVar();
  *trainer_id = resp.GetTrainerId();
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
