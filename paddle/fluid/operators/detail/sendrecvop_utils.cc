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

#include "paddle/fluid/operators/detail/sendrecvop_utils.h"

#include <nccl.h>
#include <sys/time.h>
#include <thread>  // NOLINT

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/operators/detail/bytebuffer_stream.h"
#include "paddle/fluid/operators/detail/proto_encoder_helper.h"
#include "paddle/fluid/operators/detail/variable_response.h"

namespace paddle {
namespace operators {
namespace detail {

void SerializeToByteBuffer(const std::string& name, framework::Variable* var,
                           const platform::DeviceContext& ctx,
                           ::grpc::ByteBuffer* msg,
                           const std::string& out_name) {
  using VarMsg = sendrecv::VariableMessage;
  // When using GPU, need to free the copied CPU buffer
  // when the ByteBuffer destroies
  // TODO(typhoonzero): add unref here, if we have dependent
  // parallelism execution, need to know when to free the tensor.
  DestroyCallback destroy_callback = [](void* backing) {};

  auto buffer = std::unique_ptr<char[]>(new char[1024]);
  void* buf = buffer.get();

  void* payload = nullptr;
  size_t payload_size = 0;
  ProtoEncodeHelper e(static_cast<char*>(buf), 1024);
  e.WriteString(VarMsg::kVarnameFieldNumber, name);
  if (var->IsType<framework::LoDTensor>()) {
    e.WriteUint64(VarMsg::kTypeFieldNumber, 0);
  } else if (var->IsType<framework::SelectedRows>()) {
    e.WriteUint64(VarMsg::kTypeFieldNumber, 1);
  } else if (var->IsType<ncclUniqueId>()) {
    // NOTE: sendrecv only support RAW type for NCCL_ID
    VLOG(3) << "serilizing: setting var type nccl id";
    e.WriteUint64(VarMsg::kTypeFieldNumber, 2);
  }

  if (!out_name.empty()) {
    e.WriteString(VarMsg::kOutVarnameFieldNumber, out_name);
  }
  if (var->IsType<framework::LoDTensor>()) {
    // ===========================Tensor==================================
    auto tensor = var->Get<framework::LoDTensor>();
    e.WriteUint64(VarMsg::kDataTypeFieldNumber,
                  framework::ToDataType(tensor.type()));
    for (auto& dim : framework::vectorize(tensor.dims())) {
      e.WriteUint64(VarMsg::kDimsFieldNumber, dim);
    }
    auto lod = tensor.lod();  // std::vector<Vector<size_t>>
    if (lod.size() > 0) {
      e.WriteUint64(VarMsg::kLodLevelFieldNumber, lod.size());

      for (auto& each : lod) {
        e.WriteVarlengthBeginning(VarMsg::kLodFieldNumber,
                                  2 +      // tag + varintlength of submessage
                                      1 +  // kLodDataFieldNumber
                                      each.size());
        // auto copied from GPU
        for (auto& d : each) {
          e.WriteUint64(VarMsg::LodData::kLodDataFieldNumber, d);
        }
      }
    }
    if (platform::is_gpu_place(ctx.GetPlace())) {
#ifdef PADDLE_WITH_CUDA
      PADDLE_ENFORCE(platform::is_gpu_place(tensor.place()));
      platform::CPUPlace cpu;
      auto& gpu_dev_ctx = static_cast<const platform::CUDADeviceContext&>(ctx);
      auto copy_size = tensor.numel() * framework::SizeOfType(tensor.type());
      payload = memory::Alloc(cpu, copy_size);

      memory::Copy(cpu, payload,
                   boost::get<platform::CUDAPlace>(tensor.place()),
                   reinterpret_cast<const void*>(tensor.data<void>()),
                   copy_size, gpu_dev_ctx.stream());
      ctx.Wait();
      destroy_callback = [](void* backing) {
        platform::CPUPlace cpu;
        memory::Free(cpu, backing);
      };

#endif
    } else {
      payload = tensor.data<void>();
    }
    payload_size = tensor.numel() * framework::SizeOfType(tensor.type());
    e.WriteVarlengthBeginning(VarMsg::kSerializedFieldNumber, payload_size);
  } else if (var->IsType<framework::SelectedRows>()) {
    // ===========================SELECTED
    // ROWS==================================
    // TODO(typhoonzero): selectedrows implement should not use unique_ptr
    auto* slr = var->GetMutable<framework::SelectedRows>();
    e.WriteUint64(VarMsg::kDataTypeFieldNumber,
                  framework::ToDataType(slr->value().type()));
    for (auto& dim : framework::vectorize(slr->value().dims())) {
      e.WriteUint64(VarMsg::kDimsFieldNumber, dim);
    }
    e.WriteUint64(VarMsg::kLodLevelFieldNumber, 0);
    e.WriteUint64(VarMsg::kSlrHeightFieldNumber, slr->height());
    auto* tensor = slr->mutable_value();
    if (platform::is_gpu_place(ctx.GetPlace())) {
#ifdef PADDLE_WITH_CUDA
      platform::CPUPlace cpu;
      auto& gpu_dev_ctx = static_cast<const platform::CUDADeviceContext&>(ctx);
      auto copy_size = tensor->numel() * framework::SizeOfType(tensor->type());
      payload = memory::Alloc(cpu, copy_size);
      memory::Copy(cpu, payload,
                   boost::get<platform::CUDAPlace>(tensor->place()),
                   reinterpret_cast<const void*>(tensor->data<void>()),
                   copy_size, gpu_dev_ctx.stream());
      ctx.Wait();
      destroy_callback = [](void* backing) {
        platform::CPUPlace cpu;
        memory::Free(cpu, backing);
      };
#endif
    } else {
      payload = slr->mutable_value()->data<void>();
    }
    payload_size = tensor->numel() * framework::SizeOfType(tensor->type());
    e.WriteVarlengthBeginning(VarMsg::kSerializedFieldNumber, payload_size);
  } else if (var->IsType<ncclUniqueId>()) {
    // ===========================NCCL ID==================================
    e.WriteVarlengthBeginning(VarMsg::kSerializedFieldNumber,
                              NCCL_UNIQUE_ID_BYTES);
    ncclUniqueId* uid = var->GetMutable<ncclUniqueId>();
    e.WriteRawBytes(std::string(uid->internal, NCCL_UNIQUE_ID_BYTES));
  } else {
    PADDLE_THROW("Serialize does not support type: %s",
                 typeid(var->Type()).name());
  }

  if (var->IsType<ncclUniqueId>()) {
    // for serialize NCCL_ID
    ::grpc::Slice slices(e.size());
    memcpy(const_cast<uint8_t*>(slices.begin()), e.data(), e.size());
    ::grpc::ByteBuffer tmp(&slices, 1);
    msg->Swap(&tmp);
    return;
  }

  // steal reference of tensor data
  ::grpc::Slice slices[4];  // metadata, tensor, rows meta, rows
  int num_slices = 2;       // only SelectedRows have rows buffer
  slices[0] = ::grpc::Slice(e.size());
  memcpy(const_cast<uint8_t*>(slices[0].begin()), e.data(), e.size());
  slices[1] = ::grpc::Slice(
      grpc_slice_new_with_user_data(payload, payload_size, destroy_callback,
                                    static_cast<char*>(payload)),
      ::grpc::Slice::STEAL_REF);

  if (framework::ToVarType(var->Type()) ==
      framework::proto::VarType_Type_SELECTED_ROWS) {
    auto* slr = var->GetMutable<framework::SelectedRows>();

    ProtoEncodeHelper e2(static_cast<char*>(buf), 128);
    // NOTE: rows is of type int64_t
    size_t rows_memory_size =
        slr->rows().size() * framework::SizeOfType(typeid(int64_t));
    e2.WriteVarlengthBeginning(VarMsg::kRowsFieldNumber, rows_memory_size);
    slices[2] = ::grpc::Slice(e2.size());
    memcpy(const_cast<uint8_t*>(slices[2].begin()), e2.data(), e2.size());

    slices[3] = ::grpc::Slice(
        grpc_slice_new_with_user_data(
            const_cast<void*>(
                reinterpret_cast<const void*>(slr->rows().data())),
            rows_memory_size,
            [](void* backing) {
              // TODO(typhoonzero): add unref here, same as above.
            },
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
                               framework::Variable** var) {
  operators::detail::VariableResponse resp(scope, &ctx);
  PADDLE_ENFORCE(resp.Parse(msg) == 0, "parse bytebuffer to tensor error!");
  *var = resp.GetVar();
}

}  // namespace detail
}  // namespace operators
}  // namespace paddle
