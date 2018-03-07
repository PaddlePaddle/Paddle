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
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/operators/detail/bytebuffer_stream.h"
#include "paddle/fluid/operators/detail/proto_encoder_helper.h"

namespace paddle {
namespace operators {
namespace detail {

void SerializeToMessage(const std::string& name, const framework::Variable* var,
                        const platform::DeviceContext& ctx,
                        sendrecv::VariableMessage* msg) {
  msg->set_varname(name);
  std::ostringstream oss;
  switch (framework::ToVarType(var->Type())) {
    case framework::proto::VarType_Type_LOD_TENSOR:
      msg->set_type(sendrecv::VarType::LOD_TENSOR);
      framework::SerializeToStream(oss, var->Get<framework::LoDTensor>(), ctx);
      break;
    case framework::proto::VarType_Type_SELECTED_ROWS:
      msg->set_type(sendrecv::VarType::SELECTED_ROWS);
      framework::SerializeToStream(oss, var->Get<framework::SelectedRows>(),
                                   ctx);
      break;
    default: {
      PADDLE_THROW("Serialize does not support type: %s",
                   typeid(var->Type()).name());
      break;
    }
  }
  msg->set_serialized(oss.str());
}

void DeserializeFromMessage(const sendrecv::VariableMessage& msg,
                            const platform::DeviceContext& ctx,
                            framework::Variable* var) {
  std::istringstream iss(msg.serialized());
  switch (msg.type()) {
    case sendrecv::VarType::LOD_TENSOR:
      DeserializeFromStream(iss, var->GetMutable<framework::LoDTensor>(), ctx);
      break;
    case sendrecv::VarType::SELECTED_ROWS: {
      DeserializeFromStream(iss, var->GetMutable<framework::SelectedRows>(),
                            ctx);
      break;
    }
    default: {
      PADDLE_THROW("Deserialize does not support type: %s",
                   typeid(var->Type()).name());
      break;
    }
  }
}

void SerializeToByteBuffer(const std::string& name, framework::Variable* var,
                           const platform::DeviceContext& ctx,
                           ::grpc::ByteBuffer* msg) {
  sendrecv::VariableMessage request;
  std::string header;
  request.AppendToString(&header);
  void* buf = malloc(1024);
  void* payload;
  size_t payload_size;
  ProtoEncodeHelper e((char*)buf, 1024);
  e.WriteString(1, name);
  if (var->IsType<framework::LoDTensor>()) {
    e.WriteUint64(2, 0);
  } else if (var->IsType<framework::SelectedRows>()) {
    e.WriteUint64(2, 1);
  }

  switch (framework::ToVarType(var->Type())) {
    case framework::proto::VarType_Type_LOD_TENSOR: {
      auto tensor = var->Get<framework::LoDTensor>();
      e.WriteUint64(3, framework::ToDataType(tensor.type()));
      for (auto& dim : framework::vectorize(tensor.dims())) {
        e.WriteUint64(4, dim);
      }
      auto lod = tensor.lod();  // std::vector<Vector<size_t>>
      if (lod.size() > 0) {
        e.WriteUint64(5, lod.size());
        for (auto& each : lod) {
          e.WriteVarlengthBeginning(6, each.size());
          // auto copy from GPU
          e.WriteRawBytes(*reinterpret_cast<const std::string*>(each.data()));
        }
      }
      // TODO(typhoonzero): Copy from GPU if needed
      payload = tensor.data<void>();
      payload_size = tensor.memory_size();
      e.WriteVarlengthBeginning(7, payload_size);
    } break;
    case framework::proto::VarType_Type_SELECTED_ROWS: {
      // TODO(typhoonzero): selectedrows implement should not use unique_ptr
      auto* slr = var->GetMutable<framework::SelectedRows>();
      e.WriteUint64(3, framework::ToDataType(slr->value().type()));
      for (auto& dim : framework::vectorize(slr->GetCompleteDims())) {
        e.WriteUint64(4, dim);
      }
      e.WriteUint64(5, 0);
      // TODO(typhoonzero): Copy from GPU if needed
      payload = slr->mutable_value()->data<void>();
      payload_size = slr->value().memory_size();
      e.WriteVarlengthBeginning(7, payload_size);
    } break;
    default:
      PADDLE_THROW("Serialize does not support type: %s",
                   typeid(var->Type()).name());
      break;
  }
  // steal reference of tensor data
  ::grpc::Slice slices[4];  // metadata, tensor, rows
  int num_slices = 2;       // only SelectedRows have rows buffer
  slices[0] = ::grpc::Slice(e.size());
  memcpy(const_cast<uint8_t*>(slices[0].begin()), e.data(), e.size());
  slices[1] =
      ::grpc::Slice(grpc_slice_new_with_user_data(payload, payload_size,
                                                  [](void* backing) {
                                                    // TODO(typhoonzero): add
                                                    // unref here, if we have
                                                    // dependent
                                                    // parallelism execution,
                                                    // need to know when to free
                                                    // the tensor.
                                                  },
                                                  static_cast<char*>(payload)),
                    ::grpc::Slice::STEAL_REF);

  if (framework::ToVarType(var->Type()) ==
      framework::proto::VarType_Type_SELECTED_ROWS) {
    auto* slr = var->GetMutable<framework::SelectedRows>();

    ProtoEncodeHelper e2((char*)buf, 128);
    // NOTE(typhoonzero): rows is of type int64_t
    size_t rows_memory_size =
        slr->rows().capacity() * framework::SizeOfType(typeid(int64_t));
    e2.WriteVarlengthBeginning(8, rows_memory_size);
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
                               framework::Variable* var) {
  sendrecv::VariableMessage meta;
  GrpcByteBufferSource source;
  source.Init(msg);
  ::google::protobuf::io::CodedInputStream input(&source);
  // do zerocopy parsing
  PADDLE_ENFORCE(meta.ParseFromCodedStream(&input));
  PADDLE_ENFORCE(input.ConsumedEntireMessage());
  if (meta.type() == sendrecv::LOD_TENSOR) {
    auto* tensor = var->GetMutable<framework::LoDTensor>();
    std::vector<int> vecdims;
    for (auto& d : meta.dims()) {
      vecdims.push_back(d);
    }
    framework::DDim dims = framework::make_ddim(vecdims);
    tensor->Resize(dims);
    void* tensor_data = tensor->mutable_data(
        ctx.GetPlace(),
        paddle::operators::detail::ToTypeIndex(meta.data_type()));
    framework::LoD lod;
    for (int i = 0; i < meta.lod_level(); ++i) {
      // lod elements are of type size_t
      int elem_num = meta.lod(i).size() / sizeof(size_t);
      std::string vvv;
      const size_t* begin = reinterpret_cast<const size_t*>(meta.lod(i).data());
      for (int j = 0; j < elem_num; ++j) {
        lod[i].push_back(begin[j]);
      }
    }
    // How to avoid copying and use the message buffer directly?
    // Maybe need to find a way to release all memory except tensor content.
    if (platform::is_gpu_place(ctx.GetPlace())) {
#ifdef PADDLE_WITH_CUDA
// do GPU copy here.
#endif
    } else {
      memcpy(tensor_data,
             reinterpret_cast<const void*>(meta.serialized().data()),
             meta.serialized().size());
    }
  }
}

}  // namespace detail
}  // namespace operators
}  // namespace paddle