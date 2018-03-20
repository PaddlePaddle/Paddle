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
#include <sys/time.h>
#include <thread>
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/operators/detail/bytebuffer_stream.h"
#include "paddle/fluid/operators/detail/proto_encoder_helper.h"
#include "paddle/fluid/operators/detail/tensor_parser.h"

namespace paddle {
namespace operators {
namespace detail {

// TODO(gongwb): from bytebuffer directedly.
void DeserializeLodTendor(const sendrecv::VariableMessage& msg,
                          const platform::DeviceContext& ctx,
                          framework::Variable* var, framework::DDim& dims) {
  auto* tensor = var->GetMutable<framework::LoDTensor>();
  tensor->Resize(dims);
  void* tensor_data = tensor->mutable_data(
      ctx.GetPlace(), paddle::operators::detail::ToTypeIndex(msg.data_type()));

  framework::LoD lod;
  for (int i = 0; i < msg.lod_level(); ++i) {
    framework::Vector<size_t> v;
    for (int j = 0; j < msg.lod(i).lod_data_size(); ++j) {
      v.push_back(msg.lod(i).lod_data(j));
    }
    lod.push_back(v);
  }
  tensor->set_lod(lod);
  // How to avoid copying and use the message buffer directly?
  // Maybe need to find a way to release all memory except tensor content.
  if (platform::is_gpu_place(ctx.GetPlace())) {
#ifdef PADDLE_WITH_CUDA
    platform::CPUPlace cpu;
    auto& gpu_dev_ctx = static_cast<const platform::CUDADeviceContext&>(ctx);
    memory::Copy(boost::get<platform::CUDAPlace>(tensor->place()), tensor_data,
                 cpu, reinterpret_cast<const void*>(msg.serialized().data()),
                 msg.serialized().size(), gpu_dev_ctx.stream());
    ctx.Wait();
#endif
  } else {
    memcpy(tensor_data, reinterpret_cast<const void*>(msg.serialized().data()),
           msg.serialized().size());
  }
}

// TODO(gongwb): from bytebuffer directedly.
void DeserializeSelectedRows(const sendrecv::VariableMessage& msg,
                             const platform::DeviceContext& ctx,
                             framework::Variable* var, framework::DDim& dims) {
  auto* slr = var->GetMutable<framework::SelectedRows>();
  auto* tensor = slr->mutable_value();
  int64_t* rows_data = slr->mutable_rows()->data();
  tensor->Resize(dims);
  void* tensor_data = tensor->mutable_data(
      ctx.GetPlace(), paddle::operators::detail::ToTypeIndex(msg.data_type()));
  if (platform::is_gpu_place(ctx.GetPlace())) {
#ifdef PADDLE_WITH_CUDA
    platform::CPUPlace cpu;
    auto& gpu_dev_ctx = static_cast<const platform::CUDADeviceContext&>(ctx);
    memory::Copy(boost::get<platform::CUDAPlace>(tensor->place()), tensor_data,
                 cpu, reinterpret_cast<const void*>(msg.serialized().data()),
                 msg.serialized().size(), gpu_dev_ctx.stream());
    ctx.Wait();
#endif
  } else {
    memcpy(tensor_data, reinterpret_cast<const void*>(msg.serialized().data()),
           msg.serialized().size());
  }

  // copy rows CPU data, GPU data will be copied lazly
  memcpy(rows_data, reinterpret_cast<const void*>(msg.rows().data()),
         msg.rows().size());
}

void DeserializeFromMessage(const sendrecv::VariableMessage& msg,
                            const platform::DeviceContext& ctx,
                            framework::Variable* var) {
  // dims is needed by both tensor and selectedrows
  std::vector<int> vecdims;
  for (auto& d : msg.dims()) {
    vecdims.push_back(d);
  }
  framework::DDim dims = framework::make_ddim(vecdims);

  if (msg.type() == sendrecv::LOD_TENSOR) {
    struct timeval t0_wait, t1_wait;
    gettimeofday(&t0_wait, 0);

    DeserializeLodTendor(msg, ctx, var, dims);

    std::thread::id this_id = std::this_thread::get_id();
    gettimeofday(&t1_wait, 0);
    double t_wait = double((t1_wait.tv_sec - t0_wait.tv_sec) * 1000.0 +
                           (t1_wait.tv_usec - t0_wait.tv_usec) / 1000.0);
    std::stringstream ss;
    ss << "from message var_name:" << msg.varname()
       << ", data length:" << msg.serialized().size() << ", time:" << t_wait
       << "ms, thread_id:" << this_id;
    std::cout << ss.str() << '\n';

    return;
  } else if (msg.type() == sendrecv::SELECTED_ROWS) {
    DeserializeSelectedRows(msg, ctx, var, dims);
    return;
  }

  PADDLE_ENFORCE(false, "must be LOD_TENSOR or SELECTED_ROWS");
}

void SerializeToByteBuffer(const std::string& name, framework::Variable* var,
                           const platform::DeviceContext& ctx,
                           ::grpc::ByteBuffer* msg) {
  using VarMsg = sendrecv::VariableMessage;
  sendrecv::VariableMessage request;
  std::string header;
  request.AppendToString(&header);
  // When using GPU, need to free the copied CPU buffer
  // when the ByteBuffer destroies
  // TODO(typhoonzero): add unref here, if we have dependent
  // parallelism execution, need to know when to free the tensor.
  DestroyCallback destroy_callback = [](void* backing) {};

  void* buf = malloc(1024);
  void* payload = nullptr;
  size_t payload_size;
  ProtoEncodeHelper e((char*)buf, 1024);
  e.WriteString(VarMsg::kVarnameFieldNumber, name);
  if (var->IsType<framework::LoDTensor>()) {
    e.WriteUint64(VarMsg::kTypeFieldNumber, 0);
  } else if (var->IsType<framework::SelectedRows>()) {
    e.WriteUint64(VarMsg::kTypeFieldNumber, 1);
  }

  switch (framework::ToVarType(var->Type())) {
    case framework::proto::VarType_Type_LOD_TENSOR: {
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
        struct timeval t0_wait, t1_wait;
        gettimeofday(&t0_wait, 0);
        std::thread::id this_id = std::this_thread::get_id();

        PADDLE_ENFORCE(platform::is_gpu_place(tensor.place()));
        platform::CPUPlace cpu;
        auto& gpu_dev_ctx =
            static_cast<const platform::CUDADeviceContext&>(ctx);
        auto copy_size = tensor.memory_size();
        payload = memory::Alloc(cpu, copy_size);

        gettimeofday(&t1_wait, 0);
        double t_wait = double((t1_wait.tv_sec - t0_wait.tv_sec) * 1000.0 +
                               (t1_wait.tv_usec - t0_wait.tv_usec) / 1000.0);

        std::stringstream ss;
        ss << "se malloc var_name:" << name << ", dims: " << tensor.dims()
           << ", time:" << t_wait << "ms, thread_id:" << this_id;
        std::cout << ss.str() << '\n';

        memory::Copy(cpu, payload,
                     boost::get<platform::CUDAPlace>(tensor.place()),
                     reinterpret_cast<const void*>(tensor.data<void>()),
                     copy_size, gpu_dev_ctx.stream());
        ctx.Wait();
        destroy_callback = [](void* backing) {
          platform::CPUPlace cpu;
          memory::Free(cpu, backing);
        };
        gettimeofday(&t1_wait, 0);
        t_wait = double((t1_wait.tv_sec - t0_wait.tv_sec) * 1000.0 +
                        (t1_wait.tv_usec - t0_wait.tv_usec) / 1000.0);
        std::stringstream ss2;
        ss2 << "se memcpy gpu var_name:" << name << ", dims: " << tensor.dims()
            << ", time:" << t_wait << "ms, thread_id:" << this_id;
        std::cout << ss2.str() << '\n';

#endif
      } else {
        payload = tensor.data<void>();
      }
      payload_size = tensor.memory_size();
      e.WriteVarlengthBeginning(VarMsg::kSerializedFieldNumber, payload_size);
    } break;
    case framework::proto::VarType_Type_SELECTED_ROWS: {
      // TODO(typhoonzero): selectedrows implement should not use unique_ptr
      auto* slr = var->GetMutable<framework::SelectedRows>();
      e.WriteUint64(VarMsg::kDataTypeFieldNumber,
                    framework::ToDataType(slr->value().type()));
      for (auto& dim : framework::vectorize(slr->value().dims())) {
        e.WriteUint64(VarMsg::kDimsFieldNumber, dim);
      }
      e.WriteUint64(VarMsg::kLodLevelFieldNumber, 0);
      auto* tensor = slr->mutable_value();
      if (platform::is_gpu_place(ctx.GetPlace())) {
#ifdef PADDLE_WITH_CUDA
        platform::CPUPlace cpu;
        auto& gpu_dev_ctx =
            static_cast<const platform::CUDADeviceContext&>(ctx);
        auto copy_size = tensor->memory_size();
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
      payload_size = tensor->memory_size();
      e.WriteVarlengthBeginning(VarMsg::kSerializedFieldNumber, payload_size);
    } break;
    default:
      PADDLE_THROW("Serialize does not support type: %s",
                   typeid(var->Type()).name());
      break;
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

    ProtoEncodeHelper e2((char*)buf, 128);
    // NOTE: rows is of type int64_t
    size_t rows_memory_size =
        slr->rows().capacity() * framework::SizeOfType(typeid(int64_t));
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
                               framework::Variable*& var) {
  operators::detail::TensorResponse resp(scope, &ctx);
  PADDLE_ENFORCE(resp.Parse(msg) == 0, "parse bytebuffer to tensor error!");
  var = resp.GetVar();
}

}  // namespace detail
}  // namespace operators
}  // namespace paddle
