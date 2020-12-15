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

#include "paddle/fluid/distributed/service/brpc_utils.h"
#include <limits>
#include <memory>
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
namespace distributed {

framework::proto::VarType::Type VarMessageToVarType(
    VariableMessage::Type type) {
  switch (type) {
    case VariableMessage::FP32:
      return framework::proto::VarType::FP32;  // NOLINT
    case VariableMessage::FP64:
      return framework::proto::VarType::FP64;  // NOLINT
    case VariableMessage::INT32:
      return framework::proto::VarType::INT32;  // NOLINT
    case VariableMessage::INT64:
      return framework::proto::VarType::INT64;  // NOLINT
    case VariableMessage::BOOL:
      return framework::proto::VarType::BOOL;  // NOLINT
    default:
      PADDLE_THROW(platform::errors::InvalidArgument(
          "VarMessageToVarType:Unsupported type %d", type));
  }
}

void SerializeToMultiVarMsgAndIOBuf(
    const std::string& message_name,
    const std::vector<std::string>& send_var_name_val,
    const std::vector<std::string>& recv_var_name_val,
    const platform::DeviceContext& ctx, const framework::Scope* scope,
    MultiVarMsg* request, butil::IOBuf* iobuf) {
  // 1. message_name
  request->set_message_name(message_name);

  // 2. var_names
  for (auto& send_var_name : send_var_name_val) {
    request->add_send_var_names(send_var_name);
  }
  for (auto& recv_var_name : recv_var_name_val) {
    request->add_recv_var_names(recv_var_name);
  }

  // 3. VarMessage
  for (auto& send_var_name : send_var_name_val) {
    auto* send_var_msg = request->add_var_messages();
    butil::IOBuf temp_iobuf;
    send_var_msg->set_varname(send_var_name);

    framework::Variable* var = scope->FindVar(send_var_name);

    if (var->IsType<framework::LoDTensor>()) {
      SerializeLodTensor(var, ctx, send_var_msg, &temp_iobuf);
    } else if (var->IsType<framework::SelectedRows>()) {
      SerializeSelectedRows(var, ctx, send_var_msg, &temp_iobuf);
    }
    iobuf->append(temp_iobuf);
  }
}

void SerializeLodTensor(framework::Variable* var,
                        const platform::DeviceContext& ctx, VarMsg* var_msg,
                        butil::IOBuf* iobuf) {
  auto* tensor = var->GetMutable<framework::LoDTensor>();
  var_msg->set_type(::paddle::LOD_TENSOR);
  const framework::LoD lod = tensor->lod();
  if (lod.size() > 0) {
    var_msg->set_lod_level(lod.size());
    for (auto& each : lod) {
      VarMsg::LodData* lod_inner = var_msg->add_lod();
      for (auto& d : each) {
        lod_inner->add_lod_data(d);
      }
    }
  }
  var_msg->set_data_type(static_cast<VarMsg::Type>(tensor->type()));
  for (auto& dim : framework::vectorize(tensor->dims())) {
    var_msg->add_dims(dim);
  }
  // IO Buffer
  if (platform::is_cpu_place(tensor->place())) {
    auto data_len = tensor->numel() * framework::SizeOfType(tensor->type());
    iobuf->append(reinterpret_cast<const char*>(&data_len), 8);
    iobuf->append(reinterpret_cast<const char*>(tensor->data<void>()),
                  data_len);
  } else {
#ifdef PADDLE_WITH_CUDA
    char* temp_ptr =
        new char[tensor->numel() * framework::SizeOfType(tensor->type())];
    auto stream =
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream();
    memory::Copy(platform::CPUPlace(), temp_ptr,
                 BOOST_GET_CONST(platform::CUDAPlace, tensor->place()),
                 tensor->data<void>(),
                 tensor->numel() * framework::SizeOfType(tensor->type()),
                 stream);
    auto data_len = tensor->numel() * framework::SizeOfType(tensor->type());
    iobuf->append(reinterpret_cast<const char*>(&data_len), 8);
    iobuf->append(reinterpret_cast<const char*>(temp_ptr), data_len);
    delete[] temp_ptr;
#endif
  }
}

void SerializeSelectedRows(framework::Variable* var,
                           const platform::DeviceContext& ctx, VarMsg* var_msg,
                           butil::IOBuf* iobuf) {
  framework::SelectedRows* slr = var->GetMutable<framework::SelectedRows>();
  auto* tensor = slr->mutable_value();
  auto* rows = slr->mutable_rows();

  var_msg->set_type(::paddle::SELECTED_ROWS);
  var_msg->set_slr_height(slr->height());

  auto* var_data = var_msg->mutable_data();
  var_data->clear();
  var_data->resize(rows->size() * sizeof(int64_t));
  char* data_ptr = const_cast<char*>(var_data->data());

  if (platform::is_cpu_place(tensor->place())) {
    memcpy(data_ptr, &(*rows)[0], rows->size() * sizeof(int64_t));
  } else {
#ifdef PADDLE_WITH_CUDA
    auto stream =
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream();
    memory::Copy(platform::CPUPlace(), data_ptr,
                 BOOST_GET_CONST(platform::CUDAPlace, tensor->place()),
                 &(*rows)[0], rows->size() * sizeof(int64_t), stream);
#endif
  }
  var_msg->set_data_type(static_cast<VarMsg::Type>(tensor->type()));
  for (auto& dim : framework::vectorize(tensor->dims())) {
    var_msg->add_dims(dim);
  }

  // IO Buffer
  if (platform::is_cpu_place(tensor->place())) {
    auto data_len = tensor->numel() * framework::SizeOfType(tensor->type());
    iobuf->append(reinterpret_cast<const char*>(&data_len), 8);
    iobuf->append(reinterpret_cast<const char*>(tensor->data<void>()),
                  data_len);
  } else {
#ifdef PADDLE_WITH_CUDA
    char* temp_ptr =
        new char[tensor->numel() * framework::SizeOfType(tensor->type())];
    auto stream =
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream();
    memory::Copy(platform::CPUPlace(), temp_ptr,
                 BOOST_GET_CONST(platform::CUDAPlace, tensor->place()),
                 tensor->data<void>(),
                 tensor->numel() * framework::SizeOfType(tensor->type()),
                 stream);
    auto data_len = tensor->numel() * framework::SizeOfType(tensor->type());
    iobuf->append(reinterpret_cast<const char*>(&data_len), 8);
    iobuf->append(reinterpret_cast<const char*>(temp_ptr), data_len);
    delete[] temp_ptr;
#endif
  }
}

void DeserializeFromMultiVarMsgAndIOBuf(const MultiVarMsg& multi_msg,
                                        const butil::IOBuf* iobuf,
                                        const platform::DeviceContext& ctx,
                                        framework::Scope* scope) {
  butil::IOBufBytesIterator io_buffer_itr(*iobuf);
  // size_t shard_buffer_remain = res_io_buffer.size();
  for (int recv_var_index = 0; recv_var_index < multi_msg.send_var_names_size();
       ++recv_var_index) {
    const auto& msg = multi_msg.var_messages(recv_var_index);
    auto* var = scope->Var(msg.varname());
    if (msg.type() == ::paddle::LOD_TENSOR) {
      DeserializeLodTensor(var, msg, io_buffer_itr, ctx);
    } else if (msg.type() == ::paddle::SELECTED_ROWS) {
      DeserializeSelectedRows(var, msg, io_buffer_itr, ctx);
    }
  }
}

void DeserializeFromMultiVarMsgAndIOBuf(const MultiVarMsg& multi_msg,
                                        const butil::IOBuf* iobuf,
                                        const platform::DeviceContext& ctx,
                                        const framework::Scope* scope) {
  butil::IOBufBytesIterator io_buffer_itr(*iobuf);
  // size_t shard_buffer_remain = res_io_buffer.size();
  for (int recv_var_index = 0; recv_var_index < multi_msg.send_var_names_size();
       ++recv_var_index) {
    const auto& msg = multi_msg.var_messages(recv_var_index);
    auto* var = scope->FindVar(msg.varname());
    PADDLE_ENFORCE_NE(var, nullptr,
                      platform::errors::InvalidArgument(
                          "Not find variable %s in scope.", msg.varname()));
    if (msg.type() == ::paddle::LOD_TENSOR) {
      DeserializeLodTensor(var, msg, io_buffer_itr, ctx);
    } else if (msg.type() == ::paddle::SELECTED_ROWS) {
      DeserializeSelectedRows(var, msg, io_buffer_itr, ctx);
    }
  }
}

void DeserializeLodTensor(framework::Variable* var, const VarMsg& msg,
                          butil::IOBufBytesIterator& io_buffer_itr,
                          const platform::DeviceContext& ctx) {
  const auto place = ctx.GetPlace();
  framework::LoDTensor* tensor = var->GetMutable<framework::LoDTensor>();
  std::vector<int> vec_dim;
  for (auto& x : msg.dims()) {
    vec_dim.push_back(x);
  }
  tensor->Resize(framework::make_ddim(vec_dim));

  framework::LoD lod;
  for (int i = 0; i < msg.lod_level(); ++i) {
    framework::Vector<size_t> v;
    for (int j = 0; j < msg.lod(i).lod_data_size(); ++j) {
      v.push_back(msg.lod(i).lod_data(j));
    }
    lod.push_back(v);
  }
  tensor->set_lod(lod);

  void* tensor_data =
      tensor->mutable_data(place, VarMessageToVarType(msg.data_type()));

  // IO Buffer
  if (platform::is_cpu_place(place)) {
    unsigned long data_len;
    io_buffer_itr.copy_and_forward((void*)(&data_len), 8);
    io_buffer_itr.copy_and_forward(tensor_data, data_len);
  } else if (platform::is_gpu_place(place)) {
#ifdef PADDLE_WITH_CUDA
    unsigned long data_len;
    char* temp_ptr =
        new char[tensor->numel() * framework::SizeOfType(tensor->type())];
    io_buffer_itr.copy_and_forward((void*)(&data_len), 8);
    io_buffer_itr.copy_and_forward((void*)temp_ptr, data_len);
    auto stream =
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream();
    memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, place), tensor_data,
                 platform::CPUPlace(), (void*)temp_ptr,
                 tensor->numel() * framework::SizeOfType(tensor->type()),
                 stream);
    delete[] temp_ptr;
#endif
  }
}

void DeserializeSelectedRows(framework::Variable* var, const VarMsg& msg,
                             butil::IOBufBytesIterator& io_buffer_itr,
                             const platform::DeviceContext& ctx) {
  const auto place = ctx.GetPlace();
  auto* slr = var->GetMutable<framework::SelectedRows>();
  framework::Tensor* tensor = slr->mutable_value();
  slr->set_height(msg.slr_height());
  std::vector<int64_t> tmp_rows(msg.slr_height());
  memcpy(&tmp_rows[0], msg.data().data(), msg.slr_height() * sizeof(int64_t));
  slr->set_rows(tmp_rows);
  std::vector<int> vec_dim;
  for (auto& x : msg.dims()) {
    vec_dim.push_back(x);
  }
  tensor->Resize(framework::make_ddim(vec_dim));
  void* tensor_data =
      tensor->mutable_data(place, VarMessageToVarType(msg.data_type()));
  // IO Buffer
  if (platform::is_cpu_place(place)) {
    unsigned long data_len;
    io_buffer_itr.copy_and_forward((void*)(&data_len), 8);
    io_buffer_itr.copy_and_forward(tensor_data, data_len);
  } else if (platform::is_gpu_place(place)) {
#ifdef PADDLE_WITH_CUDA
    char* temp_ptr =
        new char[tensor->numel() * framework::SizeOfType(tensor->type())];
    unsigned long data_len;
    io_buffer_itr.copy_and_forward((void*)(&data_len), 8);
    io_buffer_itr.copy_and_forward(temp_ptr, data_len);
    auto stream =
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream();
    memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, place), tensor_data,
                 platform::CPUPlace(), temp_ptr,
                 tensor->numel() * framework::SizeOfType(tensor->type()),
                 stream);
    delete[] temp_ptr;
#endif
  }
}

}  // namespace distributed
}  // namespace paddle
