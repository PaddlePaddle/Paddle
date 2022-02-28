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

#include "paddle/fluid/distributed/ps/service/brpc_utils.h"

#include <arpa/inet.h>
#include <netdb.h>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
class Variable;
}  // namespace framework
}  // namespace paddle

namespace phi {
class DenseTensor;
}  // namespace phi

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
    } else if (var->IsType<phi::SelectedRows>()) {
      SerializeSelectedRows(var, ctx, send_var_msg, &temp_iobuf);
    }
    iobuf->append(temp_iobuf);
  }
}

void SerializeLodTensor(framework::Variable* var,
                        const platform::DeviceContext& ctx, VarMsg* var_msg,
                        butil::IOBuf* iobuf) {
  auto* tensor = var->GetMutable<framework::LoDTensor>();
  var_msg->set_type(::paddle::distributed::LOD_TENSOR);
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
  var_msg->set_data_type(static_cast<VarMsg::Type>(
      framework::TransToProtoVarType(tensor->dtype())));
  for (auto& dim : phi::vectorize(tensor->dims())) {
    var_msg->add_dims(dim);
  }
  // IO Buffer
  if (platform::is_cpu_place(tensor->place())) {
    auto data_len = tensor->numel() * framework::DataTypeSize(tensor->dtype());
    iobuf->append(reinterpret_cast<const char*>(&data_len), 8);
    iobuf->append(reinterpret_cast<const char*>(tensor->data()), data_len);
  } else {
#ifdef PADDLE_WITH_CUDA
    char* temp_ptr =
        new char[tensor->numel() *
                 framework::DataTypeSize(tensor->dtype())];  // NOLINT
    auto stream =
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream();
    memory::Copy(
        platform::CPUPlace(), temp_ptr, tensor->place(), tensor->data(),
        tensor->numel() * framework::SizeOfType(
                              framework::TransToProtoVarType(tensor->dtype())),
        stream);
    auto data_len = tensor->numel() * framework::DataTypeSize(tensor->dtype());
    iobuf->append(reinterpret_cast<const char*>(&data_len), 8);
    iobuf->append(reinterpret_cast<const char*>(temp_ptr), data_len);
    delete[] temp_ptr;
#endif
  }
}

void SerializeSelectedRows(framework::Variable* var,
                           const platform::DeviceContext& ctx, VarMsg* var_msg,
                           butil::IOBuf* iobuf) {
  phi::SelectedRows* slr = var->GetMutable<phi::SelectedRows>();
  auto* tensor = slr->mutable_value();
  auto* rows = slr->mutable_rows();

  var_msg->set_type(::paddle::distributed::SELECTED_ROWS);
  var_msg->set_slr_height(slr->height());

  auto* var_data = var_msg->mutable_data();
  var_data->clear();
  var_data->resize(rows->size() * sizeof(int64_t));
  char* data_ptr = const_cast<char*>(var_data->data());
  memcpy(data_ptr, &((*rows)[0]), rows->size() * sizeof(int64_t));
  var_msg->set_data_type(static_cast<VarMsg::Type>(
      framework::TransToProtoVarType(tensor->dtype())));
  for (auto& dim : phi::vectorize(tensor->dims())) {
    var_msg->add_dims(dim);
  }
  // IO Buffer
  if (platform::is_cpu_place(tensor->place())) {
    auto data_len = tensor->numel() * framework::DataTypeSize(tensor->dtype());
    iobuf->append(reinterpret_cast<const char*>(&data_len), 8);
    iobuf->append(reinterpret_cast<const char*>(tensor->data()), data_len);
  } else {
#ifdef PADDLE_WITH_CUDA
    char* temp_ptr =
        new char[tensor->numel() *
                 framework::DataTypeSize(tensor->dtype())];  // NOLINT
    auto stream =
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream();
    memory::Copy(
        platform::CPUPlace(), temp_ptr, tensor->place(), tensor->data(),
        tensor->numel() * framework::SizeOfType(
                              framework::TransToProtoVarType(tensor->dtype())),
        stream);
    auto data_len = tensor->numel() * framework::DataTypeSize(tensor->dtype());
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
    if (msg.type() == ::paddle::distributed::LOD_TENSOR) {
      DeserializeLodTensor(var, msg, io_buffer_itr, ctx);
    } else if (msg.type() == ::paddle::distributed::SELECTED_ROWS) {
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
    if (msg.type() == ::paddle::distributed::LOD_TENSOR) {
      DeserializeLodTensor(var, msg, io_buffer_itr, ctx);
    } else if (msg.type() == ::paddle::distributed::SELECTED_ROWS) {
      DeserializeSelectedRows(var, msg, io_buffer_itr, ctx);
    }
  }
}

void DeserializeLodTensor(framework::Variable* var, const VarMsg& msg,
                          butil::IOBufBytesIterator& io_buffer_itr,  // NOLINT
                          const platform::DeviceContext& ctx) {
  const auto place = ctx.GetPlace();
  framework::LoDTensor* tensor = var->GetMutable<framework::LoDTensor>();
  std::vector<int> vec_dim;
  for (auto& x : msg.dims()) {
    vec_dim.push_back(x);
  }
  tensor->Resize(phi::make_ddim(vec_dim));

  framework::LoD lod;
  for (int i = 0; i < msg.lod_level(); ++i) {
    framework::Vector<size_t> v;
    for (int j = 0; j < msg.lod(i).lod_data_size(); ++j) {
      v.push_back(msg.lod(i).lod_data(j));
    }
    lod.push_back(v);
  }
  tensor->set_lod(lod);

  void* tensor_data = tensor->mutable_data(
      place,
      framework::TransToPhiDataType(VarMessageToVarType(msg.data_type())));

  // IO Buffer
  if (platform::is_cpu_place(place)) {
    unsigned long data_len;                                 // NOLINT
    io_buffer_itr.copy_and_forward((void*)(&data_len), 8);  // NOLINT
    io_buffer_itr.copy_and_forward(tensor_data, data_len);
  } else if (platform::is_gpu_place(place)) {
#ifdef PADDLE_WITH_CUDA
    unsigned long data_len;  // NOLINT
    char* temp_ptr =
        new char[tensor->numel() *
                 framework::DataTypeSize(tensor->dtype())];     // NOLINT
    io_buffer_itr.copy_and_forward((void*)(&data_len), 8);      // NOLINT
    io_buffer_itr.copy_and_forward((void*)temp_ptr, data_len);  // NOLINT
    auto stream =
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream();
    memory::Copy(
        place, tensor_data, platform::CPUPlace(), (void*)temp_ptr,  // NOLINT
        tensor->numel() * framework::DataTypeSize(tensor->dtype()), stream);
    delete[] temp_ptr;
#endif
  }
}

void DeserializeSelectedRows(
    framework::Variable* var, const VarMsg& msg,
    butil::IOBufBytesIterator& io_buffer_itr,  // NOLINT
    const platform::DeviceContext& ctx) {
  const auto place = ctx.GetPlace();
  auto* slr = var->GetMutable<phi::SelectedRows>();
  framework::Tensor* tensor = slr->mutable_value();
  slr->set_height(msg.slr_height());
  std::vector<int64_t> tmp_rows(msg.dims()[0]);
  memcpy(tmp_rows.data(), msg.data().data(), msg.dims()[0] * sizeof(int64_t));
  slr->set_rows(tmp_rows);
  std::vector<int> vec_dim;
  for (auto& x : msg.dims()) {
    vec_dim.push_back(x);
  }
  tensor->Resize(phi::make_ddim(vec_dim));
  void* tensor_data = tensor->mutable_data(
      place,
      framework::TransToPhiDataType(VarMessageToVarType(msg.data_type())));
  // IO Buffer
  if (platform::is_cpu_place(place)) {
    unsigned long data_len;                                 // NOLINT
    io_buffer_itr.copy_and_forward((void*)(&data_len), 8);  // NOLINT
    io_buffer_itr.copy_and_forward(tensor_data, data_len);
  } else if (platform::is_gpu_place(place)) {
#ifdef PADDLE_WITH_CUDA
    char* temp_ptr =
        new char[tensor->numel() *
                 framework::DataTypeSize(tensor->dtype())];  // NOLINT
    unsigned long data_len;                                  // NOLINT
    io_buffer_itr.copy_and_forward((void*)(&data_len), 8);   // NOLINT
    io_buffer_itr.copy_and_forward(temp_ptr, data_len);
    auto stream =
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream();
    memory::Copy(place, tensor_data, platform::CPUPlace(), temp_ptr,
                 tensor->numel() * framework::DataTypeSize(tensor->dtype()),
                 stream);
    delete[] temp_ptr;
#endif
  }
}

std::string GetIntTypeEndpoint(const std::string& ip, const uint32_t& port) {
  // There are usually two forms of IP address: ip(int) / ip (hostname)
  // If there're some problem with DNS, or ip triggers the bug of Brpc
  // We will try to get the IP address of the domain name manually again
  std::string ip_port = ip + ":" + std::to_string(port);
  struct hostent* hp = NULL;
  hp = gethostbyname(ip.c_str());

  if (NULL == hp) {
    LOG(ERROR) << "Brpc Start failed, ip_port= " << ip_port
               << " , Error infomation: " << hstrerror(h_errno);
  }

  int i = 0;
  char* int_ip = NULL;

  while (hp->h_addr_list[i] != NULL) {
    int_ip = inet_ntoa(*(struct in_addr*)hp->h_addr_list[i]);
    VLOG(3) << "Brpc Get host by name, host:" << ip << " -> ip: " << int_ip;
    break;
  }

  std::string str_ip = int_ip;
  std::string int_ip_port = str_ip + ":" + std::to_string(port);
  return int_ip_port;
}

}  // namespace distributed
}  // namespace paddle
