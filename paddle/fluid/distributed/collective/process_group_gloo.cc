// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>

#ifdef _WIN32
#include <gloo/common/win.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <gloo/reduce.h>

#include "glog/logging.h"
#include "paddle/fluid/distributed/collective/common.h"
#include "paddle/fluid/distributed/collective/process_group_gloo.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/enforce.h"

namespace paddle::distributed {

#ifdef _WIN32
#define GENERATE_FUNC(type, func, ...)       \
  switch (type) {                            \
    case phi::DataType::FLOAT32:             \
      func<float>(__VA_ARGS__);              \
      break;                                 \
    case phi::DataType::FLOAT64:             \
      func<double>(__VA_ARGS__);             \
      break;                                 \
    case phi::DataType::FLOAT16:             \
      func<gloo::float16>(__VA_ARGS__);      \
      break;                                 \
    case phi::DataType::INT32:               \
      func<int32_t>(__VA_ARGS__);            \
      break;                                 \
    case phi::DataType::INT64:               \
      func<int64_t>(__VA_ARGS__);            \
      break;                                 \
    default:                                 \
      VLOG(0) << "Error: Unknown DataType."; \
      exit(-1);                              \
  }

#define HOST_NAME_MAX 256

#else
#define GENERATE_FUNC(type, func, args...)   \
  switch (type) {                            \
    case phi::DataType::FLOAT32:             \
      func<float>(args);                     \
      break;                                 \
    case phi::DataType::FLOAT64:             \
      func<double>(args);                    \
      break;                                 \
    case phi::DataType::FLOAT16:             \
      func<gloo::float16>(args);             \
      break;                                 \
    case phi::DataType::INT32:               \
      func<int32_t>(args);                   \
      break;                                 \
    case phi::DataType::INT64:               \
      func<int64_t>(args);                   \
      break;                                 \
    case phi::DataType::INT8:                \
      func<int8_t>(args);                    \
      break;                                 \
    case phi::DataType::UINT8:               \
      func<uint8_t>(args);                   \
      break;                                 \
    case phi::DataType::BOOL:                \
      func<bool>(args);                      \
      break;                                 \
    case phi::DataType::BFLOAT16:            \
      func<bfloat16>(args);                  \
      break;                                 \
    default:                                 \
      VLOG(0) << "Error: Unknown DataType."; \
      exit(-1);                              \
  }
#endif

template <typename T>
T* get_data(phi::DenseTensor& tensor) {  // NOLINT
  return reinterpret_cast<T*>(tensor.data());
}

template <typename T>
std::vector<T*> get_multi_data(
    std::vector<phi::DenseTensor>& tensors) {  // NOLINT
  std::vector<T*> ret;
  ret.reserve(tensors.size());
  for (auto& tensor : tensors) {
    ret.push_back(get_data<T>(tensor));
  }
  return ret;
}

template <typename T, typename P>
void set_output(P& opts, phi::DenseTensor& tensor) {  // NOLINT
  opts.setOutput(get_data<T>(tensor), tensor.numel());
}

template <typename T, typename P>
void set_input(P& opts, phi::DenseTensor& tensor) {  // NOLINT
  opts.setInput(get_data<T>(tensor), tensor.numel());
}

template <typename T, typename P>
void set_outputs(P& opts,                                   // NOLINT
                 std::vector<phi::DenseTensor>& tensors) {  // NOLINT
  opts.setOutputs(get_multi_data<T>(tensors), tensors[0].numel());
}

template <typename T, typename P>
void set_inputs(P& opts,                                   // NOLINT
                std::vector<phi::DenseTensor>& tensors) {  // NOLINT
  opts.setInputs(get_multi_data<T>(tensors), tensors[0].numel());
}

template <typename T, typename P>
void set_inputs_for_scatter(P& opts,                   // NOLINT
                            phi::DenseTensor& tensor,  // NOLINT
                            int nranks) {
  std::vector<T*> ret;
  ret.reserve(nranks);
  T* raw_pointer = reinterpret_cast<T*>(tensor.data());
  size_t offset = 0;
  for (int i = 0; i < nranks; i++) {
    ret.push_back(raw_pointer + offset);
    offset += tensor.numel() / nranks;
  }
  opts.setInputs(ret, tensor.numel() / nranks);
}

ProcessGroupGloo::GlooTask::GlooTask(
    int rank, const std::vector<phi::DenseTensor>& inputs, CommType comm_type)
    : ProcessGroup::Task(rank, inputs, comm_type) {}

ProcessGroupGloo::ProcessGroupGloo(
    const std::shared_ptr<phi::distributed::Store>& store,
    int rank,
    int world_size,
    int gid,
    const std::shared_ptr<GlooOptions> options)
    : ProcessGroupWithoutStream(rank, world_size, gid),
      _tag(0),
      _store(new GlooStore(store)) {
  _context = std::make_shared<gloo::rendezvous::Context>(rank, world_size);
  _context->connectFullMesh(*_store, options->device);
}

class BroadcastGlooTask : public ProcessGroupGloo::GlooTask {
 public:
  BroadcastGlooTask(phi::distributed::GlooCommContext* comm_context,
                    std::vector<phi::DenseTensor>& inputs,   // NOLINT
                    std::vector<phi::DenseTensor>& outputs,  // NOLINT
                    int rank,
                    int root,
                    uint32_t tag)
      : ProcessGroupGloo::GlooTask(rank, inputs, CommType::BROADCAST),
        _comm_context(comm_context),
        _root(root),
        _inputs(inputs),
        _outputs(outputs),
        _tag(tag) {}

  void Run() override { _do_broadcast(_inputs[0], _outputs[0]); }

 private:
  phi::distributed::GlooCommContext* _comm_context;
  const int _root;
  std::vector<phi::DenseTensor> _inputs{};
  std::vector<phi::DenseTensor> _outputs{};
  const uint32_t _tag;

  void _do_broadcast(phi::DenseTensor& in, phi::DenseTensor& out) {  // NOLINT
    _comm_context->Broadcast(&(out), in, _root, _tag);
  }
};

// TODO(sunyilun): for compatibility, will be updated later
std::shared_ptr<ProcessGroup::Task> ProcessGroupGloo::Broadcast(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const BroadcastOptions& opts,
    bool sync_op) {
  std::vector<phi::DenseTensor> in_wrapper{in_tensor};
  std::vector<phi::DenseTensor> out_wrapper{*out_tensor};
  return Broadcast(in_wrapper, out_wrapper, opts, true);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupGloo::Broadcast(
    std::vector<phi::DenseTensor>& inputs,
    std::vector<phi::DenseTensor>& outputs,
    const BroadcastOptions& opts) {
  return Broadcast(inputs, outputs, opts, true);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupGloo::Broadcast(
    std::vector<phi::DenseTensor>& inputs,
    std::vector<phi::DenseTensor>& outputs,
    const BroadcastOptions& opts,
    bool sync_op) {
  CheckTensorContiguous(inputs);
  CheckTensorContiguous(outputs);

  auto root = opts.source_rank;
  std::unique_ptr<BroadcastGlooTask> task;
  auto tag = next_tag();
  auto comm_context = this->GetCommContext();
  task = std::make_unique<BroadcastGlooTask>(
      comm_context, inputs, outputs, rank_, root, tag);
  task->Run();
  return task;
}

class SendGlooTask : public ProcessGroupGloo::GlooTask {
 public:
  SendGlooTask(phi::distributed::GlooCommContext* comm_context,
               std::vector<phi::DenseTensor>* inputs,
               int rank,
               int dst_rank,
               uint32_t tag)
      : ProcessGroupGloo::GlooTask(rank, *inputs, CommType::SEND),
        _comm_context(comm_context),
        _inputs(*inputs),
        _dst(dst_rank),
        _tag(tag) {}

  void Run() override { _do_send(_inputs); }

 private:
  phi::distributed::GlooCommContext* _comm_context;
  std::vector<phi::DenseTensor> _inputs;
  int _dst;
  uint32_t _tag;

  void _do_send(std::vector<phi::DenseTensor>& in) {  // NOLINT
    _comm_context->Send(in[0], _dst, _tag);
  }
};

std::shared_ptr<ProcessGroup::Task> ProcessGroupGloo::Send(
    const phi::DenseTensor& tensor, int dst_rank, bool sync_op) {
  std::vector<phi::DenseTensor> in_wrapper{tensor};
  return Send(in_wrapper, dst_rank);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupGloo::Send(
    std::vector<phi::DenseTensor>& inputs, int dst_rank) {
  CheckTensorContiguous(inputs);
  std::unique_ptr<SendGlooTask> task;
  auto tag = next_tag();
  auto comm_context = this->GetCommContext();
  task = std::make_unique<SendGlooTask>(
      comm_context, &inputs, rank_, dst_rank, tag);
  task->Run();

  return task;
}

class RecvGlooTask : public ProcessGroupGloo::GlooTask {
 public:
  RecvGlooTask(phi::distributed::GlooCommContext* comm_context,
               std::vector<phi::DenseTensor>* outputs,
               int rank,
               int src_rank,
               uint32_t tag)
      : ProcessGroupGloo::GlooTask(rank, *outputs, CommType::RECV),
        _comm_context(comm_context),
        _outputs(*outputs),
        _src(src_rank),
        _tag(tag) {}

  void Run() override { _do_recv(_outputs); }

 private:
  phi::distributed::GlooCommContext* _comm_context;
  std::vector<phi::DenseTensor> _outputs;
  const int _src;
  const uint32_t _tag;

  void _do_recv(std::vector<phi::DenseTensor>& out) {  // NOLINT
    _comm_context->Recv(&(out[0]), _src, _tag);
  }
};

std::shared_ptr<ProcessGroup::Task> ProcessGroupGloo::Recv(
    phi::DenseTensor* tensor, int src_rank, bool sync_op) {
  std::vector<phi::DenseTensor> in_wrapper{*tensor};
  return Recv(in_wrapper, src_rank);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupGloo::Recv(
    std::vector<phi::DenseTensor>& outputs, int src_rank) {
  std::unique_ptr<RecvGlooTask> task;
  auto tag = next_tag();
  auto comm_context = this->GetCommContext();

  task = std::make_unique<RecvGlooTask>(
      comm_context, &outputs, rank_, src_rank, tag);
  task->Run();
  return task;
}

class AllreduceGlooTask : public ProcessGroupGloo::GlooTask {
 public:
  AllreduceGlooTask(int rank,
                    phi::distributed::GlooCommContext* comm_context,
                    std::vector<phi::DenseTensor>& inputs,   // NOLINT
                    std::vector<phi::DenseTensor>& outputs,  // NOLINT
                    ReduceOp reduce_op,
                    uint32_t tag)
      : ProcessGroupGloo::GlooTask(rank, inputs, CommType::ALLREDUCE),
        _comm_context(comm_context),
        _inputs(inputs),
        _outputs(outputs),
        _reduce_op(reduce_op),
        _tag(tag) {}

  void Run() override { _do_allreduce(_inputs, _outputs); }

 private:
  phi::distributed::GlooCommContext* _comm_context;
  std::vector<phi::DenseTensor> _inputs;
  std::vector<phi::DenseTensor> _outputs;
  const ReduceOp _reduce_op;
  uint32_t _tag;

  void _do_allreduce(std::vector<phi::DenseTensor>& ins,     // NOLINT
                     std::vector<phi::DenseTensor>& outs) {  // NOLINT
    _comm_context->AllReduce(
        &(outs[0]), ins[0], static_cast<int>(_reduce_op), _tag);
  }
};

std::shared_ptr<ProcessGroup::Task> ProcessGroupGloo::AllReduce(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const AllreduceOptions& opts,
    bool sync_op) {
  std::vector<phi::DenseTensor> in_wrapper{in_tensor};
  std::vector<phi::DenseTensor> out_wrapper{*out_tensor};
  return AllReduce(in_wrapper, out_wrapper, opts, true);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupGloo::AllReduce(
    std::vector<phi::DenseTensor>& inputs,
    std::vector<phi::DenseTensor>& outputs,
    const AllreduceOptions& opts) {
  return AllReduce(inputs, outputs, opts, true);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupGloo::AllReduce(
    std::vector<phi::DenseTensor>& inputs,
    std::vector<phi::DenseTensor>& outputs,
    const AllreduceOptions& opts,
    bool sync_op) {
  CheckTensorContiguous(inputs);
  CheckTensorContiguous(outputs);

  auto tag = next_tag();
  std::shared_ptr<GlooTask> task;
  auto comm_context = this->GetCommContext();
  task = std::make_shared<AllreduceGlooTask>(
      rank_, comm_context, inputs, outputs, opts.reduce_op, tag);
  task->Run();
  return task;
}

class BarrierGlooTask : public ProcessGroupGloo::GlooTask {
 public:
  BarrierGlooTask(int rank, phi::distributed::GlooCommContext* comm_context)
      : ProcessGroupGloo::GlooTask(
            rank, std::vector<phi::DenseTensor>{}, CommType::BARRIER),
        _comm_context(comm_context) {}

  void Run() override { _do_barrier(); }

 private:
  phi::distributed::GlooCommContext* _comm_context;

  void _do_barrier() { _comm_context->Barrier(); }
};

std::shared_ptr<ProcessGroup::Task> ProcessGroupGloo::Barrier(
    const BarrierOptions& opts) {
  std::shared_ptr<BarrierGlooTask> task;
  auto comm_context = this->GetCommContext();
  task = std::make_shared<BarrierGlooTask>(rank_, comm_context);
  task->Run();
  return task;
}

class AllgatherGlooTask : public ProcessGroupGloo::GlooTask {
 public:
  AllgatherGlooTask(int rank,
                    phi::distributed::GlooCommContext* comm_context,
                    std::vector<phi::DenseTensor>& inputs,   // NOLINT
                    std::vector<phi::DenseTensor>& outputs,  // NOLINT
                    uint32_t tag)
      : ProcessGroupGloo::GlooTask(rank, inputs, CommType::ALLGATHER),
        _comm_context(comm_context),
        _inputs(inputs),
        _outputs(outputs),
        _tag(tag) {}

  void Run() override { _do_allgather(_inputs, _outputs); }

 private:
  phi::distributed::GlooCommContext* _comm_context;
  std::vector<phi::DenseTensor> _inputs;
  std::vector<phi::DenseTensor> _outputs;
  uint32_t _tag;

  void _do_allgather(std::vector<phi::DenseTensor>& in,     // NOLINT
                     std::vector<phi::DenseTensor>& out) {  // NOLINT
    _comm_context->AllGather(&(out[0]), in[0], _tag);
  }
};

std::shared_ptr<ProcessGroup::Task> ProcessGroupGloo::AllGather(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    int64_t /*offset*/,
    int64_t /*offset*/,
    bool sync_op) {
  std::vector<phi::DenseTensor> in_wrapper{in_tensor};
  std::vector<phi::DenseTensor> out_wrapper{*out_tensor};
  return AllGather(in_wrapper, out_wrapper, true);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupGloo::AllGather(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors) {
  return AllGather(in_tensors, out_tensors, true);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupGloo::AllGather(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    bool sync_op) {
  CheckTensorContiguous(in_tensors);
  CheckTensorContiguous(out_tensors);
  std::shared_ptr<AllgatherGlooTask> task;
  auto tag = next_tag();
  auto comm_context = this->GetCommContext();
  task = std::make_shared<AllgatherGlooTask>(
      rank_, comm_context, in_tensors, out_tensors, tag);
  task->Run();
  return task;
}

class ReduceGlooTask : public ProcessGroupGloo::GlooTask {
 public:
  ReduceGlooTask(int rank,
                 phi::distributed::GlooCommContext* comm_context,
                 std::vector<phi::DenseTensor>& inputs,   // NOLINT
                 std::vector<phi::DenseTensor>& outputs,  // NOLINT
                 ReduceOp reduce_op,
                 int dst,
                 uint32_t tag)
      : ProcessGroupGloo::GlooTask(rank, inputs, CommType::REDUCE),
        _comm_context(comm_context),
        _inputs(inputs),
        _outputs(outputs),
        _reduce_op(reduce_op),
        _dst(dst),
        _tag(tag) {}

  void Run() override { _do_reduce(_inputs, _outputs, _dst); }

 private:
  phi::distributed::GlooCommContext* _comm_context;
  std::vector<phi::DenseTensor> _inputs;
  std::vector<phi::DenseTensor> _outputs;
  const ReduceOp _reduce_op;
  int _dst;
  uint32_t _tag;

  void _do_reduce(std::vector<phi::DenseTensor>& inputs,   // NOLINT
                  std::vector<phi::DenseTensor>& outputs,  // NOLINT
                  int dst) {
    _comm_context->Reduce(
        &(outputs[0]), inputs[0], static_cast<int>(_reduce_op), _dst, _tag);
  }
};

std::shared_ptr<ProcessGroup::Task> ProcessGroupGloo::Reduce(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const ReduceOptions& opts,
    bool sync_op  // for compatibility, no use now
) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);

  std::shared_ptr<ReduceGlooTask> task;
  auto tag = next_tag();
  auto comm_context = this->GetCommContext();
  std::vector<phi::DenseTensor> in_wrapper{in_tensor};
  std::vector<phi::DenseTensor> out_wrapper{*out_tensor};
  task = std::make_shared<ReduceGlooTask>(rank_,
                                          comm_context,
                                          in_wrapper,
                                          out_wrapper,
                                          opts.reduce_op,
                                          opts.root_rank,
                                          tag);
  task->Run();
  return task;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupGloo::Reduce(
    std::vector<phi::DenseTensor>& inputs,
    std::vector<phi::DenseTensor>& outputs,
    const ReduceOptions& opts) {
  return Reduce(&outputs[0], inputs[0], opts, true);
}

class ScatterGlooTask : public ProcessGroupGloo::GlooTask {
 public:
  ScatterGlooTask(int rank,
                  phi::distributed::GlooCommContext* comm_context,
                  std::vector<phi::DenseTensor>& inputs,   // NOLINT
                  std::vector<phi::DenseTensor>& outputs,  // NOLINT
                  int src,
                  int size,
                  uint32_t tag)
      : ProcessGroupGloo::GlooTask(rank, inputs, CommType::SCATTER),
        _comm_context(comm_context),
        _inputs(inputs),
        _outputs(outputs),
        _src(src),
        _size(size),
        _tag(tag) {}

  void Run() override { _do_scatter(_inputs, _outputs, _src); }

 private:
  phi::distributed::GlooCommContext* _comm_context;
  std::vector<phi::DenseTensor> _inputs;
  std::vector<phi::DenseTensor> _outputs;
  int _src;
  int _size;
  uint32_t _tag;

  void _do_scatter(std::vector<phi::DenseTensor>& in,   // NOLINT
                   std::vector<phi::DenseTensor>& out,  // NOLINT
                   int src) {
    _comm_context->Scatter(&(out[0]), in[0], _src, _size, _tag);
  }
};

std::shared_ptr<ProcessGroup::Task> ProcessGroupGloo::Scatter(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const ScatterOptions& opts,
    bool sync_op) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);
  std::shared_ptr<ScatterGlooTask> task;

  auto tag = next_tag();
  auto comm_context = this->GetCommContext();
  std::vector<phi::DenseTensor> in_wrapper{in_tensor};
  std::vector<phi::DenseTensor> out_wrapper{*out_tensor};
  task = std::make_shared<ScatterGlooTask>(
      rank_, comm_context, in_wrapper, out_wrapper, opts.root_rank, size_, tag);
  task->Run();
  return task;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupGloo::Scatter(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const ScatterOptions& opts) {
  return Scatter(&out_tensors[0], in_tensors[0], opts, true);
}

class GatherGlooTask : public ProcessGroupGloo::GlooTask {
 public:
  GatherGlooTask(int rank,
                 phi::distributed::GlooCommContext* comm_context,
                 const phi::DenseTensor& input,  // NOLINT
                 phi::DenseTensor* output,       // NOLINT
                 int src,
                 uint32_t tag)
      : ProcessGroupGloo::GlooTask(rank, {input}, CommType::GATHER),
        _comm_context(comm_context),
        _input(input),
        _output(*output),
        _src(src),
        _tag(tag) {}

  void Run() override { _do_gather(_input, _output, _src); }

 private:
  phi::distributed::GlooCommContext* _comm_context;
  phi::DenseTensor _input;
  phi::DenseTensor _output;
  int _src;
  uint32_t _tag;

  void _do_gather(phi::DenseTensor& in,   // NOLINT
                  phi::DenseTensor& out,  // NOLINT
                  int src) {
    _comm_context->Gather(&(out), in, src, _tag);
  }
};

std::shared_ptr<ProcessGroup::Task> ProcessGroupGloo::Gather(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const GatherOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);

  PADDLE_ENFORCE_NE(
      use_calc_stream,
      true,
      common::errors::InvalidArgument("Gloo cannot use use_calc_stream."));
  std::shared_ptr<GatherGlooTask> task;
  auto tag = next_tag();
  auto comm_context = this->GetCommContext();
  task = std::make_shared<GatherGlooTask>(
      rank_, comm_context, in_tensor, out_tensor, opts.root_rank, tag);
  task->Run();
  return task;
}

std::shared_ptr<::gloo::transport::Device>
ProcessGroupGloo::createDeviceForInterface(const std::string& ifname) {
  ::gloo::transport::tcp::attr attr;
  attr.iface = ifname;
  return ::gloo::transport::tcp::CreateDevice(attr);
}

std::shared_ptr<::gloo::transport::Device>
ProcessGroupGloo::createDeviceForHostname(const std::string& hostname) {
  ::gloo::transport::tcp::attr attr;
  attr.hostname = hostname;
  return ::gloo::transport::tcp::CreateDevice(attr);
}

std::shared_ptr<::gloo::transport::Device>
ProcessGroupGloo::createDefaultDevice() {
  std::array<char, HOST_NAME_MAX> hostname{};
  auto ret = ::gethostname(hostname.data(), HOST_NAME_MAX);
  PADDLE_ENFORCE_EQ(
      ret,
      0,
      common::errors::Fatal("Get hostname error for createDefaultDevice."));
  ::addrinfo* result;
  result = phi::distributed::tcputils::get_addr_info(
      hostname.data(), "", 0, AF_UNSPEC);
  ::addrinfo* cur;
  for (cur = result; cur != nullptr; cur = cur->ai_next) {
    phi::distributed::SocketType socket =
        ::socket(cur->ai_family, cur->ai_socktype, cur->ai_protocol);
    if (socket == -1) {
      continue;
    }
    ret = ::bind(socket, cur->ai_addr, cur->ai_addrlen);
#ifdef _WIN32
    closesocket(socket);
#else
    close(socket);
#endif
    if (ret == -1) {
      continue;
    }
    break;
  }
  freeaddrinfo(result);
  if (cur != nullptr) {
    return createDeviceForHostname(hostname.data());
  }
  return createDeviceForHostname("127.0.0.1");
}

std::shared_ptr<ProcessGroupGloo> ProcessGroupGloo::CreateProcessGroupGloo(
    const std::shared_ptr<phi::distributed::Store>& store,
    int rank,
    int size,
    int gid) {
  std::string GLOO_SOCKET_IFNAME_ENV = "GLOO_SOCKET_IFNAME";
  auto opts = GlooOptions::create();
  char* ifname = getenv(GLOO_SOCKET_IFNAME_ENV.c_str());
  if (ifname && strlen(ifname) > 1) {
    opts->device =
        ProcessGroupGloo::createDeviceForInterface(std::string(ifname));
  } else {
    opts->device = ProcessGroupGloo::createDefaultDevice();
  }
  phi::distributed::CommContextManager::CreateGlooCommContext(
      store, std::to_string(gid), rank, size);
  auto process_group =
      std::make_shared<ProcessGroupGloo>(store, rank, size, gid, opts);
  ProcessGroupIdMap::GetInstance().emplace(gid, process_group);
  return process_group;
}

phi::distributed::GlooCommContext* ProcessGroupGloo::GetCommContext() {
  const auto& comm_context_manager =
      phi::distributed::CommContextManager::GetInstance();
  auto comm_context = static_cast<phi::distributed::GlooCommContext*>(
      comm_context_manager.Get(std::to_string(this->gid_)));
  PADDLE_ENFORCE_NE(comm_context,
                    nullptr,
                    common::errors::Unavailable("GlooCommContext is nullptr"));
  return comm_context;
}

std::vector<char> ProcessGroupGloo::GlooStore::get(const std::string& key) {
  VLOG(3) << "GlooStore::get";
  auto value = _store->get(key);
  return std::vector<char>(value.begin(), value.end());
}

void ProcessGroupGloo::GlooStore::wait(const std::vector<std::string>& keys) {
  VLOG(3) << "GlooStore::wait";
  for (auto& key : keys) {
    _store->wait(key);
  }
}

void ProcessGroupGloo::GlooStore::set(const std::string& key,
                                      const std::vector<char>& value) {
  VLOG(3) << "GlooStore::set";
  std::vector<uint8_t> tmp(value.begin(), value.end());
  _store->set(key, tmp);
}

void ProcessGroupGloo::GlooStore::wait(
    const std::vector<std::string>& keys,
    const std::chrono::milliseconds& timeout) {
  VLOG(3) << "GlooStore::wait";
  for (auto& key : keys) {
    _store->wait(key);
  }
  // wait(keys);
}

}  // namespace paddle::distributed
