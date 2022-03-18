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

#include <gloo/broadcast.h>
#include <gloo/reduce.h>
#include <gloo/scatter.h>
#include "paddle/fluid/distributed/collective/ProcessGroupGloo.h"
#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {

#ifdef _WIN32
#define GENERATE_FUNC(type, func, ...)       \
  switch (type) {                            \
    case experimental::DataType::FLOAT32:    \
      func<float>(__VA_ARGS__);              \
      break;                                 \
    case experimental::DataType::FLOAT64:    \
      func<double>(__VA_ARGS__);             \
      break;                                 \
    case experimental::DataType::FLOAT16:    \
      func<gloo::float16>(__VA_ARGS__);      \
      break;                                 \
    case experimental::DataType::INT32:      \
      func<int32_t>(__VA_ARGS__);            \
      break;                                 \
    case experimental::DataType::INT64:      \
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
    case experimental::DataType::FLOAT32:    \
      func<float>(args);                     \
      break;                                 \
    case experimental::DataType::FLOAT64:    \
      func<double>(args);                    \
      break;                                 \
    case experimental::DataType::FLOAT16:    \
      func<gloo::float16>(args);             \
      break;                                 \
    case experimental::DataType::INT32:      \
      func<int32_t>(args);                   \
      break;                                 \
    case experimental::DataType::INT64:      \
      func<int64_t>(args);                   \
      break;                                 \
    default:                                 \
      VLOG(0) << "Error: Unknown DataType."; \
      exit(-1);                              \
  }
#endif

typedef void (*reduce_func)(void*, const void*, const void*, size_t);

template <typename T>
reduce_func get_function(const ReduceOp& r) {
  switch (r) {
    case ReduceOp::SUM:
      return reduce_func(&::gloo::sum<T>);
    case ReduceOp::PRODUCT:
      return reduce_func(&::gloo::product<T>);
    case ReduceOp::MIN:
      return reduce_func(&::gloo::min<T>);
    case ReduceOp::MAX:
      return reduce_func(&::gloo::max<T>);
    case ReduceOp::AVG:
      VLOG(0) << "Error: Unsupported ReduceOp::AVG.";
      exit(-1);
  }

  VLOG(0) << "Error: Unknown ReduceOp.";
  exit(-1);
}

bool CheckTensorsInCPUPlace(const std::vector<Tensor>& tensors) {
  return std::all_of(tensors.cbegin(), tensors.cend(), [&](const Tensor& t) {
    return t.place() == PlaceType::kCPU;
  });
}

template <typename T>
T* get_data(const Tensor& tensor) {
  auto raw_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
  return static_cast<T*>(raw_tensor->data());
}

template <typename T>
std::vector<T*> get_multi_data(const std::vector<Tensor>& tensors) {
  std::vector<T*> ret(tensors.size());
  for (size_t i = 0; i < tensors.size(); i++) {
    ret[i] = get_data<T>(tensors[i]);
  }
  return ret;
}

template <typename T, typename P>
void set_output(P& opts, const Tensor& tensor) {  // NOLINT
  opts.setOutput(get_data<T>(tensor), tensor.numel());
}

template <typename T, typename P>
void set_input(P& opts, const Tensor& tensor) {  // NOLINT
  opts.setInput(get_data<T>(tensor), tensor.numel());
}

template <typename T, typename P>
void set_outputs(P& opts, const std::vector<Tensor>& tensors) {  // NOLINT
  opts.setOutputs(get_multi_data<T>(tensors), tensors[0].numel());
}

template <typename T, typename P>
void set_inputs(P& opts, const std::vector<Tensor>& tensors) {  // NOLINT
  opts.setInputs(get_multi_data<T>(tensors), tensors[0].numel());
}

template <typename T, typename P>
void set_inputs_for_scatter(P& opts,                             // NOLINT
                            const std::vector<Tensor>& tensors,  // NOLINT
                            int nranks) {
  std::vector<T*> ret(nranks);
  auto raw_tensor =
      std::dynamic_pointer_cast<phi::DenseTensor>(tensors[0].impl());
  T* raw_pointer = reinterpret_cast<T*>(raw_tensor->data());
  size_t offset = 0;
  for (int i = 0; i < nranks; i++) {
    ret[i] = raw_pointer + offset;
    offset += tensors[0].numel() / nranks;
  }
  opts.setInputs(ret, tensors[0].numel() / nranks);
}

ProcessGroupGloo::GlooTask::GlooTask(int rank,
                                     const std::vector<Tensor>& inputs,
                                     CommType comm_type)
    : ProcessGroup::Task(rank, inputs, comm_type) {
  PADDLE_ENFORCE_EQ(CheckTensorsInCPUPlace(inputs), true,
                    platform::errors::Fatal(
                        "Only CPU place is supported for ProcessGroupGloo."));
}

ProcessGroupGloo::ProcessGroupGloo(
    const std::shared_ptr<paddle::distributed::Store>& store, int rank,
    int world_size, const std::shared_ptr<GlooOptions> options)
    : ProcessGroup(rank, world_size), _tag(0), _store(new GlooStore(store)) {
  _context = std::make_shared<gloo::rendezvous::Context>(rank, world_size);
  auto prefix_store =
      ::gloo::rendezvous::PrefixStore(std::to_string(0), *_store);
  _context->connectFullMesh(prefix_store, options->device);
}

class BroadcastGlooTask : public ProcessGroupGloo::GlooTask {
 public:
  BroadcastGlooTask(const std::shared_ptr<gloo::Context>& context,
                    const std::vector<Tensor>& inputs, int rank, int root,
                    uint32_t tag)
      : ProcessGroupGloo::GlooTask(rank, inputs, CommType::BROADCAST),
        _context(context),
        _root(root),
        _inputs(inputs),
        _tag(tag) {}

  void Run() override { _do_broadcast(_inputs[0]); }

 private:
  std::shared_ptr<gloo::Context> _context;
  const int _root;
  std::vector<Tensor> _inputs{};
  const uint32_t _tag;

  void _do_broadcast(const Tensor& tensor) {
    gloo::BroadcastOptions opts(_context);
    const auto& dtype = tensor.type();
    GENERATE_FUNC(dtype, set_output, opts, tensor);
    opts.setRoot(_root);
    opts.setTag(_tag);
    gloo::broadcast(opts);
  }
};

std::shared_ptr<ProcessGroup::Task> ProcessGroupGloo::Broadcast(
    std::vector<Tensor>& inputs, const BroadcastOptions& opts) {
  auto root = opts.source_rank;
  std::unique_ptr<BroadcastGlooTask> task;
  auto tag = next_tag();
  auto context = get_context();
  task = std::make_unique<BroadcastGlooTask>(context, inputs, rank_, root, tag);
  task->Run();
  return task;
}

class AllreduceGlooTask : public ProcessGroupGloo::GlooTask {
 public:
  AllreduceGlooTask(int rank, const std::shared_ptr<gloo::Context>& context,
                    std::vector<Tensor>& inputs, ReduceOp reduce_op,  // NOLINT
                    uint32_t tag)
      : ProcessGroupGloo::GlooTask(rank, inputs, CommType::ALLREDUCE),
        _context(context),
        _inputs(inputs),
        _reduce_op(reduce_op),
        _tag(tag) {}

  void Run() override { _do_allreduce(_inputs); }

 private:
  std::shared_ptr<gloo::Context> _context;
  std::vector<Tensor> _inputs;
  const ReduceOp _reduce_op;
  uint32_t _tag;

  gloo::AllreduceOptions::Func _get_function(const experimental::DataType type,
                                             const ReduceOp op) {
    gloo::AllreduceOptions::Func fn;
    GENERATE_FUNC(type, _get_function_impl, fn, op);
    return fn;
  }

  template <typename T>
  void _get_function_impl(gloo::AllreduceOptions::Func& fn,  // NOLINT
                          const ReduceOp op) {
    fn = get_function<T>(op);
  }

  void _do_allreduce(std::vector<Tensor>& tensors) {  // NOLINT
    const auto& dtype = tensors[0].type();
    gloo::AllreduceOptions opts(_context);
    GENERATE_FUNC(dtype, set_inputs, opts, tensors);
    GENERATE_FUNC(dtype, set_outputs, opts, tensors);
    opts.setReduceFunction(_get_function(dtype, _reduce_op));
    opts.setTag(_tag);
    gloo::allreduce(opts);
  }
};

std::shared_ptr<ProcessGroup::Task> ProcessGroupGloo::AllReduce(
    std::vector<Tensor>& inputs, const AllreduceOptions& opts) {
  auto tag = next_tag();
  std::shared_ptr<GlooTask> task;
  auto context = get_context();
  task = std::make_shared<AllreduceGlooTask>(rank_, context, inputs,
                                             opts.reduce_op, tag);
  task->Run();
  return task;
}

class BarrierGlooTask : public ProcessGroupGloo::GlooTask {
 public:
  BarrierGlooTask(int rank, const std::shared_ptr<gloo::Context>& context)
      : ProcessGroupGloo::GlooTask(rank, std::vector<Tensor>{},
                                   CommType::BARRIER),
        _context(context) {}

  void Run() override { _do_barrier(); }

 private:
  std::shared_ptr<gloo::Context> _context;

  void _do_barrier() {
    gloo::BarrierOptions opts(_context);
    gloo::barrier(opts);
  }
};

std::shared_ptr<ProcessGroup::Task> ProcessGroupGloo::Barrier(
    const BarrierOptions& opts) {
  std::shared_ptr<BarrierGlooTask> task;
  auto context = get_context();
  task = std::make_shared<BarrierGlooTask>(rank_, context);
  task->Run();
  return task;
}

class AllgatherGlooTask : public ProcessGroupGloo::GlooTask {
 public:
  AllgatherGlooTask(int rank, const std::shared_ptr<gloo::Context>& context,
                    std::vector<Tensor>& inputs,   // NOLINT
                    std::vector<Tensor>& outputs,  // NOLINT
                    uint32_t tag)
      : ProcessGroupGloo::GlooTask(rank, inputs, CommType::ALLGATHER),
        _context(context),
        _inputs(inputs),
        _outputs(outputs),
        _tag(tag) {}

  void Run() override { _do_allgather(_inputs, _outputs); }

 private:
  std::shared_ptr<gloo::Context> _context;
  std::vector<Tensor> _inputs;
  std::vector<Tensor> _outputs;
  uint32_t _tag;

  void _do_allgather(std::vector<Tensor>& in,     // NOLINT
                     std::vector<Tensor>& out) {  // NOLINT
    const auto& dtype = in[0].type();
    gloo::AllgatherOptions opts(_context);
    GENERATE_FUNC(dtype, set_input, opts, in[0]);
    GENERATE_FUNC(dtype, set_output, opts, out[0]);
    opts.setTag(_tag);
    gloo::allgather(opts);
  }
};

std::shared_ptr<ProcessGroup::Task> ProcessGroupGloo::AllGather(
    std::vector<Tensor>& in_tensors, std::vector<Tensor>& out_tensors) {
  std::shared_ptr<AllgatherGlooTask> task;
  auto tag = next_tag();
  auto context = get_context();
  task = std::make_shared<AllgatherGlooTask>(rank_, context, in_tensors,
                                             out_tensors, tag);
  task->Run();
  return task;
}

class ReduceGlooTask : public ProcessGroupGloo::GlooTask {
 public:
  ReduceGlooTask(int rank, const std::shared_ptr<gloo::Context>& context,
                 std::vector<Tensor>& in, ReduceOp reduce_op,  // NOLINT
                 int dst, uint32_t tag)
      : ProcessGroupGloo::GlooTask(rank, in, CommType::REDUCE),
        _context(context),
        _inputs(in),
        _reduce_op(reduce_op),
        _dst(dst),
        _tag(tag) {}

  void Run() override { _do_reduce(_inputs, _dst); }

 private:
  std::shared_ptr<gloo::Context> _context;
  std::vector<Tensor> _inputs;
  const ReduceOp _reduce_op;
  int _dst;
  uint32_t _tag;

  gloo::ReduceOptions::Func _get_function(const experimental::DataType type,
                                          const ReduceOp op) {
    gloo::ReduceOptions::Func fn;
    GENERATE_FUNC(type, _get_function_impl, fn, op);
    return fn;
  }

  template <typename T>
  void _get_function_impl(gloo::ReduceOptions::Func& fn,  // NOLINT
                          const ReduceOp op) {
    fn = get_function<T>(op);
  }

  void _do_reduce(std::vector<Tensor>& tensors, int dst) {  // NOLINT
    const auto& dtype = tensors[0].type();
    gloo::ReduceOptions opts(_context);
    GENERATE_FUNC(dtype, set_input, opts, tensors[0]);
    GENERATE_FUNC(dtype, set_output, opts, tensors[0]);
    opts.setReduceFunction(_get_function(dtype, _reduce_op));
    opts.setTag(_tag);
    opts.setRoot(dst);
    gloo::reduce(opts);
  }
};

std::shared_ptr<ProcessGroup::Task> ProcessGroupGloo::Reduce(
    std::vector<Tensor>& tensors, const ReduceOptions& opts) {
  std::shared_ptr<ReduceGlooTask> task;
  auto tag = next_tag();
  auto context = get_context();
  task = std::make_shared<ReduceGlooTask>(rank_, context, tensors,
                                          opts.reduce_op, opts.root_rank, tag);
  task->Run();
  return task;
}

class ScatterGlooTask : public ProcessGroupGloo::GlooTask {
 public:
  ScatterGlooTask(int rank, const std::shared_ptr<gloo::Context>& context,
                  std::vector<Tensor>& inputs,   // NOLINT
                  std::vector<Tensor>& outputs,  // NOLINT
                  int src, int size, uint32_t tag)
      : ProcessGroupGloo::GlooTask(rank, inputs, CommType::SCATTER),
        _context(context),
        _inputs(inputs),
        _outputs(outputs),
        _src(src),
        _size(size),
        _tag(tag) {}

  void Run() override { _do_scatter(_inputs, _outputs, _src); }

 private:
  std::shared_ptr<gloo::Context> _context;
  std::vector<Tensor> _inputs;
  std::vector<Tensor> _outputs;
  int _src;
  int _size;
  uint32_t _tag;

  void _do_scatter(std::vector<Tensor>& in, std::vector<Tensor>& out,  // NOLINT
                   int src) {
    const auto& dtype = in[0].type();
    gloo::ScatterOptions opts(_context);
    if (rank_ == src) {
      GENERATE_FUNC(dtype, set_inputs_for_scatter, opts, in, _size);
    }
    GENERATE_FUNC(dtype, set_output, opts, out[0]);
    opts.setRoot(src);
    opts.setTag(_tag);
    gloo::scatter(opts);
  }
};

std::shared_ptr<ProcessGroup::Task> ProcessGroupGloo::Scatter(
    std::vector<Tensor>& in_tensors, std::vector<Tensor>& out_tensors,
    const ScatterOptions& opts) {
  std::shared_ptr<ScatterGlooTask> task;
  auto tag = next_tag();
  auto context = get_context();
  task = std::make_shared<ScatterGlooTask>(
      rank_, context, in_tensors, out_tensors, opts.root_rank, size_, tag);
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
  PADDLE_ENFORCE_EQ(ret, 0, platform::errors::Fatal(
                                "Get hostname error for createDefaultDevice."));
  ::addrinfo* result;
  result = tcputils::get_addr_info(hostname.data(), "", 0, AF_UNSPEC);
  ::addrinfo* cur;
  for (cur = result; cur != nullptr; cur = cur->ai_next) {
    SocketType socket =
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

}  // namespace distributed
}  // namespace paddle
