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
#include <gloo/gather.h>
#include <gloo/reduce.h>
#include <gloo/scatter.h>

#include "paddle/fluid/distributed/collective/common.h"
#include "paddle/fluid/distributed/collective/gloo_send_recv.h"
#include "paddle/fluid/distributed/collective/process_group_gloo.h"
#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {

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

template <typename T>
T* get_data(phi::DenseTensor& tensor) {  // NOLINT
  return reinterpret_cast<T*>(tensor.data());
}

template <typename T>
std::vector<T*> get_multi_data(
    std::vector<phi::DenseTensor>& tensors) {  // NOLINT
  std::vector<T*> ret;
  ret.reserve(tensors.size());
  for (size_t i = 0; i < tensors.size(); i++) {
    ret.push_back(get_data<T>(tensors[i]));
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
  auto prefix_store =
      ::gloo::rendezvous::PrefixStore(std::to_string(gid), *_store);
  _context->connectFullMesh(prefix_store, options->device);
}

class BroadcastGlooTask : public ProcessGroupGloo::GlooTask {
 public:
  BroadcastGlooTask(const std::shared_ptr<gloo::Context>& context,
                    std::vector<phi::DenseTensor>& inputs,   // NOLINT
                    std::vector<phi::DenseTensor>& outputs,  // NOLINT
                    int rank,
                    int root,
                    uint32_t tag)
      : ProcessGroupGloo::GlooTask(rank, inputs, CommType::BROADCAST),
        _context(context),
        _root(root),
        _inputs(inputs),
        _outputs(outputs),
        _tag(tag) {}

  void Run() override { _do_broadcast(_inputs[0], _outputs[0]); }

 private:
  std::shared_ptr<gloo::Context> _context;
  const int _root;
  std::vector<phi::DenseTensor> _inputs{};
  std::vector<phi::DenseTensor> _outputs{};
  const uint32_t _tag;

  void _do_broadcast(phi::DenseTensor& in, phi::DenseTensor& out) {  // NOLINT
    gloo::BroadcastOptions opts(_context);
    const auto& dtype = in.dtype();
    if (rank_ == _root) {
      GENERATE_FUNC(dtype, set_input, opts, in);
    }
    GENERATE_FUNC(dtype, set_output, opts, out);
    opts.setRoot(_root);
    opts.setTag(_tag);
    gloo::broadcast(opts);
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
  auto root = opts.source_rank;
  std::unique_ptr<BroadcastGlooTask> task;
  auto tag = next_tag();
  auto context = get_context();
  task = std::make_unique<BroadcastGlooTask>(
      context, inputs, outputs, rank_, root, tag);
  task->Run();
  return task;
}

class SendGlooTask : public ProcessGroupGloo::GlooTask {
 public:
  SendGlooTask(const std::shared_ptr<gloo::Context>& context,
               std::vector<phi::DenseTensor>* inputs,
               int rank,
               int dst_rank,
               uint32_t tag)
      : ProcessGroupGloo::GlooTask(rank, *inputs, CommType::SEND),
        _context(context),
        _inputs(*inputs),
        _dst(dst_rank),
        _tag(tag) {}

  void Run() override { _do_send(_inputs); }

 private:
  std::shared_ptr<gloo::Context> _context;
  std::vector<phi::DenseTensor> _inputs;
  int _dst;
  uint32_t _tag;

  void _do_send(std::vector<phi::DenseTensor>& in) {  // NOLINT
    SendRecvOptions opts(_context);
    const auto& dtype = in[0].dtype();
    GENERATE_FUNC(dtype, set_input, opts, in[0]);

    opts.setSrc(_context.get()->rank);
    opts.setDst(_dst);
    opts.setTag(_tag);
    send_recv(&opts);
  }
};

std::shared_ptr<ProcessGroup::Task> ProcessGroupGloo::Send(
    const phi::DenseTensor& tensor, int dst_rank, bool sync_op) {
  std::vector<phi::DenseTensor> in_wrapper{tensor};
  return Send(in_wrapper, dst_rank);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupGloo::Send(
    std::vector<phi::DenseTensor>& inputs, int dst_rank) {
  std::unique_ptr<SendGlooTask> task;
  auto tag = next_tag();
  auto context = get_context();
  task = std::make_unique<SendGlooTask>(context, &inputs, rank_, dst_rank, tag);
  task->Run();

  return task;
}

class RecvGlooTask : public ProcessGroupGloo::GlooTask {
 public:
  RecvGlooTask(const std::shared_ptr<gloo::Context>& context,
               std::vector<phi::DenseTensor>* outputs,
               int rank,
               int src_rank,
               uint32_t tag)
      : ProcessGroupGloo::GlooTask(rank, *outputs, CommType::RECV),
        _context(context),
        _outputs(*outputs),
        _src(src_rank),
        _tag(tag) {}

  void Run() override { _do_recv(_outputs); }

 private:
  std::shared_ptr<gloo::Context> _context;
  std::vector<phi::DenseTensor> _outputs;
  const int _src;
  const uint32_t _tag;

  void _do_recv(std::vector<phi::DenseTensor>& out) {  // NOLINT
    SendRecvOptions opts(_context);
    const auto& dtype = out[0].dtype();
    GENERATE_FUNC(dtype, set_output, opts, out[0]);

    opts.setSrc(_src);
    opts.setDst(_context.get()->rank);
    opts.setTag(_tag);
    send_recv(&opts);
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
  auto context = get_context();

  task =
      std::make_unique<RecvGlooTask>(context, &outputs, rank_, src_rank, tag);
  task->Run();
  return task;
}

class AllreduceGlooTask : public ProcessGroupGloo::GlooTask {
 public:
  AllreduceGlooTask(int rank,
                    const std::shared_ptr<gloo::Context>& context,
                    std::vector<phi::DenseTensor>& inputs,   // NOLINT
                    std::vector<phi::DenseTensor>& outputs,  // NOLINT
                    ReduceOp reduce_op,
                    uint32_t tag)
      : ProcessGroupGloo::GlooTask(rank, inputs, CommType::ALLREDUCE),
        _context(context),
        _inputs(inputs),
        _outputs(outputs),
        _reduce_op(reduce_op),
        _tag(tag) {}

  void Run() override { _do_allreduce(_inputs, _outputs); }

 private:
  std::shared_ptr<gloo::Context> _context;
  std::vector<phi::DenseTensor> _inputs;
  std::vector<phi::DenseTensor> _outputs;
  const ReduceOp _reduce_op;
  uint32_t _tag;

  gloo::AllreduceOptions::Func _get_function(const phi::DataType type,
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

  void _do_allreduce(std::vector<phi::DenseTensor>& ins,     // NOLINT
                     std::vector<phi::DenseTensor>& outs) {  // NOLINT
    const auto& dtype = ins[0].dtype();
    gloo::AllreduceOptions opts(_context);
    GENERATE_FUNC(dtype, set_inputs, opts, ins);
    GENERATE_FUNC(dtype, set_outputs, opts, outs);
    opts.setReduceFunction(_get_function(dtype, _reduce_op));
    opts.setTag(_tag);
    gloo::allreduce(opts);
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
  auto tag = next_tag();
  std::shared_ptr<GlooTask> task;
  auto context = get_context();
  task = std::make_shared<AllreduceGlooTask>(
      rank_, context, inputs, outputs, opts.reduce_op, tag);
  task->Run();
  return task;
}

class BarrierGlooTask : public ProcessGroupGloo::GlooTask {
 public:
  BarrierGlooTask(int rank, const std::shared_ptr<gloo::Context>& context)
      : ProcessGroupGloo::GlooTask(
            rank, std::vector<phi::DenseTensor>{}, CommType::BARRIER),
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
  AllgatherGlooTask(int rank,
                    const std::shared_ptr<gloo::Context>& context,
                    std::vector<phi::DenseTensor>& inputs,   // NOLINT
                    std::vector<phi::DenseTensor>& outputs,  // NOLINT
                    uint32_t tag)
      : ProcessGroupGloo::GlooTask(rank, inputs, CommType::ALLGATHER),
        _context(context),
        _inputs(inputs),
        _outputs(outputs),
        _tag(tag) {}

  void Run() override { _do_allgather(_inputs, _outputs); }

 private:
  std::shared_ptr<gloo::Context> _context;
  std::vector<phi::DenseTensor> _inputs;
  std::vector<phi::DenseTensor> _outputs;
  uint32_t _tag;

  void _do_allgather(std::vector<phi::DenseTensor>& in,     // NOLINT
                     std::vector<phi::DenseTensor>& out) {  // NOLINT
    const auto& dtype = in[0].dtype();
    gloo::AllgatherOptions opts(_context);
    GENERATE_FUNC(dtype, set_input, opts, in[0]);
    GENERATE_FUNC(dtype, set_output, opts, out[0]);
    opts.setTag(_tag);
    gloo::allgather(opts);
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
  std::shared_ptr<AllgatherGlooTask> task;
  auto tag = next_tag();
  auto context = get_context();
  task = std::make_shared<AllgatherGlooTask>(
      rank_, context, in_tensors, out_tensors, tag);
  task->Run();
  return task;
}

class ReduceGlooTask : public ProcessGroupGloo::GlooTask {
 public:
  ReduceGlooTask(int rank,
                 const std::shared_ptr<gloo::Context>& context,
                 std::vector<phi::DenseTensor>& inputs,   // NOLINT
                 std::vector<phi::DenseTensor>& outputs,  // NOLINT
                 ReduceOp reduce_op,
                 int dst,
                 uint32_t tag)
      : ProcessGroupGloo::GlooTask(rank, inputs, CommType::REDUCE),
        _context(context),
        _inputs(inputs),
        _outputs(outputs),
        _reduce_op(reduce_op),
        _dst(dst),
        _tag(tag) {}

  void Run() override { _do_reduce(_inputs, _outputs, _dst); }

 private:
  std::shared_ptr<gloo::Context> _context;
  std::vector<phi::DenseTensor> _inputs;
  std::vector<phi::DenseTensor> _outputs;
  const ReduceOp _reduce_op;
  int _dst;
  uint32_t _tag;

  gloo::ReduceOptions::Func _get_function(const phi::DataType type,
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

  void _do_reduce(std::vector<phi::DenseTensor>& inputs,   // NOLINT
                  std::vector<phi::DenseTensor>& outputs,  // NOLINT
                  int dst) {
    const auto& dtype = inputs[0].dtype();
    gloo::ReduceOptions opts(_context);
    GENERATE_FUNC(dtype, set_input, opts, inputs[0]);
    GENERATE_FUNC(dtype, set_output, opts, outputs[0]);
    opts.setReduceFunction(_get_function(dtype, _reduce_op));
    opts.setTag(_tag);
    opts.setRoot(dst);
    gloo::reduce(opts);
  }
};

std::shared_ptr<ProcessGroup::Task> ProcessGroupGloo::Reduce(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const ReduceOptions& opts,
    bool sync_op  // for compatibility, no use now
) {
  std::shared_ptr<ReduceGlooTask> task;
  auto tag = next_tag();
  auto context = get_context();
  std::vector<phi::DenseTensor> in_wrapper{in_tensor};
  std::vector<phi::DenseTensor> out_wrapper{*out_tensor};
  task = std::make_shared<ReduceGlooTask>(rank_,
                                          context,
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
                  const std::shared_ptr<gloo::Context>& context,
                  std::vector<phi::DenseTensor>& inputs,   // NOLINT
                  std::vector<phi::DenseTensor>& outputs,  // NOLINT
                  int src,
                  int size,
                  uint32_t tag)
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
  std::vector<phi::DenseTensor> _inputs;
  std::vector<phi::DenseTensor> _outputs;
  int _src;
  int _size;
  uint32_t _tag;

  void _do_scatter(std::vector<phi::DenseTensor>& in,   // NOLINT
                   std::vector<phi::DenseTensor>& out,  // NOLINT
                   int src) {
    const auto& dtype = in[0].dtype();
    gloo::ScatterOptions opts(_context);
    if (rank_ == src) {
      GENERATE_FUNC(dtype, set_inputs_for_scatter, opts, in[0], _size);
    }
    GENERATE_FUNC(dtype, set_output, opts, out[0]);
    opts.setRoot(src);
    opts.setTag(_tag);
    gloo::scatter(opts);
  }
};

std::shared_ptr<ProcessGroup::Task> ProcessGroupGloo::Scatter(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const ScatterOptions& opts,
    bool sync_op) {
  std::shared_ptr<ScatterGlooTask> task;
  auto tag = next_tag();
  auto context = get_context();
  std::vector<phi::DenseTensor> in_wrapper{in_tensor};
  std::vector<phi::DenseTensor> out_wrapper{*out_tensor};
  task = std::make_shared<ScatterGlooTask>(
      rank_, context, in_wrapper, out_wrapper, opts.root_rank, size_, tag);
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
                 const std::shared_ptr<gloo::Context>& context,
                 const phi::DenseTensor& input,  // NOLINT
                 phi::DenseTensor* output,       // NOLINT
                 int src,
                 uint32_t tag)
      : ProcessGroupGloo::GlooTask(rank, {input}, CommType::GATHER),
        _context(context),
        _input(input),
        _output(*output),
        _src(src),
        _tag(tag) {}

  void Run() override { _do_gather(_input, _output, _src); }

 private:
  std::shared_ptr<gloo::Context> _context;
  phi::DenseTensor _input;
  phi::DenseTensor _output;
  int _src;
  uint32_t _tag;

  void _do_gather(phi::DenseTensor& in,   // NOLINT
                  phi::DenseTensor& out,  // NOLINT
                  int src) {
    const auto& dtype = in.dtype();
    gloo::GatherOptions opts(_context);
    if (rank_ == src) {
      GENERATE_FUNC(dtype, set_output, opts, out);
    }
    GENERATE_FUNC(dtype, set_input, opts, in);

    opts.setRoot(src);
    opts.setTag(_tag);
    gloo::gather(opts);
  }
};

std::shared_ptr<ProcessGroup::Task> ProcessGroupGloo::Gather(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const GatherOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  PADDLE_ENFORCE_NE(
      use_calc_stream,
      true,
      platform::errors::InvalidArgument("Gloo cannot use use_calc_stream."));
  std::shared_ptr<GatherGlooTask> task;
  auto tag = next_tag();
  auto context = get_context();
  task = std::make_shared<GatherGlooTask>(
      rank_, context, in_tensor, out_tensor, opts.root_rank, tag);
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
      platform::errors::Fatal("Get hostname error for createDefaultDevice."));
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
  auto process_group =
      std::make_shared<ProcessGroupGloo>(store, rank, size, gid, opts);
  ProcessGroupIdMap::GetInstance().emplace(gid, process_group);
  return process_group;
}

}  // namespace distributed
}  // namespace paddle
