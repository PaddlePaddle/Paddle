/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <fcntl.h>
#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif

#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif

#include "paddle/fluid/distributed/collective/ProcessGroup.h"
#include "paddle/fluid/distributed/collective/ProcessGroupStream.h"
#include "paddle/fluid/distributed/collective/Types.h"
#include "paddle/fluid/distributed/collective/reducer.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/pybind/distributed_py.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/phi/api/all.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/distributed/collective/ProcessGroupNCCL.h"
#endif

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/distributed/collective/ProcessGroupHCCL.h"
#endif

#if defined(PADDLE_WITH_CUSTOM_DEVICE)
#include "paddle/fluid/distributed/collective/ProcessGroupCustom.h"
#endif

#if defined(PADDLE_WITH_GLOO) && defined(PADDLE_WITH_PSCORE) && \
    (defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_ASCEND_CL))
#include "paddle/fluid/distributed/collective/ProcessGroupHeter.h"
#endif

#if defined(PADDLE_WITH_GLOO)
#include "paddle/fluid/distributed/collective/ProcessGroupGloo.h"
#include "paddle/fluid/distributed/store/tcp_store.h"
#endif

namespace py = pybind11;

namespace paddle {
namespace pybind {

using Tensor = paddle::experimental::Tensor;

std::shared_ptr<distributed::EagerReducer> CreateEagerReducer(
    py::handle py_tensors,
    const std::vector<std::vector<size_t>> &group_indices,
    const std::vector<bool> &is_sparse_gradient,
    std::shared_ptr<distributed::ProcessGroup> process_group,
    const std::vector<size_t> &group_size_limits,
    bool find_unused_parameters) {
  auto params = CastPyArg2VectorOfTensor(py_tensors.ptr(), 0);
  return std::make_shared<distributed::EagerReducer>(params,
                                                     group_indices,
                                                     is_sparse_gradient,
                                                     process_group,
                                                     group_size_limits,
                                                     find_unused_parameters);
}

#if defined(PADDLE_WITH_GLOO)
using ProcessGroupGloo = paddle::distributed::ProcessGroupGloo;
using GlooStore = paddle::distributed::ProcessGroupGloo::GlooStore;
using GlooOptions = paddle::distributed::ProcessGroupGloo::GlooOptions;
#endif

static std::string GLOO_SOCKET_IFNAME_ENV = "GLOO_SOCKET_IFNAME";  // NOLINT

void BindDistributed(py::module *m) {
  py::enum_<distributed::ReduceOp>(*m, "ReduceOp")
      .value("SUM", distributed::ReduceOp::SUM)
      .value("AVG", distributed::ReduceOp::AVG)
      .value("MAX", distributed::ReduceOp::MAX)
      .value("MIN", distributed::ReduceOp::MIN)
      .value("PRODUCT", distributed::ReduceOp::PRODUCT);

  py::class_<distributed::AllreduceOptions>(*m, "AllreduceOptions")
      .def(py::init<>())
      .def_readwrite("reduce_op", &distributed::AllreduceOptions::reduce_op);

  py::class_<distributed::BroadcastOptions>(*m, "BroadcastOptions")
      .def(py::init<>())
      .def_readwrite("source_rank", &distributed::BroadcastOptions::source_rank)
      .def_readwrite("source_root",
                     &distributed::BroadcastOptions::source_root);

  py::class_<distributed::BarrierOptions>(*m, "BarrierOptions")
      .def(py::init<>())
      .def_readwrite("place_ids", &distributed::BarrierOptions::place_ids);

  py::class_<distributed::ReduceOptions>(*m, "ReduceOptions")
      .def(py::init<>())
      .def_readwrite("reduce_op", &distributed::ReduceOptions::reduce_op)
      .def_readwrite("source_root", &distributed::ReduceOptions::root_rank);

  auto ProcessGroup =
      py::class_<distributed::ProcessGroup,
                 std::shared_ptr<distributed::ProcessGroup>>(*m, "ProcessGroup")
          .def("rank", &distributed::ProcessGroup::GetRank)
          .def("size", &distributed::ProcessGroup::GetSize)
          .def("name", &distributed::ProcessGroup::GetBackendName)
          .def(
              "allreduce",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 distributed::ReduceOp op) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                distributed::AllreduceOptions opts;
                opts.reduce_op = op;
                auto dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                std::vector<phi::DenseTensor> tensors = {*dense};
                return self.AllReduce(tensors, tensors, opts);
              },
              py::arg("tensor"),
              py::arg("op") = distributed::ReduceOp::SUM,
              py::call_guard<py::gil_scoped_release>())

          .def(
              "allreduce",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 distributed::ReduceOp op,
                 bool sync_op) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                distributed::AllreduceOptions opts;
                opts.reduce_op = op;
                auto dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                std::vector<phi::DenseTensor> tensors = {*dense};
                return self.AllReduce(tensors, tensors, opts, sync_op);
              },
              py::arg("tensor"),
              py::arg("op"),
              py::arg("sync_op"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "broadcast",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 int source_rank) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                distributed::BroadcastOptions opts;
                opts.source_rank = source_rank;
                auto dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                std::vector<phi::DenseTensor> tensors = {*dense};
                return self.Broadcast(tensors, tensors, opts);
              },
              py::arg("tensor"),
              py::arg("source_rank"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "barrier",
              [](distributed::ProcessGroup &self, std::vector<int> place_ids) {
                distributed::BarrierOptions opts;
                opts.place_ids = place_ids;
                return self.Barrier(opts);
              },
              py::arg("place_ids") = std::vector<int>{},
              py::call_guard<py::gil_scoped_release>())

          .def(
              "send",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 int dst) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                auto dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                std::vector<phi::DenseTensor> tensors = {*dense};
                return self.Send(tensors, dst);
              },
              py::arg("tensor"),
              py::arg("dst"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "send",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 int dst,
                 bool sync_op) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                auto dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                std::vector<phi::DenseTensor> tensors = {*dense};
                return self.Send(tensors, dst, sync_op);
              },
              py::arg("tensor"),
              py::arg("dst"),
              py::arg("sync_op"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "send_partial",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 int dst_rank,
                 int nranks,
                 int rank_id) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                auto dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                int numel = (*dense).numel();
                int send_numel = numel / nranks;
                int offset = send_numel * rank_id;
                return self.Send_Partial(*dense, dst_rank, offset, send_numel);
              },
              py::arg("tensor"),
              py::arg("dst"),
              py::arg("num"),
              py::arg("id"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "send_partial",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 int dst_rank,
                 int nranks,
                 int rank_id,
                 bool sync_op) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                auto dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                int numel = (*dense).numel();
                int send_numel = numel / nranks;
                int offset = send_numel * rank_id;
                return self.Send_Partial(
                    *dense, dst_rank, offset, send_numel, sync_op);
              },
              py::arg("tensor"),
              py::arg("dst"),
              py::arg("num"),
              py::arg("id"),
              py::arg("sync_op"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "recv",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 int src) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                auto dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                std::vector<phi::DenseTensor> tensors = {*dense};
                return self.Recv(tensors, src);
              },
              py::arg("tensor"),
              py::arg("src"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "recv",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 int src,
                 bool sync_op) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                auto dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                std::vector<phi::DenseTensor> tensors = {*dense};
                return self.Recv(tensors, src, sync_op);
              },
              py::arg("tensor"),
              py::arg("src"),
              py::arg("sync_op"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "recv_partial",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 int src_rank,
                 int nranks,
                 int rank_id) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                auto dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                int numel = (*dense).numel();
                int recv_numel = numel / nranks;
                int offset = recv_numel * rank_id;
                return self.Recv_Partial(*dense, src_rank, offset, recv_numel);
              },
              py::arg("tensor"),
              py::arg("src"),
              py::arg("num"),
              py::arg("id"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "recv_partial",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 int src_rank,
                 int nranks,
                 int rank_id,
                 bool sync_op) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                auto dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                int numel = (*dense).numel();
                int recv_numel = numel / nranks;
                int offset = recv_numel * rank_id;
                return self.Recv_Partial(
                    *dense, src_rank, offset, recv_numel, sync_op);
              },
              py::arg("tensor"),
              py::arg("src"),
              py::arg("num"),
              py::arg("id"),
              py::arg("sync_op"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "all_gather",
              [](distributed::ProcessGroup &self,
                 py::handle py_in_tensor,
                 py::handle py_out_tensor) {
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto in_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                auto out_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                std::vector<phi::DenseTensor> in_tensors = {*in_dense};
                std::vector<phi::DenseTensor> out_tensors = {*out_dense};
                return self.AllGather(in_tensors, out_tensors);
              },
              py::arg("in"),
              py::arg("out"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "all_gather_partial",
              [](distributed::ProcessGroup &self,
                 py::handle py_in_tensor,
                 py::handle py_out_tensor,
                 int nranks,
                 int rank_id) {
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto in_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                auto out_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                std::vector<phi::DenseTensor> in_tensors = {*in_dense};
                std::vector<phi::DenseTensor> out_tensors = {*out_dense};
                int numel = (*in_dense).numel();
                int send_numel = numel / nranks;
                int offset = send_numel * rank_id;
                return self.AllGather_Partial(
                    in_tensors, out_tensors, offset, send_numel);
              },
              py::arg("in"),
              py::arg("out"),
              py::arg("num"),
              py::arg("id"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "alltoall",
              [](distributed::ProcessGroup &self,
                 py::handle py_in_tensor,
                 py::handle py_out_tensor) {
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto in_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                auto out_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                std::vector<phi::DenseTensor> in_tensors = {*in_dense};
                std::vector<phi::DenseTensor> out_tensors = {*out_dense};
                return self.AllToAll(in_tensors, out_tensors);
              },
              py::arg("in"),
              py::arg("out"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "alltoall_single",
              [](distributed::ProcessGroup &self,
                 py::handle py_in_tensor,
                 py::handle py_out_tensor,
                 std::vector<int64_t> in_sizes,
                 std::vector<int64_t> out_sizes) {
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto in_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                auto out_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                std::vector<phi::DenseTensor> in_tensors = {*in_dense};
                std::vector<phi::DenseTensor> out_tensors = {*out_dense};
                return self.AllToAll_Single(
                    in_tensors, out_tensors, in_sizes, out_sizes);
              },
              py::arg("in"),
              py::arg("out"),
              py::arg("in_sizes"),
              py::arg("out_sizes"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "reduce",
              [](distributed::ProcessGroup &self,
                 py::handle py_in_tensor,
                 int dst,
                 distributed::ReduceOp op) {
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                distributed::ReduceOptions opts;
                opts.reduce_op = op;
                opts.root_rank = dst;
                auto dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                std::vector<phi::DenseTensor> tensors = {*dense};
                return self.Reduce(tensors, tensors, opts);
              },
              py::arg("tensor"),
              py::arg("dst"),
              py::arg("op") = distributed::ReduceOp::SUM,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "scatter",
              [](distributed::ProcessGroup &self,
                 py::handle py_in_tensor,
                 py::handle py_out_tensor,
                 int src) {
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                distributed::ScatterOptions opts;
                opts.root_rank = src;
                auto in_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                auto out_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                std::vector<phi::DenseTensor> in_tensors = {*in_dense};
                std::vector<phi::DenseTensor> out_tensors = {*out_dense};
                return self.Scatter(in_tensors, out_tensors, opts);
              },
              py::arg("in"),
              py::arg("out"),
              py::arg("src"),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_reduce_scatter_base",
              [](distributed::ProcessGroup &self,
                 py::handle py_out_tensor,
                 py::handle py_in_tensor,
                 distributed::ReduceOp op) {
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                distributed::ReduceScatterOptions opts;
                opts.reduce_op = op;
                auto dense_out = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                auto dense_in = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                return self._ReduceScatterBase(*dense_out, *dense_in, opts);
              },
              py::arg("out_tensor"),
              py::arg("in_tensor"),
              py::arg("op") = distributed::ReduceOp::SUM,
              py::call_guard<py::gil_scoped_release>());

  auto ProcessGroupStream =
      py::class_<distributed::ProcessGroupStream,
                 std::shared_ptr<distributed::ProcessGroupStream>>(
          *m, "ProcessGroupStream", ProcessGroup)
          .def(
              "allreduce_on_calc_stream",
              [](distributed::ProcessGroupStream &self,
                 py::handle py_tensor,
                 distributed::ReduceOp op) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                distributed::AllreduceOptions opts;
                opts.reduce_op = op;
                auto dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                std::vector<phi::DenseTensor> tensors = {*dense};
                return self.AllReduce(tensors,
                                      tensors,
                                      opts,
                                      /*sync_op*/ true,
                                      /*use_calc_stream*/ true);
              },
              py::arg("tensor"),
              py::arg("op"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "send_on_calc_stream",
              [](distributed::ProcessGroupStream &self,
                 py::handle py_tensor,
                 int dst) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                auto dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                std::vector<phi::DenseTensor> tensors = {*dense};
                return self.Send(tensors,
                                 dst,
                                 /*sync_op*/ true,
                                 /*use_calc_stream*/ true);
              },
              py::arg("tensor"),
              py::arg("dst"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "send_partial_on_calc_stream",
              [](distributed::ProcessGroupStream &self,
                 py::handle py_tensor,
                 int dst_rank,
                 int nranks,
                 int rank_id) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                auto dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                int numel = (*dense).numel();
                int send_numel = numel / nranks;
                int offset = send_numel * rank_id;
                return self.Send_Partial(*dense,
                                         dst_rank,
                                         offset,
                                         send_numel,
                                         /*sync_op*/ true,
                                         /*use_calc_stream*/ true);
              },
              py::arg("tensor"),
              py::arg("dst"),
              py::arg("num"),
              py::arg("id"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "recv_on_calc_stream",
              [](distributed::ProcessGroupStream &self,
                 py::handle py_tensor,
                 int src) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                auto dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                std::vector<phi::DenseTensor> tensors = {*dense};
                return self.Recv(tensors,
                                 src,
                                 /*sync_op*/ true,
                                 /*use_calc_stream*/ true);
              },
              py::arg("tensor"),
              py::arg("src"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "recv_partial_on_calc_stream",
              [](distributed::ProcessGroupStream &self,
                 py::handle py_tensor,
                 int src_rank,
                 int nranks,
                 int rank_id) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                auto dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                int numel = (*dense).numel();
                int recv_numel = numel / nranks;
                int offset = recv_numel * rank_id;
                return self.Recv_Partial(*dense,
                                         src_rank,
                                         offset,
                                         recv_numel,
                                         /*sync_op*/ true,
                                         /*use_calc_stream*/ true);
              },
              py::arg("tensor"),
              py::arg("src"),
              py::arg("num"),
              py::arg("id"),
              py::call_guard<py::gil_scoped_release>());

#if defined(PADDLE_WITH_RCCL) || defined(PADDLE_WITH_NCCL)
  auto processGroupNCCL =
      py::class_<distributed::ProcessGroupNCCL,
                 std::shared_ptr<distributed::ProcessGroupNCCL>>(
          *m, "ProcessGroupNCCL", ProcessGroupStream)
          .def(py::init<const std::shared_ptr<distributed::Store> &,
                        int,
                        int,
                        const platform::CUDAPlace &,
                        int>(),
               py::arg("store"),
               py::arg("rank"),
               py::arg("world_size"),
               py::arg("place"),
               py::arg("group_id") = 0,
               py::call_guard<py::gil_scoped_release>());

  processGroupNCCL.def_static(
      "group_start", []() { distributed::ProcessGroupNCCL::GroupStart(); });
  processGroupNCCL.def_static(
      "group_end", []() { distributed::ProcessGroupNCCL::GroupEnd(); });

#endif

#if defined(PADDLE_WITH_GLOO) && defined(PADDLE_WITH_PSCORE) && \
    (defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_ASCEND_CL))
  py::class_<distributed::ProcessGroupHeter,
             std::shared_ptr<distributed::ProcessGroupHeter>>(
      *m, "ProcessGroupHeter", ProcessGroup)
      .def(py::init<const std::shared_ptr<distributed::Store> &,
                    int,
                    int,
#if defined(PADDLE_WITH_ASCEND_CL)
                    const platform::NPUPlace &,
#else
                    const platform::CUDAPlace &,
#endif
                    int,
                    int,
                    int,
                    int,
                    int,
                    bool,
                    std::string,
                    int,
                    int>(),
           py::arg("store"),
           py::arg("rank"),
           py::arg("world_size"),
           py::arg("place"),
           py::arg("gid") = 0,
           py::arg("local_rank") = 0,
           py::arg("local_size") = 1,
           py::arg("gloo_rank") = 0,
           py::arg("gloo_size") = 1,
           py::arg("with_switch") = false,
           py::arg("switch_endpoint") = "",
           py::arg("src_rank") = "",
           py::arg("dst_rank") = "",
           py::call_guard<py::gil_scoped_release>());
#endif

#if defined(PADDLE_WITH_ASCEND_CL)
  py::class_<distributed::ProcessGroupHCCL,
             std::shared_ptr<distributed::ProcessGroupHCCL>>(
      *m, "ProcessGroupHCCL", ProcessGroup)
      .def(py::init<const std::shared_ptr<distributed::Store> &,
                    int,
                    int,
                    const platform::NPUPlace &,
                    int>(),
           py::arg("store"),
           py::arg("rank"),
           py::arg("world_size"),
           py::arg("place"),
           py::arg("group_id") = 0,
           py::call_guard<py::gil_scoped_release>());

#endif

#if defined(PADDLE_WITH_CUSTOM_DEVICE)
  py::class_<distributed::ProcessGroupCustom,
             std::shared_ptr<distributed::ProcessGroupCustom>>(
      *m, "ProcessGroupCustom", ProcessGroup)
      .def(py::init<const std::shared_ptr<distributed::Store> &,
                    int,
                    int,
                    const platform::CustomPlace &,
                    int>(),
           py::arg("store"),
           py::arg("rank"),
           py::arg("world_size"),
           py::arg("place"),
           py::arg("group_id") = 0,
           py::call_guard<py::gil_scoped_release>());

#endif

  py::class_<distributed::ProcessGroup::Task,
             std::shared_ptr<distributed::ProcessGroup::Task>>(*m, "task")
      .def("is_completed", &distributed::ProcessGroup::Task::IsCompleted)
      .def("is_sync", &distributed::ProcessGroup::Task::IsSync)
      .def("wait",
           &distributed::ProcessGroup::Task::Wait,
           py::arg("timeout") = kWaitTimeout,
           py::call_guard<py::gil_scoped_release>())
      .def("synchronize",
           &distributed::ProcessGroup::Task::Synchronize,
           py::call_guard<py::gil_scoped_release>());

#if defined(PADDLE_WITH_GLOO)
  py::class_<ProcessGroupGloo, std::shared_ptr<ProcessGroupGloo>>(
      *m, "ProcessGroupGloo", ProcessGroup)
      .def(py::init<const std::shared_ptr<paddle::distributed::Store> &,
                    int,
                    int,
                    const platform::CPUPlace &,
                    int,
                    std::shared_ptr<GlooOptions> &>(),
           py::call_guard<py::gil_scoped_release>())
      .def(py::init([](const std::shared_ptr<paddle::distributed::Store> &store,
                       int rank,
                       int world_size,
                       const platform::CPUPlace &place,
                       int gid) {
             auto opts = GlooOptions::create();
             char *ifname = getenv(GLOO_SOCKET_IFNAME_ENV.c_str());
             if (ifname && strlen(ifname) > 1) {
               opts->device = ProcessGroupGloo::createDeviceForInterface(
                   std::string(ifname));
             } else {
               opts->device = ProcessGroupGloo::createDefaultDevice();
             }
             return std::make_shared<ProcessGroupGloo>(
                 store, rank, world_size, place, gid, opts);
           }),
           py::arg("store"),
           py::arg("rank"),
           py::arg("world_size"),
           py::arg("place"),
           py::arg("group_id") = 0,
           py::call_guard<py::gil_scoped_release>())
      .def_static("create_default_device",
                  &ProcessGroupGloo::createDefaultDevice);
#endif

  m->def(
      "eager_assign_group_by_size",
      [](py::handle py_tensors,
         std::vector<bool> is_sparse_gradient,
         std::vector<size_t> group_size_limits,
         std::vector<int64_t> tensor_indices) {
        auto tensors = CastPyArg2VectorOfTensor(py_tensors.ptr(), 0);
        return distributed::Eager_AssignGroupBySize(
            tensors, is_sparse_gradient, group_size_limits, tensor_indices);
      },
      py::arg("tensors"),
      py::arg("is_sparse_gradient"),
      py::arg("group_size_limits") = std::vector<size_t>{25 * 1024 * 1024},
      py::arg("tensor_indices") = std::vector<int64_t>{},
      py::call_guard<py::gil_scoped_release>());

  py::class_<distributed::EagerReducer,
             std::shared_ptr<distributed::EagerReducer>>(
      *m, "EagerReducer", R"DOC()DOC")
      .def(py::init(&CreateEagerReducer))
      .def(
          "prepare_for_backward",
          [](distributed::EagerReducer &self, py::handle py_tensors) {
            auto params = CastPyArg2VectorOfTensor(py_tensors.ptr(), 0);
            self.PrepareForBackward(params);
          },
          py::arg("tensors"),
          py::call_guard<py::gil_scoped_release>());
}

}  // end namespace pybind
}  // namespace paddle
