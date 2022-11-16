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
#include "paddle/fluid/pybind/process_group_utils.h"
#include "paddle/phi/api/all.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/distributed/collective/ProcessGroupNCCL.h"
#endif

#if defined(PADDLE_WITH_MPI)
#include "paddle/fluid/distributed/collective/ProcessGroupMPI.h"
#endif

#if defined(PADDLE_WITH_CUSTOM_DEVICE)
#include "paddle/fluid/distributed/collective/ProcessGroupCustom.h"
#endif

#if defined(PADDLE_WITH_GLOO)
#include "paddle/fluid/distributed/collective/ProcessGroupGloo.h"
#include "paddle/fluid/distributed/store/tcp_store.h"
#endif

#if defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/fluid/distributed/collective/ProcessGroupBKCL.h"
#endif

#include "paddle/phi/kernels/sync_batch_norm_kernel.h"

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

static UNUSED void *use_ccl_comm_func =
    phi::detail::GetCCLComm(phi::CPUPlace());

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
      .def_readwrite("device_id", &distributed::BarrierOptions::device_id);

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
              "all_reduce",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 distributed::ReduceOp op,
                 bool sync_op) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                auto p_dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                auto *out_dense = p_dense.get();
                auto in_dense = *p_dense;
                distributed::AllreduceOptions opts{op};
                return self.AllReduce(out_dense, in_dense, opts, sync_op);
              },
              py::arg("tensor"),
              py::arg("op"),
              py::arg("sync_op"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "broadcast",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 int src,
                 bool sync_op) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                auto p_dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                auto *out_dense = p_dense.get();
                auto in_dense = *p_dense;
                distributed::BroadcastOptions opts{src};
                return self.Broadcast(out_dense, in_dense, opts, sync_op);
              },
              py::arg("tensor"),
              py::arg("src"),
              py::arg("sync_op"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "send",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 int dst,
                 bool sync_op) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                auto p_dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                auto *out_dense = p_dense.get();
                // numel == -1 indicates sending the whole tensor
                return self.Send(
                    out_dense, dst, /*offset*/ 0, /*numel*/ -1, sync_op);
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
                 int rank_id,
                 bool sync_op) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                auto p_dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                auto *out_dense = p_dense.get();

                int64_t numel = p_dense->numel();
                int64_t send_numel = numel / nranks;
                int64_t offset = send_numel * rank_id;

                return self.Send(
                    out_dense, dst_rank, offset, send_numel, sync_op);
              },
              py::arg("tensor"),
              py::arg("dst"),
              py::arg("num"),
              py::arg("id"),
              py::arg("sync_op") = true,
              py::call_guard<py::gil_scoped_release>())

          .def(
              "recv",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 int src,
                 bool sync_op) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                auto p_dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                auto *in_dense = p_dense.get();
                // numel == -1 indicates receiving the whole tensor
                return self.Recv(
                    in_dense, src, /*offset*/ 0, /*numel*/ -1, sync_op);
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
                 int rank_id,
                 bool sync_op) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                auto p_dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                auto *out_dense = p_dense.get();

                int64_t numel = p_dense->numel();
                int64_t recv_numel = numel / nranks;
                int64_t offset = recv_numel * rank_id;

                return self.Recv(
                    out_dense, src_rank, offset, recv_numel, sync_op);
              },
              py::arg("tensor"),
              py::arg("src"),
              py::arg("num"),
              py::arg("id"),
              py::arg("sync_op") = true,
              py::call_guard<py::gil_scoped_release>())

          .def(
              "all_gather",
              [](distributed::ProcessGroup &self,
                 py::handle py_out_tensor_list,
                 py::handle py_in_tensor,
                 bool sync_op) {
                auto out_tensor_list =
                    CastPyArg2VectorOfTensor(py_out_tensor_list.ptr(), 0);
                Tensor concat_out_tensor = paddle::concat(out_tensor_list, 0);
                auto p_out_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    concat_out_tensor.impl());
                auto *out_dense = p_out_tensor.get();

                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                auto p_in_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                auto in_dense = *p_in_tensor;

                const auto &dev_ctx = self.GetDeviceContext(in_tensor.place());
                auto task = self.AllGather(out_dense,
                                           in_dense,
                                           /*offset*/ 0,
                                           /*numel*/ -1,
                                           sync_op);
                distributed::SplitTensor(dev_ctx, *out_dense, &out_tensor_list);
                task->UpdateWaitChain(dev_ctx);
                return task;
              },
              py::arg("out"),
              py::arg("in"),
              py::arg("sync_op"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "all_gather_into_tensor",
              [](distributed::ProcessGroup &self,
                 py::handle py_out_tensor,
                 py::handle py_in_tensor,
                 bool sync_op) {
                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto p_out_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                auto *out_dense = p_out_tensor.get();

                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                auto p_in_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                auto in_dense = *p_in_tensor;

                return self.AllGather(out_dense,
                                      in_dense,
                                      /*offset*/ 0,
                                      /*numel*/ -1,
                                      sync_op);
              },
              py::arg("out"),
              py::arg("in"),
              py::arg("sync_op"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "all_to_all",
              [](distributed::ProcessGroup &self,
                 py::handle py_in_tensor_list,
                 py::handle py_out_tensor_list,
                 bool sync_op) {
                auto in_tensor_list =
                    CastPyArg2VectorOfTensor(py_in_tensor_list.ptr(), 0);
                Tensor concat_in_tensor = paddle::concat(in_tensor_list, 0);
                auto in_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    concat_in_tensor.impl());
                std::vector<phi::DenseTensor> in_wrapper = {*in_dense};

                auto out_tensor_list =
                    CastPyArg2VectorOfTensor(py_out_tensor_list.ptr(), 0);
                Tensor concat_out_tensor = paddle::concat(out_tensor_list, 0);
                auto out_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    concat_out_tensor.impl());
                std::vector<phi::DenseTensor> out_wrapper = {*out_dense};

                // in_tensor_list should not be empty
                const auto &dev_ctx =
                    self.GetDeviceContext(in_tensor_list.back().place());
                auto task = self.AllToAll(in_wrapper, out_wrapper, sync_op);
                distributed::SplitTensor(dev_ctx, *out_dense, &out_tensor_list);
                task->UpdateWaitChain(dev_ctx);
                return task;
              },
              py::arg("in"),
              py::arg("out"),
              py::arg("sync_op"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "all_to_all_tensor",
              [](distributed::ProcessGroup &self,
                 py::handle py_in_tensor,
                 py::handle py_out_tensor,
                 bool sync_op) {
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                auto in_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                std::vector<phi::DenseTensor> in_wrapper = {*in_dense};

                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto out_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                std::vector<phi::DenseTensor> out_wrapper = {*out_dense};

                return self.AllToAll(in_wrapper, out_wrapper, sync_op);
              },
              py::arg("in"),
              py::arg("out"),
              py::arg("sync_op"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "all_to_all_single",
              [](distributed::ProcessGroup &self,
                 py::handle py_in_tensor,
                 py::handle py_out_tensor,
                 std::vector<int64_t> &in_sizes,
                 std::vector<int64_t> &out_sizes,
                 bool sync_op) {
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                auto in_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                std::vector<phi::DenseTensor> in_wrapper = {*in_dense};

                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto out_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                std::vector<phi::DenseTensor> out_wrapper = {*out_dense};

                return self.AllToAllSingle(
                    in_wrapper, out_wrapper, in_sizes, out_sizes, sync_op);
              },
              py::arg("in"),
              py::arg("out"),
              py::arg("in_sizes"),
              py::arg("out_sizes"),
              py::arg("sync_op"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "reduce",
              [](distributed::ProcessGroup &self,
                 py::handle py_in_tensor,
                 int dst,
                 distributed::ReduceOp op,
                 bool sync_op) {
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                distributed::ReduceOptions opts{op, dst};
                auto dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                std::vector<phi::DenseTensor> tensors = {*dense};
                return self.Reduce(tensors, tensors, opts, sync_op);
              },
              py::arg("tensor"),
              py::arg("dst"),
              py::arg("op"),
              py::arg("sync_op"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "reduce_scatter",
              [](distributed::ProcessGroup &self,
                 py::handle py_in_tensor_list,
                 py::handle py_out_tensor,
                 distributed::ReduceOp op,
                 bool sync_op) {
                auto in_tensor_list =
                    CastPyArg2VectorOfTensor(py_in_tensor_list.ptr(), 0);
                Tensor concat_in_tensor = paddle::concat(in_tensor_list, 0);
                auto in_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    concat_in_tensor.impl());
                std::vector<phi::DenseTensor> in_wrapper = {*in_dense};

                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto out_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                std::vector<phi::DenseTensor> out_wrapper = {*out_dense};

                distributed::ReduceScatterOptions opts{op};
                return self.ReduceScatter(
                    in_wrapper, out_wrapper, opts, sync_op);
              },
              py::arg("in"),
              py::arg("out"),
              py::arg("op"),
              py::arg("sync_op"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "reduce_scatter_tensor",
              [](distributed::ProcessGroup &self,
                 py::handle py_in_tensor,
                 py::handle py_out_tensor,
                 distributed::ReduceOp op,
                 bool sync_op) {
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                auto in_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                std::vector<phi::DenseTensor> in_wrapper = {*in_dense};

                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto out_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                std::vector<phi::DenseTensor> out_wrapper = {*out_dense};

                distributed::ReduceScatterOptions opts{op};
                return self.ReduceScatter(
                    in_wrapper, out_wrapper, opts, sync_op);
              },
              py::arg("in"),
              py::arg("out"),
              py::arg("op"),
              py::arg("sync_op"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "scatter",
              [](distributed::ProcessGroup &self,
                 py::handle py_in_tensor_list,
                 py::handle py_out_tensor,
                 int src,
                 bool sync_op) {
                auto in_tensor_list =
                    CastPyArg2VectorOfTensor(py_in_tensor_list.ptr(), 0);
                Tensor concat_in_tensor = paddle::concat(in_tensor_list, 0);
                auto in_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    concat_in_tensor.impl());
                std::vector<phi::DenseTensor> in_wrapper = {*in_dense};

                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto out_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                std::vector<phi::DenseTensor> out_wrapper = {*out_dense};

                distributed::ScatterOptions opts{src};
                return self.Scatter(in_wrapper, out_wrapper, opts, sync_op);
              },
              py::arg("in"),
              py::arg("out"),
              py::arg("src"),
              py::arg("sync_op"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "scatter_tensor",
              [](distributed::ProcessGroup &self,
                 py::handle py_in_tensor,
                 py::handle py_out_tensor,
                 int src,
                 bool sync_op) {
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                auto in_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                std::vector<phi::DenseTensor> in_wrapper = {*in_dense};

                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto out_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                std::vector<phi::DenseTensor> out_wrapper = {*out_dense};

                distributed::ScatterOptions opts{src};
                return self.Scatter(in_wrapper, out_wrapper, opts, sync_op);
              },
              py::arg("in"),
              py::arg("out"),
              py::arg("src"),
              py::arg("sync_op"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "barrier",
              [](distributed::ProcessGroup &self, int8_t device_id) {
                distributed::BarrierOptions opts;
                opts.device_id = device_id;
                return self.Barrier(opts);
              },
              py::arg("device_id") = -1,
              py::call_guard<py::gil_scoped_release>())

          // TODO(liyurui): Interface below will be removed in the future.
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
                 py::handle py_out_tensor,
                 py::handle py_in_tensor,
                 int nranks,
                 int rank_id) {
                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto p_out_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                auto *out_dense = p_out_tensor.get();

                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                auto p_in_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                auto in_dense = *p_in_tensor;

                int64_t numel = in_dense.numel();
                int64_t send_numel = numel / nranks;
                int64_t offset = send_numel * rank_id;
                return self.AllGather(
                    out_dense, in_dense, offset, send_numel, /*sync_op*/ true);
              },
              py::arg("out"),
              py::arg("in"),
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
              py::call_guard<py::gil_scoped_release>());

  auto ProcessGroupStream =
      py::class_<distributed::ProcessGroupStream,
                 std::shared_ptr<distributed::ProcessGroupStream>>(
          *m, "ProcessGroupStream", ProcessGroup)
          .def(
              "all_gather_on_calc_stream",
              [](distributed::ProcessGroupStream &self,
                 py::handle py_out_tensor_list,
                 py::handle py_in_tensor) {
                auto out_tensor_list =
                    CastPyArg2VectorOfTensor(py_out_tensor_list.ptr(), 0);
                Tensor concat_out_tensor = paddle::concat(out_tensor_list, 0);
                auto p_out_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    concat_out_tensor.impl());
                auto *out_dense = p_out_tensor.get();

                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                auto p_in_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                auto in_dense = *p_in_tensor;

                const auto &dev_ctx =
                    self.GetDeviceContext(in_tensor.place(), true);
                auto task = self.AllGather(out_dense,
                                           in_dense,
                                           /*offset*/ 0,
                                           /*numel*/ -1,
                                           /*sync_op*/ true,
                                           /*use_calc_stream*/ true);
                distributed::SplitTensor(dev_ctx, *out_dense, &out_tensor_list);
                return task;
              },
              py::arg("out"),
              py::arg("in"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "all_gather_into_tensor_on_calc_stream",
              [](distributed::ProcessGroupStream &self,
                 py::handle py_out_tensor,
                 py::handle py_in_tensor) {
                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto p_out_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                auto *out_dense = p_out_tensor.get();

                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                auto p_in_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                auto in_dense = *p_in_tensor;

                return self.AllGather(out_dense,
                                      in_dense,
                                      /*offset*/ 0,
                                      /*numel*/ -1,
                                      /*sync_op*/ true,
                                      /*use_calc_stream*/ true);
              },
              py::arg("out"),
              py::arg("in"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "all_gather_partial_on_calc_stream",
              [](distributed::ProcessGroupStream &self,
                 py::handle py_out_tensor,
                 py::handle py_in_tensor,
                 int nranks,
                 int rank_id) {
                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto p_out_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                auto *out_dense = p_out_tensor.get();

                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                auto p_in_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                auto in_dense = *p_in_tensor;

                int64_t numel = in_dense.numel();
                int64_t send_numel = numel / nranks;
                int64_t offset = send_numel * rank_id;

                return self.AllGather(out_dense,
                                      in_dense,
                                      offset,
                                      send_numel,
                                      /*sync_op*/ true,
                                      /*use_calc_stream*/ true);
              },
              py::arg("out"),
              py::arg("in"),
              py::arg("num"),
              py::arg("id"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "all_reduce_on_calc_stream",
              [](distributed::ProcessGroupStream &self,
                 py::handle py_tensor,
                 distributed::ReduceOp op) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                auto p_dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                auto in_dense = *p_dense;
                auto *out_dense = p_dense.get();
                distributed::AllreduceOptions opts{op};
                return self.AllReduce(out_dense,
                                      in_dense,
                                      opts,
                                      /*sync_op*/ true,
                                      /*use_calc_stream*/ true);
              },
              py::arg("tensor"),
              py::arg("op") = distributed::ReduceOp::SUM,
              py::call_guard<py::gil_scoped_release>())

          .def(
              "all_to_all_on_calc_stream",
              [](distributed::ProcessGroupStream &self,
                 py::handle py_in_tensor_list,
                 py::handle py_out_tensor_list) {
                auto in_tensor_list =
                    CastPyArg2VectorOfTensor(py_in_tensor_list.ptr(), 0);
                Tensor concat_in_tensor = paddle::concat(in_tensor_list, 0);
                auto in_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    concat_in_tensor.impl());
                std::vector<phi::DenseTensor> in_wrapper = {*in_dense};

                auto out_tensor_list =
                    CastPyArg2VectorOfTensor(py_out_tensor_list.ptr(), 0);
                Tensor concat_out_tensor = paddle::concat(out_tensor_list, 0);
                auto out_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    concat_out_tensor.impl());
                std::vector<phi::DenseTensor> out_wrapper = {*out_dense};

                // in_tensor_list must not be empty
                const auto &dev_ctx = self.GetDeviceContext(
                    in_tensor_list.back().place(), /*use_calc_stream*/ true);
                auto task = self.AllToAll(in_wrapper,
                                          out_wrapper,
                                          /*sync_op*/ true,
                                          /*use_calc_stream*/ true);
                distributed::SplitTensor(dev_ctx, *out_dense, &out_tensor_list);
                return task;
              },
              py::arg("in"),
              py::arg("out"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "all_to_all_tensor_on_calc_stream",
              [](distributed::ProcessGroupStream &self,
                 py::handle py_in_tensor,
                 py::handle py_out_tensor) {
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                auto in_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                std::vector<phi::DenseTensor> in_wrapper = {*in_dense};

                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto out_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                std::vector<phi::DenseTensor> out_wrapper = {*out_dense};

                return self.AllToAll(in_wrapper,
                                     out_wrapper,
                                     /*sync_op*/ true,
                                     /*use_calc_stream*/ true);
              },
              py::arg("in"),
              py::arg("out"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "all_to_all_single_on_calc_stream",
              [](distributed::ProcessGroupStream &self,
                 py::handle py_in_tensor,
                 py::handle py_out_tensor,
                 std::vector<int64_t> &in_sizes,
                 std::vector<int64_t> &out_sizes) {
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                auto in_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                std::vector<phi::DenseTensor> in_wrapper = {*in_dense};

                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto out_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                std::vector<phi::DenseTensor> out_wrapper = {*out_dense};

                return self.AllToAllSingle(in_wrapper,
                                           out_wrapper,
                                           in_sizes,
                                           out_sizes,
                                           /*sync_op*/ true,
                                           /*use_calc_stream*/ true);
              },
              py::arg("in"),
              py::arg("out"),
              py::arg("in_sizes"),
              py::arg("out_sizes"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "broadcast_on_calc_stream",
              [](distributed::ProcessGroupStream &self,
                 py::handle py_tensor,
                 int src) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                auto p_dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                auto *out_dense = p_dense.get();
                auto in_dense = *p_dense;
                distributed::BroadcastOptions opts{src};
                return self.Broadcast(out_dense,
                                      in_dense,
                                      opts,
                                      /*sync_op*/ true,
                                      /*use_calc_stream*/ true);
              },
              py::arg("tensor"),
              py::arg("src"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "reduce_on_calc_stream",
              [](distributed::ProcessGroupStream &self,
                 py::handle py_in_tensor,
                 int dst,
                 distributed::ReduceOp op) {
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                distributed::ReduceOptions opts{op, dst};
                auto dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                std::vector<phi::DenseTensor> tensors = {*dense};
                return self.Reduce(tensors,
                                   tensors,
                                   opts,
                                   /*sync_op*/ true,
                                   /*use_calc_stream*/ true);
              },
              py::arg("tensor"),
              py::arg("dst"),
              py::arg("op"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "reduce_scatter_on_calc_stream",
              [](distributed::ProcessGroupStream &self,
                 py::handle py_in_tensor_list,
                 py::handle py_out_tensor,
                 distributed::ReduceOp op) {
                auto in_tensor_list =
                    CastPyArg2VectorOfTensor(py_in_tensor_list.ptr(), 0);
                Tensor concat_in_tensor = paddle::concat(in_tensor_list, 0);
                auto in_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    concat_in_tensor.impl());
                std::vector<phi::DenseTensor> in_wrapper = {*in_dense};

                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto out_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                std::vector<phi::DenseTensor> out_wrapper = {*out_dense};

                distributed::ReduceScatterOptions opts{op};
                return self.ReduceScatter(in_wrapper,
                                          out_wrapper,
                                          opts,
                                          /*sync_op*/ true,
                                          /*use_calc_stream*/ true);
              },
              py::arg("in"),
              py::arg("out"),
              py::arg("op"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "reduce_scatter_tensor_on_calc_stream",
              [](distributed::ProcessGroupStream &self,
                 py::handle py_in_tensor,
                 py::handle py_out_tensor,
                 distributed::ReduceOp op) {
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                auto in_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                std::vector<phi::DenseTensor> in_wrapper = {*in_dense};

                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto out_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                std::vector<phi::DenseTensor> out_wrapper = {*out_dense};

                distributed::ReduceScatterOptions opts{op};
                return self.ReduceScatter(in_wrapper,
                                          out_wrapper,
                                          opts,
                                          /*sync_op*/ true,
                                          /*use_calc_stream*/ true);
              },
              py::arg("in"),
              py::arg("out"),
              py::arg("op"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "scatter_on_calc_stream",
              [](distributed::ProcessGroupStream &self,
                 py::handle py_in_tensor_list,
                 py::handle py_out_tensor,
                 int src) {
                auto in_tensor_list =
                    CastPyArg2VectorOfTensor(py_in_tensor_list.ptr(), 0);
                Tensor concat_in_tensor = paddle::concat(in_tensor_list, 0);
                auto in_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    concat_in_tensor.impl());
                std::vector<phi::DenseTensor> in_wrapper = {*in_dense};

                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto out_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                std::vector<phi::DenseTensor> out_wrapper = {*out_dense};

                distributed::ScatterOptions opts{src};
                return self.Scatter(in_wrapper,
                                    out_wrapper,
                                    opts,
                                    /*sync_op*/ true,
                                    /*use_calc_stream*/ true);
              },
              py::arg("in"),
              py::arg("out"),
              py::arg("src"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "scatter_tensor_on_calc_stream",
              [](distributed::ProcessGroupStream &self,
                 py::handle py_in_tensor,
                 py::handle py_out_tensor,
                 int src) {
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                auto in_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                std::vector<phi::DenseTensor> in_wrapper = {*in_dense};

                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto out_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                std::vector<phi::DenseTensor> out_wrapper = {*out_dense};

                distributed::ScatterOptions opts{src};
                return self.Scatter(in_wrapper,
                                    out_wrapper,
                                    opts,
                                    /*sync_op*/ true,
                                    /*use_calc_stream*/ true);
              },
              py::arg("in"),
              py::arg("out"),
              py::arg("src"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "send_on_calc_stream",
              [](distributed::ProcessGroupStream &self,
                 py::handle py_tensor,
                 int dst) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                auto p_dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                auto *out_dense = p_dense.get();
                // numel == -1 indicates sending the whole tensor
                return self.Send(out_dense,
                                 dst,
                                 /*offset*/ 0,
                                 /*numel*/ -1,
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
                auto p_dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                auto *out_dense = p_dense.get();

                int64_t numel = p_dense->numel();
                int64_t send_numel = numel / nranks;
                int64_t offset = send_numel * rank_id;

                return self.Send(out_dense,
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
                auto p_dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                auto *in_dense = p_dense.get();
                // numel == -1 indicates receiving the whole tensor
                return self.Recv(in_dense,
                                 src,
                                 /*offset*/ 0,
                                 /*numel*/ -1,
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
                auto p_dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                auto *out_dense = p_dense.get();

                int64_t numel = p_dense->numel();
                int64_t recv_numel = numel / nranks;
                int64_t offset = recv_numel * rank_id;

                return self.Recv(out_dense,
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
                        int>(),
               py::arg("store"),
               py::arg("rank"),
               py::arg("world_size"),
               py::arg("group_id") = 0,
               py::call_guard<py::gil_scoped_release>());

  processGroupNCCL.def_static(
      "group_start", []() { distributed::ProcessGroupNCCL::GroupStart(); });
  processGroupNCCL.def_static(
      "group_end", []() { distributed::ProcessGroupNCCL::GroupEnd(); });

#endif

#if defined(PADDLE_WITH_MPI)
  py::class_<distributed::ProcessGroupMPI,
             std::shared_ptr<distributed::ProcessGroupMPI>>(
      *m, "ProcessGroupMPI", ProcessGroup)
      .def_static(
          "create",
          [](const std::vector<int> &ranks,
             int gid) -> std::shared_ptr<distributed::ProcessGroupMPI> {
            return paddle::distributed::ProcessGroupMPI::CreateProcessGroupMPI(
                ranks, gid);
          })
      .def("get_rank",
           &distributed::ProcessGroup::GetRank,
           py::call_guard<py::gil_scoped_release>())
      .def("get_world_size",
           &distributed::ProcessGroup::GetSize,
           py::call_guard<py::gil_scoped_release>());
#endif

#if defined(PADDLE_WITH_CUSTOM_DEVICE)
  py::class_<distributed::ProcessGroupCustom,
             std::shared_ptr<distributed::ProcessGroupCustom>>(
      *m, "ProcessGroupCustom", ProcessGroup)
      .def(py::init<const std::shared_ptr<distributed::Store> &,
                    const std::string &,
                    int,
                    int,
                    int>(),
           py::arg("store"),
           py::arg("device_type"),
           py::arg("rank"),
           py::arg("world_size"),
           py::arg("group_id") = 0,
           py::call_guard<py::gil_scoped_release>());

#endif

#if defined(PADDLE_WITH_XPU_BKCL)
  auto processGroupBKCL =
      py::class_<distributed::ProcessGroupBKCL,
                 std::shared_ptr<distributed::ProcessGroupBKCL>>(
          *m, "ProcessGroupBKCL", ProcessGroup)
          .def(py::init<const std::shared_ptr<distributed::Store> &,
                        int,
                        int,
                        int>(),
               py::arg("store"),
               py::arg("rank"),
               py::arg("world_size"),
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
                    int,
                    std::shared_ptr<GlooOptions> &>(),
           py::call_guard<py::gil_scoped_release>())
      .def(py::init([](const std::shared_ptr<paddle::distributed::Store> &store,
                       int rank,
                       int world_size,
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
                 store, rank, world_size, gid, opts);
           }),
           py::arg("store"),
           py::arg("rank"),
           py::arg("world_size"),
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
