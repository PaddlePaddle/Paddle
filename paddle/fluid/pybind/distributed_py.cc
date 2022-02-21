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
#include "paddle/fluid/distributed/collective/Types.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/pybind/distributed_py.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/phi/api/all.h"

#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/distributed/collective/ProcessGroupNCCL.h"
#endif

namespace py = pybind11;

namespace paddle {
namespace pybind {

using Tensor = paddle::experimental::Tensor;

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

  auto ProcessGroup =
      py::class_<distributed::ProcessGroup,
                 std::shared_ptr<distributed::ProcessGroup>>(*m, "ProcessGroup")
          .def("rank", &distributed::ProcessGroup::GetRank)
          .def("size", &distributed::ProcessGroup::GetSize)
          .def("name", &distributed::ProcessGroup::GetBackendName)
          .def("allreduce",
               [](distributed::ProcessGroup &self, py::handle py_tensor,
                  distributed::ReduceOp op) {
                 auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                 distributed::AllreduceOptions opts;
                 opts.reduce_op = op;
                 std::vector<Tensor> tensors = {tensor};
                 return self.AllReduce(tensors, opts);
               },
               py::arg("tensor"), py::arg("op") = distributed::ReduceOp::SUM,
               py::call_guard<py::gil_scoped_release>())

          .def("broadcast",
               [](distributed::ProcessGroup &self, py::handle py_tensor,
                  int source_rank) {
                 auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                 distributed::BroadcastOptions opts;
                 opts.source_rank = source_rank;
                 std::vector<Tensor> tensors = {tensor};
                 return self.Broadcast(tensors, opts);
               },
               py::arg("tensor"), py::arg("source_rank"),
               py::call_guard<py::gil_scoped_release>());

#if defined(PADDLE_WITH_NCCL)
  py::class_<distributed::ProcessGroupNCCL,
             std::shared_ptr<distributed::ProcessGroupNCCL>>(
      *m, "ProcessGroupNCCL", ProcessGroup)
      .def(py::init<const distributed::ProcessGroupStrategy &, int, int>(),
           py::call_guard<py::gil_scoped_release>());

  py::class_<distributed::ProcessGroup::Task,
             std::shared_ptr<distributed::ProcessGroup::Task>>(*m, "task")
      .def("is_completed", &distributed::ProcessGroup::Task::IsCompleted)
      .def("wait", &distributed::ProcessGroup::Task::Wait,
           py::arg("timeout") = kWaitTimeout,
           py::call_guard<py::gil_scoped_release>())
      .def("synchronize", &distributed::ProcessGroup::Task::Synchronize,
           py::call_guard<py::gil_scoped_release>());
#endif

  // define parallel strategy, it will be removed
  py::class_<distributed::ProcessGroupStrategy> pg_strategy(
      *m, "ProcessGroupStrategy", "");
  pg_strategy.def(py::init())
      .def_property("nranks",
                    [](const distributed::ProcessGroupStrategy &self) {
                      return self.nranks_;
                    },
                    [](distributed::ProcessGroupStrategy &self, int nranks) {
                      self.nranks_ = nranks;
                    })
      .def_property("local_rank",
                    [](const distributed::ProcessGroupStrategy &self) {
                      return self.local_rank_;
                    },
                    [](distributed::ProcessGroupStrategy &self,
                       int local_rank) { self.local_rank_ = local_rank; })
      .def_property(
          "trainer_endpoints",
          [](const distributed::ProcessGroupStrategy &self) {
            return self.trainer_endpoints_;
          },
          [](distributed::ProcessGroupStrategy &self,
             std::vector<std::string> eps) { self.trainer_endpoints_ = eps; })
      .def_property("current_endpoint",
                    [](const distributed::ProcessGroupStrategy &self) {
                      return self.current_endpoint_;
                    },
                    [](distributed::ProcessGroupStrategy &self,
                       const std::string &ep) { self.current_endpoint_ = ep; })
      .def_property("nrings",
                    [](const distributed::ProcessGroupStrategy &self) {
                      return self.nrings_;
                    },
                    [](distributed::ProcessGroupStrategy &self, int nrings) {
                      self.nrings_ = nrings;
                    });
}

}  // end namespace pybind
}  // namespace paddle
