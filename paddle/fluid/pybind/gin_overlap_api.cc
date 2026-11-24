// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pybind/gin_overlap_api.h"

#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "paddle/fluid/distributed/collective/gin_overlap/grad_comm_scheduler.h"
#include "paddle/fluid/distributed/collective/process_group_nccl.h"
#include "pybind11/stl.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {

namespace {

using distributed::GinBucketDesc;
using distributed::GinGatherDesc;
using distributed::GinGradCommScheduler;
using distributed::GinGradCommSchedulerConfig;
using distributed::GinGroupWindows;
using distributed::ProcessGroupNCCL;

// (send_window, recv_window, host_window) handles as ints.
using WindowsTuple = std::tuple<uint64_t, uint64_t, uint64_t>;
// (kind, send_offset, signal_offset, shard_numel, arrive_offset,
//  recv_offset_0, recv_offset_1, depth_0, depth_1)
using BucketTuple = std::tuple<int,
                               int64_t,
                               int64_t,
                               int64_t,
                               int64_t,
                               int64_t,
                               int64_t,
                               int64_t,
                               int64_t>;
// (bucket, owner, arrive_offset, intra_arrive_offset)
using GatherTuple = std::tuple<int, int, int64_t, int64_t>;

GinGroupWindows ToWindows(const WindowsTuple& windows) {
  GinGroupWindows result;
  result.send_window = reinterpret_cast<void*>(std::get<0>(windows));
  result.recv_window = reinterpret_cast<void*>(std::get<1>(windows));
  result.host_window = reinterpret_cast<void*>(std::get<2>(windows));
  return result;
}

std::unique_ptr<GinGradCommScheduler> CreateScheduler(
    ProcessGroupNCCL* dense_group,
    ProcessGroupNCCL* expert_group,
    const WindowsTuple& dense_windows,
    const WindowsTuple& expert_windows,
    const std::vector<BucketTuple>& buckets,
    const std::vector<int>& dense_order,
    const std::vector<int>& expert_order,
    int64_t credit_intra_offset,
    int64_t credit_inter_offset,
    int64_t credit_expert_offset,
    const std::vector<GatherTuple>& gather,
    int num_window_marks,
    int num_tail_marks) {
  GinGradCommSchedulerConfig config;
  config.dense_group = dense_group;
  config.expert_group = expert_group;
  config.dense_windows = ToWindows(dense_windows);
  config.expert_windows = ToWindows(expert_windows);
  config.buckets.reserve(buckets.size());
  for (const BucketTuple& bucket_tuple : buckets) {
    GinBucketDesc bucket_desc;
    bucket_desc.kind = std::get<0>(bucket_tuple);
    bucket_desc.send_offset = std::get<1>(bucket_tuple);
    bucket_desc.signal_offset = std::get<2>(bucket_tuple);
    bucket_desc.shard_numel = std::get<3>(bucket_tuple);
    bucket_desc.arrive_offset = std::get<4>(bucket_tuple);
    bucket_desc.recv_offset[0] = std::get<5>(bucket_tuple);
    bucket_desc.recv_offset[1] = std::get<6>(bucket_tuple);
    bucket_desc.depth[0] = std::get<7>(bucket_tuple);
    bucket_desc.depth[1] = std::get<8>(bucket_tuple);
    config.buckets.push_back(bucket_desc);
  }
  config.dense_order = dense_order;
  config.expert_order = expert_order;
  config.credit_intra_offset = credit_intra_offset;
  config.credit_inter_offset = credit_inter_offset;
  config.credit_expert_offset = credit_expert_offset;
  config.gather.reserve(gather.size());
  for (const GatherTuple& gather_tuple : gather) {
    GinGatherDesc gather_desc;
    gather_desc.bucket = std::get<0>(gather_tuple);
    gather_desc.owner = std::get<1>(gather_tuple);
    gather_desc.arrive_offset = std::get<2>(gather_tuple);
    gather_desc.intra_arrive_offset = std::get<3>(gather_tuple);
    config.gather.push_back(gather_desc);
  }
  config.num_window_marks = num_window_marks;
  config.num_tail_marks = num_tail_marks;
  py::gil_scoped_release release;
  return std::make_unique<GinGradCommScheduler>(config);
}

}  // namespace

void BindGinOverlapApi(py::module* m) {
  py::module gin = m->def_submodule(
      "gin_overlap",
      "Staged sharding-stage1 gradient reduce_scatter on NCCL GIN "
      "one-sided puts.");
  gin.attr("INTRA_COPY") = static_cast<int>(distributed::kGinOpIntraCopy);
  gin.attr("INTRA_SUM") = static_cast<int>(distributed::kGinOpIntraSum);
  gin.attr("PUT") = static_cast<int>(distributed::kGinOpPut);
  gin.attr("FINAL_SUM") = static_cast<int>(distributed::kGinOpFinalSum);
  gin.attr("GATHER") = static_cast<int>(distributed::kGinOpGather);
  gin.attr("DENSE_BUCKET") = static_cast<int>(distributed::kGinDenseBucket);
  gin.attr("EXPERT_BUCKET") = static_cast<int>(distributed::kGinExpertBucket);

  py::class_<GinGradCommScheduler>(gin, "GinGradCommScheduler", R"DOC(
      Releases the stages of the gradient buckets of a sharding-stage1
      optimizer. See gin_overlap/grad_comm_scheduler.h.
      )DOC")
      .def(py::init(&CreateScheduler),
           py::arg("dense_group"),
           py::arg("expert_group"),
           py::arg("dense_windows"),
           py::arg("expert_windows"),
           py::arg("buckets"),
           py::arg("dense_order"),
           py::arg("expert_order"),
           py::arg("credit_intra_offset"),
           py::arg("credit_inter_offset"),
           py::arg("credit_expert_offset"),
           py::arg("gather"),
           py::arg("num_window_marks"),
           py::arg("num_tail_marks"),
           py::keep_alive<1, 2>(),
           py::keep_alive<1, 3>(),
           R"DOC(
           Builds the scheduler from plain data. dense_group / expert_group
           are ProcessGroupNCCL objects (expert_group may be None without
           expert buckets) whose device communicators exist; *_windows are
           (send, recv, host) window handles of each group; buckets holds one
           (kind, send_offset, signal_offset, shard_numel, arrive_offset,
           recv_offset_0, recv_offset_1, depth_0, depth_1) per bucket id;
           gather holds one (bucket, owner, arrive_offset,
           intra_arrive_offset) per Muon 2D bucket.
           )DOC")
      .def("bucket_ready",
           &GinGradCommScheduler::BucketReady,
           py::arg("bucket"),
           py::arg("scale"))
      .def("release",
           &GinGradCommScheduler::Release,
           py::arg("ops"),
           py::arg("intra_sum_blocks"),
           py::arg("final_sum_blocks"),
           py::call_guard<py::gil_scoped_release>())
      .def("release_async",
           &GinGradCommScheduler::ReleaseAsync,
           py::arg("intra_sum_blocks"),
           py::arg("final_sum_blocks"),
           py::call_guard<py::gil_scoped_release>())
      .def("finish_step",
           &GinGradCommScheduler::FinishStep,
           py::arg("timeout_s"),
           py::call_guard<py::gil_scoped_release>())
      .def("completed_steps", &GinGradCommScheduler::CompletedSteps)
      .def("has_expert_buckets", &GinGradCommScheduler::HasExpertBuckets)
      .def("status",
           [](const GinGradCommScheduler& self) {
             const distributed::GinGradCommStatus status = self.Status();
             py::dict result;
             result["state"] = status.state;
             result["completed_steps"] = status.completed_steps;
             result["reported"] = status.reported;
             result["positions"] = status.positions;
             result["pending"] = status.pending;
             return result;
           })
      .def("stuck_report", &GinGradCommScheduler::StuckReport);
}

}  // namespace pybind
}  // namespace paddle
