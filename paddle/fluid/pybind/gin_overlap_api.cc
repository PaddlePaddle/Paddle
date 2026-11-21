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

// gin_overlap_api.cc: exports GradCommEngine as the
// paddle.base.core.gin_overlap submodule (the GradCommEngine class, GROUP_* and
// the release layer constants). What the methods do, the return-code convention
// and the call order are described in the comment before GradCommEngine in
// gin_overlap/gin_engine.h.
//   - Python calls the methods by name, so method names and py::arg names must
//     not change.
//   - Methods with several outputs return a tuple whose first element is the
//     return code. Pointers are passed as Python ints (uintptr_t).
//   - Without PADDLE_WITH_GIN_OVERLAP nothing is registered, and
//     paddle.base.core has no gin_overlap attribute.
#include "paddle/fluid/pybind/gin_overlap_api.h"

#include <cstdint>
#include <string>
#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#ifdef PADDLE_WITH_GIN_OVERLAP
#include "paddle/fluid/distributed/collective/gin_overlap/gin_engine.h"
#include "paddle/fluid/distributed/collective/gin_overlap/paddle_tensor.h"
#include "paddle/utils/pybind.h"
#endif

namespace py = pybind11;

namespace paddle {
namespace pybind {

#ifdef PADDLE_WITH_GIN_OVERLAP
using gin_overlap::GradCommEngine;
#endif

void BindGinOverlapApi(py::module* m) {
#ifdef PADDLE_WITH_GIN_OVERLAP
  py::module gin = m->def_submodule(
      "gin_overlap",
      "CUDA backend of sharding-stage1 gradient communication (NCCL GIN "
      "one-sided puts)");
  // Group ids: dense buckets use the sharding group, expert buckets the
  // moe_sharding group
  gin.attr("GROUP_DENSE") = static_cast<int>(gin_overlap::GROUP_DENSE);
  gin.attr("GROUP_MOE") = static_cast<int>(gin_overlap::GROUP_MOE);
  // Release layer constants (ADVANCE_*, STAGE_*, COUNTER_*, TAIL_*,
  // ACTIVITY_STAGE_SCALE, from release::constants) become module attributes;
  // release_constants() returns the same table as a dict. Python must read
  // these values by name and never hard-code them.
  py::dict release_constants;
  {
    std::vector<std::string> names;
    std::vector<int> values;
    gin_overlap::release::constants(&names, &values);
    for (size_t i = 0; i < names.size(); ++i) {
      gin.attr(names[i].c_str()) = values[i];
      release_constants[py::str(names[i])] = values[i];
    }
  }

  py::class_<GradCommEngine> engine(gin, "GradCommEngine");
  engine
      .def(py::init<>())
      // ---- init
      .def_static("release_constants",
                  [release_constants]() { return release_constants; })
      .def_static(
          "error_string",
          [](int code) {
            return std::string(GradCommEngine::error_string(code));
          },
          py::arg("code"))
      .def_static("nccl_runtime_version", &GradCommEngine::nccl_runtime_version)
      .def_static("nccl_build_version", &GradCommEngine::nccl_build_version)
      // Takes Paddle's existing ncclComm: returns (return code, reason);
      // expert_ring_id < 0 means there are no expert buckets
      .def(
          "set_comms",
          [](GradCommEngine& self,
             int dense_ring_id,
             int expert_ring_id,
             int device_id,
             int dense_size,
             int dense_rank,
             int expert_size,
             int expert_rank) {
            std::string err;
            const int rc = self.set_comms(dense_ring_id,
                                          expert_ring_id,
                                          device_id,
                                          dense_size,
                                          dense_rank,
                                          expert_size,
                                          expert_rank,
                                          &err);
            return py::make_tuple(rc, err);
          },
          py::arg("dense_ring_id"),
          py::arg("expert_ring_id"),
          py::arg("device_id"),
          py::arg("dense_size"),
          py::arg("dense_rank"),
          py::arg("expert_size"),
          py::arg("expert_rank"))
      .def("alloc_windows",
           &GradCommEngine::alloc_windows,
           py::arg("group"),
           py::arg("send_bytes"),
           py::arg("recv_bytes"),
           py::arg("signal_bytes"))
      .def(
          "create_dev_comm",
          [](GradCommEngine& self, int group, int num_gin_contexts_hint) {
            int nctx = 0, lsa_rank = 0, lsa_size = 0;
            const int rc = self.create_dev_comm(
                group, num_gin_contexts_hint, &nctx, &lsa_rank, &lsa_size);
            return py::make_tuple(rc, nctx, lsa_rank, lsa_size);
          },
          py::arg("group"),
          py::arg("num_gin_contexts_hint"))
      .def("set_group_topology",
           &GradCommEngine::set_group_topology,
           py::arg("group"),
           py::arg("num_nodes"),
           py::arg("node_index"))
      // The only place bucket counts are passed in (side streams, per-bucket
      // events, window mark ring, offsets of the three credit areas)
      .def("configure",
           &GradCommEngine::configure,
           py::arg("num_dense_buckets"),
           py::arg("num_moe_buckets"),
           py::arg("num_window_marks"),
           py::arg("credit_intra_offset"),
           py::arg("credit_inter_offset"),
           py::arg("credit_moe_offset"))
      .def("record_window_mark", &GradCommEngine::record_window_mark)
      .def("gather_gate", &GradCommEngine::gather_gate, py::arg("mark"))
      // ---- host window, join, scale, flush
      .def(
          "alloc_host_counters",
          [](GradCommEngine& self, int group, size_t want_bytes) {
            uintptr_t cpu = 0;
            size_t nbytes = 0;
            const int rc =
                self.alloc_host_counters(group, want_bytes, &cpu, &nbytes);
            return py::make_tuple(
                rc, static_cast<uint64_t>(cpu), static_cast<uint64_t>(nbytes));
          },
          py::arg("group"),
          py::arg("want_bytes"))
      .def("join_bucket_streams",
           &GradCommEngine::join_bucket_streams,
           py::arg("kind"),
           py::arg("bucket"))
      .def("gather_wait_bucket",
           &GradCommEngine::gather_wait_bucket,
           py::arg("kind"),
           py::arg("bucket"))
      .def("scale_range",
           &GradCommEngine::scale_range,
           py::arg("group"),
           py::arg("send_offset"),
           py::arg("numel"),
           py::arg("scale"))
      .def("end_step", &GradCommEngine::end_step)
      .def("stats",
           [](GradCommEngine& self) {
             uint64_t out[3] = {0, 0, 0};
             self.stats(out);
             return py::make_tuple(out[0], out[1], out[2]);
           })
      // ---- release layer (advance on the main thread + the tail thread)
      // Bucket counts, credit area offsets and the expert group's size and
      // rank come from configure / set_comms
      .def("release_setup",
           &GradCommEngine::release_setup,
           py::arg("dense_order"),
           py::arg("moe_order"),
           py::arg("dense_table"),
           py::arg("moe_table"),
           py::arg("put_blocks"),
           py::arg("intra_arrive_offset"),
           py::arg("num_marks"))
      .def("release_step_reset",
           &GradCommEngine::release_step_reset,
           py::arg("step"))
      // Returns (return code, [kind, bucket, ...] = the shard reduces
      // released by this call, in release order)
      .def(
          "advance",
          [](GradCommEngine& self,
             int mark,
             int stages,
             int intra_node_reduce_blocks,
             int shard_reduce_blocks) {
            std::vector<int> released;
            const int rc = self.advance(mark,
                                        stages,
                                        intra_node_reduce_blocks,
                                        shard_reduce_blocks,
                                        &released);
            return py::make_tuple(rc, released);
          },
          py::arg("mark"),
          py::arg("stages"),
          py::arg("intra_node_reduce_blocks"),
          py::arg("shard_reduce_blocks"))
      .def("bucket_ready",
           &GradCommEngine::bucket_ready,
           py::arg("kind"),
           py::arg("bucket"),
           py::arg("scale"),
           py::arg("mark"))
      // Returns (return code, positions of the six stages, pending buckets of
      // the four stages, pending buckets at the moment the tail thread saw
      // everything posted, whether everything is released)
      .def("release_status",
           [](GradCommEngine& self) {
             std::vector<int> positions, counts, posted;
             int done = 0;
             const int rc =
                 self.release_status(&positions, &counts, &posted, &done);
             return py::make_tuple(rc, positions, counts, posted, done);
           })
      .def("release_stuck_report",
           [](GradCommEngine& self) {
             std::vector<int64_t> flat;
             const int rc = self.release_stuck_report(&flat);
             return py::make_tuple(rc, flat);
           })
      .def("release_error",
           [](GradCommEngine& self) {
             int code = 0, stage = 0, bucket = -1;
             const int rc = self.release_error(&code, &stage, &bucket);
             return py::make_tuple(rc, code, stage, bucket);
           })
      .def("release_activity", &GradCommEngine::release_activity)
      .def("tail_mark", &GradCommEngine::tail_mark)
      .def("tail_start",
           &GradCommEngine::tail_start,
           py::arg("intra_node_reduce_blocks"),
           py::arg("shard_reduce_blocks"),
           py::arg("mark"),
           py::arg("all_posted"))
      .def("tail_all_posted", &GradCommEngine::tail_all_posted)
      // tail_wait / tail_abort block on a condition variable for up to
      // timeout_s / wait_s; the tail thread never touches Python objects, so
      // the GIL is released while they wait.
      .def("tail_wait",
           &GradCommEngine::tail_wait,
           py::arg("timeout_s"),
           py::call_guard<py::gil_scoped_release>())
      .def("tail_abort",
           &GradCommEngine::tail_abort,
           py::arg("wait_s"),
           py::call_guard<py::gil_scoped_release>())
      .def("tail_finish", &GradCommEngine::tail_finish)
      // ---- Muon 2D-bucket gather
      .def("gather_setup", &GradCommEngine::gather_setup, py::arg("num_slots"))
      .def("gather_mark_ready",
           &GradCommEngine::gather_mark_ready,
           py::arg("slot"))
      // Arguments (slot / offsets / owner) come from the table built by
      // tail_gather_setup; only the entry index and this step's sequence
      // number are passed. Returns (return code, number of signals)
      .def(
          "gather_put_entry",
          [](GradCommEngine& self, int entry_index, uint64_t seq) {
            int n = 0;
            const int rc = self.gather_put_entry(entry_index, seq, &n);
            return py::make_tuple(rc, n);
          },
          py::arg("entry_index"),
          py::arg("seq"))
      .def("gather_join", &GradCommEngine::gather_join, py::arg("slot"))
      .def("tail_gather_setup",
           &GradCommEngine::tail_gather_setup,
           py::arg("table"))
      .def("tail_gather_set_sequences",
           &GradCommEngine::tail_gather_set_sequences,
           py::arg("seqs"))
      .def("tail_gather_get_sent", [](GradCommEngine& self) {
        std::vector<int64_t> flat;
        const int rc = self.tail_gather_get_sent(&flat);
        return py::make_tuple(rc, flat);
      });

  // paddle::Tensor of a send window slice, and the pitched copy (implemented
  // in gin_overlap/paddle_tensor.cc). They throw on error instead of
  // returning a code.
  engine.def(
      "send_window_tensor",
      [](GradCommEngine&, int group, size_t send_offset, int64_t numel) {
        return gin_overlap::send_window_tensor(group, send_offset, numel);
      },
      py::arg("group"),
      py::arg("send_offset"),
      py::arg("numel"));
  engine.def(
      "copy_2d",
      [](GradCommEngine&,
         const paddle::Tensor& dst,
         size_t dst_offset,
         size_t dst_pitch,
         const paddle::Tensor& src,
         size_t src_offset,
         size_t src_pitch,
         size_t width,
         size_t height) {
        gin_overlap::copy_2d(dst,
                             dst_offset,
                             dst_pitch,
                             src,
                             src_offset,
                             src_pitch,
                             width,
                             height);
      },
      py::arg("dst"),
      py::arg("dst_offset"),
      py::arg("dst_pitch"),
      py::arg("src"),
      py::arg("src_offset"),
      py::arg("src_pitch"),
      py::arg("width"),
      py::arg("height"));
#endif
}

}  // namespace pybind
}  // namespace paddle
