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

// gin_overlap_api.cc —— 把 GradCommEngine 导出成 paddle.base.core.gin_overlap
// 子模块（GradCommEngine 类、GROUP_* 和放行层常量）。方法的含义、返回码约定和
// 调用顺序见 gin_overlap/gin_engine.h 里 GradCommEngine 前的注释。
//   - Python 侧按名字调用，方法名和 py::arg 名字不能改。
//   - 有多个输出的方法返回 tuple，第一个元素是返回码。指针用 Python int
//     （uintptr_t）传。
//   - 没定义 PADDLE_WITH_GIN_OVERLAP 时不注册任何东西，paddle.base.core 上没有
//     gin_overlap 属性。
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
      "gin_overlap", "sharding stage1 梯度通信（NCCL GIN 单边）的 CUDA 后端");
  // 通信组编号：dense 桶走 sharding 组，专家桶走 moe_sharding 组
  gin.attr("GROUP_DENSE") = static_cast<int>(gin_overlap::GROUP_DENSE);
  gin.attr("GROUP_MOE") = static_cast<int>(gin_overlap::GROUP_MOE);
  // 放行层常量（ADVANCE_*、STAGE_*、COUNTER_*、TAIL_*、ACTIVITY_STAGE_SCALE，
  // 来自 release::constants）导出成模块属性；release_constants() 以 dict
  // 返回同一张表。Python 侧必须按名字取这些数值，不能自己写数值。
  py::dict release_constants;
  {
    std::vector<std::string> names;
    std::vector<int> values;
    gin_overlap::release::constants(names, values);
    for (size_t i = 0; i < names.size(); ++i) {
      gin.attr(names[i].c_str()) = values[i];
      release_constants[py::str(names[i])] = values[i];
    }
  }

  py::class_<GradCommEngine> engine(gin, "GradCommEngine");
  engine
      .def(py::init<>())
      // ---- 初始化
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
      // 取 Paddle 已有的 ncclComm：返回 (返回码, 出错原因)；expert_ring_id < 0
      // = 没有专家桶
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
                                          err);
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
      // 桶数只在这里传入（侧流、每桶 event、窗口 mark 环、三块 credit 区偏移）
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
      // ---- host window、join、scale、flush
      .def(
          "alloc_host_counters",
          [](GradCommEngine& self, int group, size_t want_bytes) {
            uintptr_t cpu = 0;
            size_t nbytes = 0;
            const int rc =
                self.alloc_host_counters(group, want_bytes, &cpu, &nbytes);
            return py::make_tuple(rc, (uint64_t)cpu, (uint64_t)nbytes);
          },
          py::arg("group"),
          py::arg("want_bytes"))
      .def("join_bucket_streams",
           &GradCommEngine::join_bucket_streams,
           py::arg("is_dense"),
           py::arg("bucket"))
      .def("gather_wait_bucket",
           &GradCommEngine::gather_wait_bucket,
           py::arg("is_dense"),
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
             unsigned long long out[3] = {0, 0, 0};  // NOLINT
             self.stats(out);
             return py::make_tuple(out[0], out[1], out[2]);
           })
      // ---- 放行层（主线程的 advance + 尾部线程）
      // 桶数、credit 区偏移、专家组大小和序号来自 configure / set_comms
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
      // 返回 (返回码, [kind, 桶号, ...] = 本次放了哪些分片求和，按放行顺序)
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
                                        released);
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
      // 返回 (返回码, 六个阶段的位置, 四个阶段的待放行桶数,
      // 尾部线程看到全部 post 那一刻的待放行桶数, 是否全部放完)
      .def("release_status",
           [](GradCommEngine& self) {
             std::vector<int> positions, counts, posted;
             int done = 0;
             const int rc =
                 self.release_status(positions, counts, posted, &done);
             return py::make_tuple(rc, positions, counts, posted, done);
           })
      .def("release_stuck_report",
           [](GradCommEngine& self) {
             std::vector<long long> flat;  // NOLINT
             const int rc = self.release_stuck_report(flat);
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
      .def("tail_wait", &GradCommEngine::tail_wait, py::arg("timeout_s"))
      .def("tail_abort", &GradCommEngine::tail_abort, py::arg("wait_s"))
      .def("tail_finish", &GradCommEngine::tail_finish)
      // ---- Muon 2D 桶 gather
      .def("gather_setup", &GradCommEngine::gather_setup, py::arg("num_slots"))
      .def("gather_mark_ready",
           &GradCommEngine::gather_mark_ready,
           py::arg("slot"))
      // 参数（slot / 偏移 / owner）从 tail_gather_setup 建的表里查，这里只给
      // 表项下标和本步序号；返回 (返回码, signal 笔数)
      .def(
          "gather_put_entry",
          [](GradCommEngine& self,
             int entry_index,
             unsigned long long seq) {  // NOLINT
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
        std::vector<long long> flat;  // NOLINT
        const int rc = self.tail_gather_get_sent(flat);
        return py::make_tuple(rc, flat);
      });

  // send window 切片的 paddle::Tensor 和 pitched 拷贝（实现在
  // gin_overlap/paddle_tensor.cc）。出错时抛异常，不返回错误码。
  engine.def(
      "send_window_tensor",
      [](GradCommEngine&,
         int group,
         size_t send_offset,
         long long numel) {  // NOLINT
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
