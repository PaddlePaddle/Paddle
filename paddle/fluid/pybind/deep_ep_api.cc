// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <Python.h>
#include "pybind11/functional.h"
#include "pybind11/stl.h"

#ifdef PADDLE_WITH_DEEP_EP
#include "paddle/fluid/distributed/collective/deep_ep/deep_ep.hpp"
#endif
#include "paddle/fluid/pybind/deep_ep_api.h"
#include "paddle/utils/pybind.h"

namespace py = pybind11;

namespace paddle::pybind {

void BindDeepEPApi(pybind11::module *m) {
#ifdef PADDLE_WITH_DEEP_EP
  pybind11::class_<deep_ep::Config>(*m, "Config")
      .def(pybind11::init<int, int, int, int, int>(),
           py::arg("num_sms") = 20,
           py::arg("num_max_nvl_chunked_send_tokens") = 6,
           py::arg("num_max_nvl_chunked_recv_tokens") = 256,
           py::arg("num_max_rdma_chunked_send_tokens") = 6,
           py::arg("num_max_rdma_chunked_recv_tokens") = 256)
      .def("get_nvl_buffer_size_hint",
           &deep_ep::Config::get_nvl_buffer_size_hint)
      .def("get_rdma_buffer_size_hint",
           &deep_ep::Config::get_rdma_buffer_size_hint);
  m->def("get_low_latency_rdma_size_hint",
         &deep_ep::get_low_latency_rdma_size_hint);
  m->def("get_low_latency_rdma_size_hint_two_stage",
         &deep_ep::get_low_latency_rdma_size_hint_two_stage);
  m->def("get_low_latency_nvl_size_hint_two_stage",
         &deep_ep::get_low_latency_nvl_size_hint_two_stage);

  pybind11::class_<deep_ep::EventHandle>(*m, "EventHandle")
      .def(pybind11::init<>())
      .def("current_stream_wait", &deep_ep::EventHandle::current_stream_wait)
      .def("calc_stream_wait", &deep_ep::EventHandle::CalcStreamWait)
      .def("comm_stream_wait", &deep_ep::EventHandle::CommStreamWait);

  m->def("get_event_handle_from_calc_stream",
         &deep_ep::GetEventHandleFromCalcStream);
  m->def("get_event_handle_from_comm_stream",
         &deep_ep::GetEventHandleFromCommStream);

  m->def("get_event_handle_from_custom_stream",
         &deep_ep::GetEventHandleFromCustomStream);

  pybind11::class_<deep_ep::Buffer>(*m, "Buffer")
      .def(pybind11::init<int, int, int64_t, int64_t, bool, int>())
      .def("is_available", &deep_ep::Buffer::is_available)
      .def("get_num_rdma_ranks", &deep_ep::Buffer::get_num_rdma_ranks)
      .def("get_rdma_rank", &deep_ep::Buffer::get_rdma_rank)
      .def("get_root_rdma_rank", &deep_ep::Buffer::get_root_rdma_rank)
      .def("get_local_device_id", &deep_ep::Buffer::get_local_device_id)
      .def("get_comm_stream",
           [](deep_ep::Buffer &self) {
             int device_id = self.get_local_device_id();
             cudaStream_t comm_stream = self.get_comm_stream();
             auto s = phi::Stream(reinterpret_cast<phi::StreamId>(comm_stream));
             return phi::CUDAStream(phi::GPUPlace(device_id), s);
           })
      .def("get_local_ipc_handle", &deep_ep::Buffer::get_local_ipc_handle)
      .def("get_local_nvshmem_unique_id",
           &deep_ep::Buffer::get_local_nvshmem_unique_id)
      .def("sync", &deep_ep::Buffer::sync)
      .def("get_dispatch_layout",
           [](deep_ep::Buffer &self,
              py::handle topk_idx,
              int num_experts,
              std::optional<deep_ep::EventHandle> &previous_event,
              bool async,
              bool allocate_on_comm_stream) {
             auto topk_idx_tensor = CastPyArg2Tensor(topk_idx.ptr(), 0);
             return self.get_dispatch_layout_api(topk_idx_tensor,
                                                 num_experts,
                                                 previous_event,
                                                 async,
                                                 allocate_on_comm_stream);
           })
      .def("intranode_dispatch", &deep_ep::Buffer::intranode_dispatch_api)
      .def("intranode_combine", &deep_ep::Buffer::intranode_combine_api)
      .def("internode_dispatch", &deep_ep::Buffer::internode_dispatch_api)
      .def("internode_combine", &deep_ep::Buffer::internode_combine_api)
      .def("barrier_all", &deep_ep::Buffer::barrier_all)
      .def("clean_low_latency_buffer",
           &deep_ep::Buffer::clean_low_latency_buffer)
      .def("low_latency_dispatch", &deep_ep::Buffer::low_latency_dispatch_api)
      .def("low_latency_combine", &deep_ep::Buffer::low_latency_combine_api)
      .def("low_latency_dispatch_two_stage",
           &deep_ep::Buffer::low_latency_dispatch_two_stage_api)
      .def("low_latency_combine_two_stage",
           &deep_ep::Buffer::low_latency_combine_two_stage_api);
#endif
}

}  // namespace paddle::pybind
