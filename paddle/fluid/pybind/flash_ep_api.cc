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
#include "paddle/fluid/distributed/collective/flash_ep/flash_ep.hpp"
#endif
#include "paddle/fluid/pybind/flash_ep_api.h"
#include "paddle/utils/pybind.h"

namespace py = pybind11;

namespace paddle::pybind {

void BindFlashEPApi(pybind11::module *m) {
#ifdef PADDLE_WITH_DEEP_EP
  pybind11::class_<flash_ep::Config>(*m, "FEConfig")
      .def(pybind11::init<int, int, int, int, int>(),
           py::arg("num_sms") = 20,
           py::arg("num_max_nvl_chunked_send_tokens") = 6,
           py::arg("num_max_nvl_chunked_recv_tokens") = 256,
           py::arg("num_max_rdma_chunked_send_tokens") = 6,
           py::arg("num_max_rdma_chunked_recv_tokens") = 256)
      .def("get_nvl_buffer_size_hint",
           &flash_ep::Config::get_nvl_buffer_size_hint)
      .def("get_rdma_buffer_size_hint",
           &flash_ep::Config::get_rdma_buffer_size_hint);

  pybind11::class_<flash_ep::EventHandle>(*m, "_EventHandle")
      .def(pybind11::init<>())
      .def("current_stream_wait", &flash_ep::EventHandle::current_stream_wait)
      .def("calc_stream_wait", &flash_ep::EventHandle::CalcStreamWait)
      .def("comm_stream_wait", &flash_ep::EventHandle::CommStreamWait);

  m->def("fe_get_event_handle_from_calc_stream",
         &flash_ep::GetEventHandleFromCalcStream);
  m->def("fe_get_event_handle_from_comm_stream",
         &flash_ep::GetEventHandleFromCommStream);
  m->def("fe_get_event_handle_from_custom_stream",
         &flash_ep::GetEventHandleFromCustomStream);

  m->def("get_flash_ep_coalesce_rdma_layout",
         &flash_ep::get_flash_ep_coalesce_rdma_layout_api);
  m->def("get_flash_ep_coalesce_rdma_schedule",
         &flash_ep::get_flash_ep_coalesce_rdma_schedule_api);
  m->def("get_flashep_rowmap", &flash_ep::get_flashep_rowmap_api);
  m->def("local_dispatch_forward", &flash_ep::local_dispatch_forward_api);
  m->def("local_dispatch_backward", &flash_ep::local_dispatch_backward_api);
  m->def("local_combine_forward", &flash_ep::local_combine_forward_api);
  m->def("local_combine_backward", &flash_ep::local_combine_backward_api);

  pybind11::class_<flash_ep::Buffer>(*m, "FEBuffer")
      .def(pybind11::init<int, int, int, int64_t, int64_t, bool, int>())
      .def("is_available", &flash_ep::Buffer::is_available)
      .def("get_num_rdma_ranks", &flash_ep::Buffer::get_num_rdma_ranks)
      .def("get_rdma_rank", &flash_ep::Buffer::get_rdma_rank)
      .def("get_root_rdma_rank", &flash_ep::Buffer::get_root_rdma_rank)
      .def("get_local_device_id", &flash_ep::Buffer::get_local_device_id)
      .def("get_comm_stream",
           [](flash_ep::Buffer &self) {
             int device_id = self.get_local_device_id();
             cudaStream_t comm_stream = self.get_comm_stream();
             auto s = phi::Stream(reinterpret_cast<phi::StreamId>(comm_stream));
             return phi::CUDAStream(phi::GPUPlace(device_id), s);
           })
      .def("get_local_ipc_handle", &flash_ep::Buffer::get_local_ipc_handle)
      .def("get_local_nvshmem_unique_id",
           &flash_ep::Buffer::get_local_nvshmem_unique_id)
      .def("sync", &flash_ep::Buffer::sync)
      .def("internode_dispatch", &flash_ep::Buffer::internode_dispatch_api)
      .def("internode_fused_notify",
           &flash_ep::Buffer::internode_fused_notify_api)
      .def("internode_combine", &flash_ep::Buffer::internode_combine_api);
#endif
}

}  // namespace paddle::pybind
