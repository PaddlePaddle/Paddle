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

#include <Python.h>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "paddle/fluid/pybind/symmetric_memory_api.h"
#include "paddle/phi/core/distributed/symmetric_memory/symmetric_memory.h"
#include "paddle/phi/core/distributed/store/store.h"
#include "paddle/phi/core/dense_tensor.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {

void BindSymmetricMemory(py::module* m) {
  using phi::distributed::SymmetricMemory;
  using phi::distributed::SymmetricMemoryAllocator;

  // Bind SymmetricMemory class
  py::class_<SymmetricMemory, std::shared_ptr<SymmetricMemory>>(
      *m, "_SymmetricMemory")
      .def_property_readonly("rank", &SymmetricMemory::rank)
      .def_property_readonly("world_size", &SymmetricMemory::world_size)
      .def_property_readonly("buffer_size", &SymmetricMemory::buffer_size)
      .def_property_readonly("signal_pad_size", &SymmetricMemory::signal_pad_size)
      .def_property_readonly("device_id", &SymmetricMemory::device_id)
      .def_property_readonly("buffer_ptrs",
          [](const SymmetricMemory& self) {
            std::vector<int64_t> ptrs;
            for (auto* p : self.buffer_ptrs()) {
              ptrs.push_back(reinterpret_cast<int64_t>(p));
            }
            return ptrs;
          })
      .def_property_readonly("signal_pad_ptrs",
          [](const SymmetricMemory& self) {
            std::vector<int64_t> ptrs;
            for (auto* p : self.signal_pad_ptrs()) {
              ptrs.push_back(reinterpret_cast<int64_t>(p));
            }
            return ptrs;
          })
      .def("get_buffer", &SymmetricMemory::get_buffer,
           py::arg("rank"),
           py::arg("sizes"),
           py::arg("dtype"),
           py::arg("storage_offset") = 0)
      .def("get_signal_pad", &SymmetricMemory::get_signal_pad,
           py::arg("rank"),
           py::arg("sizes") = std::vector<int64_t>{},
           py::arg("dtype") = phi::DataType::UINT32,
           py::arg("storage_offset") = 0)
      .def("barrier", &SymmetricMemory::barrier,
           py::arg("channel") = 0,
           py::arg("timeout_ms") = 0)
      .def("put_signal", &SymmetricMemory::put_signal,
           py::arg("dst_rank"),
           py::arg("channel") = 0,
           py::arg("timeout_ms") = 0)
      .def("wait_signal", &SymmetricMemory::wait_signal,
           py::arg("src_rank"),
           py::arg("channel") = 0,
           py::arg("timeout_ms") = 0);

  // Bind SymmetricMemoryAllocator static methods
  py::class_<SymmetricMemoryAllocator>(
      *m, "_SymmetricMemoryAllocator")
      .def_static("instance", &SymmetricMemoryAllocator::Instance,
                  py::return_value_policy::reference)
      .def("alloc", &SymmetricMemoryAllocator::alloc,
           py::arg("size"),
           py::arg("device_id"),
           py::arg("group_name"))
      .def("alloc_persistent", &SymmetricMemoryAllocator::alloc_persistent,
           py::arg("size"),
           py::arg("device_id"),
           py::arg("group_name"),
           py::arg("alloc_id"))
      .def("rendezvous", &SymmetricMemoryAllocator::rendezvous,
           py::arg("tensor"))
      .def("is_symm_mem_tensor", &SymmetricMemoryAllocator::is_symm_mem_tensor,
           py::arg("tensor"))
      .def("set_group_info",
           [](SymmetricMemoryAllocator& self,
              const std::string& group_name,
              int rank,
              int world_size,
              std::shared_ptr<phi::distributed::Store> store) {
             self.set_group_info(group_name, rank, world_size, store);
           },
           py::arg("group_name"),
           py::arg("rank"),
           py::arg("world_size"),
           py::arg("store"))
      .def_static("get_signal_pad_size",
                  &SymmetricMemoryAllocator::get_signal_pad_size)
      .def_static("set_signal_pad_size",
                  &SymmetricMemoryAllocator::set_signal_pad_size,
                  py::arg("size"))
      .def_static("stream_write_value32",
                  &SymmetricMemoryAllocator::stream_write_value32,
                  py::arg("tensor"),
                  py::arg("offset"),
                  py::arg("val"))
      .def_static("memset32",
                  &SymmetricMemoryAllocator::memset32,
                  py::arg("tensor"),
                  py::arg("offset"),
                  py::arg("val"),
                  py::arg("count"));
}

}  // namespace pybind
}  // namespace paddle
