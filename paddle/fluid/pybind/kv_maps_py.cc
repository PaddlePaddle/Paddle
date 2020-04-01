/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <memory>
#include <string>
#include "glog/logging.h"
#include "paddle/fluid/framework/fleet/kv_maps.h"
#include "paddle/fluid/pybind/kv_maps_py.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {
void BindKvMaps(py::module* m) {
  py::class_<framework::KV_MAPS, std::shared_ptr<framework::KV_MAPS>>(*m,
                                                                      "KV_MAPS")
      .def(py::init([](const std::string& filename) {
        VLOG(0) << "using KvMaps, init from " << filename;
        framework::KV_MAPS::InitInstance(filename);
        return framework::KV_MAPS::GetInstance();
      }));
}
}  // namespace pybind
}  // namespace paddle
