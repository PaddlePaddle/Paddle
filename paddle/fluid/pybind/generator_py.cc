/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/generator.h"

#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif

#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "paddle/fluid/pybind/generator_py.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {
void BindGenerator(py::module* m_ptr) {
  auto& m = *m_ptr;
  py::class_<phi::Generator::GeneratorState,
             std::shared_ptr<phi::Generator::GeneratorState>>(m,
                                                              "GeneratorState")
      .def("current_seed",
           [](std::shared_ptr<phi::Generator::GeneratorState>& self) {
             return self->current_seed;
           })
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      // NOTE(shenliang03): Due to the inability to serialize mt19937_64
      // type, resulting in a problem with precision under the cpu.
      .def(py::pickle(
          [](const phi::Generator::GeneratorState& s) {  // __getstate__
            return py::make_tuple(s.device, s.current_seed, s.thread_offset);
          },
          [](py::tuple s) {  // __setstate__
            if (s.size() != 3)
              throw std::runtime_error(
                  "Invalid Random state. Please check the format(device, "
                  "current_seed, thread_offset).");

            phi::Generator::GeneratorState state;
            state.device = s[0].cast<std::int64_t>();
            state.current_seed = s[1].cast<std::uint64_t>();
            state.thread_offset = s[2].cast<std::uint64_t>();

            std::seed_seq seq({state.current_seed});
            auto engine = std::make_shared<std::mt19937_64>(seq);
            state.cpu_engine = *engine;
            return state;
          }))
#endif
      .def("__str__", [](const phi::Generator::GeneratorState& self) {
        std::stringstream ostr;
        ostr << self.device << " " << self.current_seed << " "
             << self.thread_offset << " " << self.cpu_engine;
        return ostr.str();
      });

  py::class_<std::mt19937_64>(m, "mt19937_64", "");
  py::class_<phi::Generator, std::shared_ptr<phi::Generator>>(m, "Generator")
      .def("__init__",
           [](phi::Generator& self) { new (&self) phi::Generator(); })
      .def("get_state", &phi::Generator::GetState)
      .def("set_state", &phi::Generator::SetState)
      .def("manual_seed",
           [](std::shared_ptr<phi::Generator>& self, uint64_t seed) {
             self->SetCurrentSeed(seed);
             return self;
           })
      .def("seed", &phi::Generator::Seed)
      .def("initial_seed", &phi::Generator::GetCurrentSeed)
      .def("random", &phi::Generator::Random64);
  m.def("default_cpu_generator", &phi::DefaultCPUGenerator);
  m.def("default_cuda_generator", &phi::DefaultCUDAGenerator);
  m.def("default_xpu_generator", &phi::DefaultXPUGenerator);
  m.def("set_random_seed_generator", &phi::SetRandomSeedGenerator);
  m.def("get_random_seed_generator", &phi::GetRandomSeedGenerator);
}
}  // namespace pybind
}  // namespace paddle
