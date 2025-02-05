/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/pybind/io.h"

#include "paddle/fluid/framework/io/save_load_tensor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/selected_rows_utils.h"
#include "paddle/fluid/pir/serialize_deserialize/include/interface.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/pybind_variant_caster.h"
#include "paddle/utils/pybind.h"

namespace py = pybind11;
namespace paddle::pybind {
template <typename PlaceType>
void LoadCombine(const std::string &file_path,
                 const std::vector<std::string> &names,
                 std::vector<phi::DenseTensor *> *out,
                 bool load_as_fp16,
                 const PlaceType place) {
  pir::LoadCombineFunction(file_path, names, out, load_as_fp16, place);
}

template <typename PlaceType>
void Load(const std::string &file_path,
          int64_t seek,
          const std::vector<int64_t> &shape,
          bool load_as_fp16,
          phi::DenseTensor *out,
          const PlaceType place) {
  pir::LoadFunction(file_path, seek, shape, load_as_fp16, out, place);
}
void BindIO(pybind11::module *m) {
  m->def("save_dense_tensor",
         [](const phi::DenseTensor &tensor, const std::string &str_file_name) {
           std::ofstream fout(str_file_name, std::ios::binary);
           PADDLE_ENFORCE_EQ(
               static_cast<bool>(fout),
               true,
               common::errors::Unavailable("Cannot open %s to save variables.",
                                           str_file_name));
           phi::SerializeToStream(fout, tensor);

           int64_t tellp = fout.tellp();
           fout.close();
           return tellp;
         });

  m->def("load_dense_tensor",
         [](phi::DenseTensor &tensor, const std::string &str_file_name) {
           std::ifstream fin(str_file_name, std::ios::binary);
           PADDLE_ENFORCE_EQ(
               static_cast<bool>(fin),
               true,
               common::errors::Unavailable("Cannot open %s to load variables.",
                                           str_file_name));

           phi::DeserializeFromStream(fin, &tensor);
           int64_t tellg = fin.tellg();
           fin.close();
           return tellg;
         });

  m->def("save_selected_rows",
         [](const phi::SelectedRows &selected_rows,
            const std::string &str_file_name) {
           std::ofstream fout(str_file_name, std::ios::binary);
           PADDLE_ENFORCE_EQ(
               static_cast<bool>(fout),
               true,
               common::errors::Unavailable(
                   "Cannot open %s to save SelectedRows.", str_file_name));

           phi::SerializeToStream(fout, selected_rows);
           int64_t tellp = fout.tellp();
           fout.close();
           return tellp;
         });

  m->def(
      "load_selected_rows",
      [](phi::SelectedRows &selected_rows, const std::string &str_file_name) {
        std::ifstream fin(str_file_name, std::ios::binary);
        PADDLE_ENFORCE_EQ(
            static_cast<bool>(fin),
            true,
            common::errors::Unavailable("Cannot open %s to load SelectedRows.",
                                        str_file_name));

        phi::DeserializeFromStream(fin, &selected_rows);
        int64_t tellg = fin.tellg();
        fin.close();
        return tellg;
      });

  m->def("save_dense_tensor_to_memory",
         [](const phi::DenseTensor &tensor) -> py::bytes {
           std::ostringstream ss;
           phi::SerializeToStream(ss, tensor);
           return ss.str();
         });

  m->def("load_dense_tensor_from_memory",
         [](phi::DenseTensor &tensor, const std::string &tensor_bytes) {
           std::istringstream fin(tensor_bytes,
                                  std::ios::in | std::ios::binary);
           phi::DeserializeFromStream(fin, &tensor);
         });

  m->def("save_selected_rows_to_memory",
         [](const phi::SelectedRows &selected_rows) -> py::bytes {
           std::ostringstream ss;
           phi::SerializeToStream(ss, selected_rows);
           return ss.str();
         });

  m->def("load_selected_rows_from_memory",
         [](phi::SelectedRows &selected_rows,
            const std::string &selected_rows_bytes) {
           std::istringstream fin(selected_rows_bytes,
                                  std::ios::in | std::ios::binary);
           phi::DeserializeFromStream(fin, &selected_rows);
         });

  m->def("load_dense_tensor", [](const std::string path) {
    phi::DenseTensor tensor_load;
    paddle::framework::LoadTensor(path, &tensor_load);
    return tensor_load;
  });

  m->def("save_func", &pir::SaveFunction);

  m->def("save_combine_func", &pir::SaveCombineFunction);

  m->def("load_func", &Load<phi::CPUPlace>);
  m->def("load_func", &Load<phi::CustomPlace>);
  m->def("load_func", &Load<phi::XPUPlace>);
  m->def("load_func", &Load<phi::GPUPinnedPlace>);
  m->def("load_func", &Load<phi::GPUPlace>);
  m->def("load_func", &Load<phi::IPUPlace>);
  m->def("load_func", &Load<phi::Place>);
  m->def("load_combine_func", &LoadCombine<phi::CPUPlace>);
  m->def("load_combine_func", &LoadCombine<phi::CustomPlace>);
  m->def("load_combine_func", &LoadCombine<phi::XPUPlace>);
  m->def("load_combine_func", &LoadCombine<phi::GPUPinnedPlace>);
  m->def("load_combine_func", &LoadCombine<phi::GPUPlace>);
  m->def("load_combine_func", &LoadCombine<phi::IPUPlace>);
  m->def("load_combine_func", &LoadCombine<phi::Place>);

  m->def("serialize_pir_program",
         &pir::WriteModule,
         py::arg("program"),
         py::arg("file_path"),
         py::arg("pir_version"),
         py::arg("overwrite") = true,
         py::arg("readable") = false,
         py::arg("trainable") = true);
  m->def("deserialize_pir_program",
         &pir::ReadModule,
         py::arg("file_path"),
         py::arg("program"),
         py::arg("pir_version") = -1);
}
}  // namespace paddle::pybind
