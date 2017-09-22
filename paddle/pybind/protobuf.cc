/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/pybind/protobuf.h"

namespace paddle {
namespace pybind {

void BindProgramDesc(py::module &m) {
  using namespace paddle::framework;  // NOLINT
  py::class_<ProgramDesc>(m, "ProgramDesc", "")
      .def_static("instance",
                  [] { return &GetProgramDesc(); },
                  py::return_value_policy::reference)
      .def_static("__create_program_desc__",
                  [] {
                    // Only used for unit-test
                    auto *prog_desc = new ProgramDesc;
                    auto *block = prog_desc->mutable_blocks()->Add();
                    block->set_idx(0);
                    block->set_parent_idx(-1);
                    return prog_desc;
                  })
      .def("append_block",
           [](ProgramDesc &self, BlockDesc &parent) {
             auto desc = self.add_blocks();
             desc->set_idx(self.mutable_blocks()->size() - 1);
             desc->set_parent_idx(parent.idx());
             return desc;
           },
           py::return_value_policy::reference)
      .def("root_block",
           [](ProgramDesc &self) { return self.mutable_blocks()->Mutable(0); },
           py::return_value_policy::reference)
      .def("block",
           [](ProgramDesc &self, int id) { return self.blocks(id); },
           py::return_value_policy::reference)
      .def("__str__", [](ProgramDesc &self) { return self.DebugString(); });
}

void BindBlockDesc(py::module &m) {
  using namespace paddle::framework;  // NOLINT
  py::class_<BlockDesc>(m, "BlockDesc", "")
      .def("id", [](BlockDesc &self) { return self.idx(); })
      .def("parent", [](BlockDesc &self) { return self.parent_idx(); })
      .def("append_op",
           [](BlockDesc &self) { return self.add_ops(); },
           py::return_value_policy::reference)
      .def("new_var",
           [](BlockDesc &self) { return self.add_vars(); },
           py::return_value_policy::reference);
}

void BindVarDsec(py::module &m) {
  using namespace paddle::framework;  // NOLINT
  py::class_<VarDesc>(m, "VarDesc", "")
      .def(py::init<>())
      .def("set_name",
           [](VarDesc &self, const std::string &name) { self.set_name(name); })
      .def("set_shape",
           [](VarDesc &self, const std::vector<int64_t> &dims) {
             LoDTensorDesc *lod_tensor_desc = self.mutable_lod_tensor();
             for (const int64_t &i : dims) {
               lod_tensor_desc->add_dims(i);
             }
           })
      .def("set_data_type",
           [](VarDesc &self, int type_id) {
             LoDTensorDesc *lod_tensor_desc = self.mutable_lod_tensor();
             lod_tensor_desc->set_data_type(static_cast<DataType>(type_id));
           })
      .def("shape", [](VarDesc &self) {
        const LoDTensorDesc &lod_tensor_desc = self.lod_tensor();
        int rank = lod_tensor_desc.dims_size();
        std::vector<int64_t> res(rank);
        for (int i = 0; i < rank; ++i) {
          res[i] = lod_tensor_desc.dims(i);
        }
        return res;
      });
}

void BindOpDesc(py::module &m) {
  using namespace paddle::framework;  // NOLINT
  auto op_desc_set_var = [](OpDesc::Var *var,
                            const std::string &parameter,
                            const std::vector<std::string> &arguments) {
    var->set_parameter(parameter);
    VectorToRepeated(arguments, var->mutable_arguments());
  };

  auto op_desc_set_attr = [](OpDesc &desc, const std::string &name) {
    auto attr = desc.add_attrs();
    attr->set_name(name);
    return attr;
  };

  py::class_<OpDesc>(m, "OpDesc", "")
      .def("type", [](OpDesc &op) { return op.type(); })
      .def("set_input",
           [op_desc_set_var](OpDesc &self,
                             const std::string &parameter,
                             const std::vector<std::string> &arguments) {
             auto ipt = self.add_inputs();
             op_desc_set_var(ipt, parameter, arguments);
           })
      .def("input_names",
           [](OpDesc &self) {
             std::vector<std::string> ret_val;
             ret_val.reserve(static_cast<size_t>(self.inputs().size()));
             std::transform(
                 self.inputs().begin(),
                 self.inputs().end(),
                 std::back_inserter(ret_val),
                 [](const OpDesc::Var &var) { return var.parameter(); });
             return ret_val;
           })
      .def("__str__", [](OpDesc &self) { return self.DebugString(); })
      .def("set_output",
           [op_desc_set_var](OpDesc &self,
                             const std::string &parameter,
                             const std::vector<std::string> &arguments) {
             auto opt = self.add_outputs();
             op_desc_set_var(opt, parameter, arguments);
           })
      .def("set_attr",
           [op_desc_set_attr](OpDesc &self, const std::string &name, int i) {
             op_desc_set_attr(self, name)->set_i(i);
           });
}
}  // namespace pybind
}  // namespace paddle
