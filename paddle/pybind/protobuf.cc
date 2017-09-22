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
#include <deque>

namespace paddle {
namespace pybind {

using namespace paddle::framework;  // NOLINT

template <typename T>
inline std::vector<T> RepeatedToVector(
    const google::protobuf::RepeatedField<T> &repeated_field) {
  std::vector<T> ret;
  ret.reserve(repeated_field.size());
  std::copy(
      repeated_field.begin(), repeated_field.end(), std::back_inserter(ret));
  return ret;
}

template <typename T, typename RepeatedField>
inline void VectorToRepeated(const std::vector<T> &vec,
                             RepeatedField *repeated_field) {
  repeated_field->Reserve(vec.size());
  for (auto &elem : vec) {
    *repeated_field->Add() = elem;
  }
}

class ProgramDescBind;
class OpDescBind;
class BlockDescBind;

class OpDescBind {
public:
  explicit OpDescBind(BlockDescBind *block) : block_(block) {}

  operator OpDesc *() { return &op_desc_; }

private:
  BlockDescBind *block_;
  OpDesc op_desc_;
};

class BlockDescBind {
public:
  BlockDescBind(ProgramDescBind *prog, BlockDesc *desc)
      : prog_(prog), desc_(desc), need_update_(false) {}

  ~BlockDescBind() {
    std::cerr << "dtor " << this << "," << desc_ << std::endl;
  }

  int32_t id() const {
    std::cerr << "desc ptr " << desc_ << std::endl;
    return desc_->idx();
  }

  int32_t Parent() const { return desc_->parent_idx(); }

  OpDescBind *AppendOp() {
    need_update_ = true;
    ops_.emplace_back(this);
    return &ops_.back();
  }

  void Sync() {
    if (need_update_) {
      auto &op_field = *this->desc_->mutable_ops();
      op_field.Clear();
      op_field.Reserve(static_cast<int>(ops_.size()));
      for (auto &op_desc : ops_) {
        op_field.AddAllocated(op_desc);
      }
    }
  }

private:
  ProgramDescBind *prog_;  // not_own
  BlockDesc *desc_;        // not_own
  bool need_update_;

  std::deque<OpDescBind> ops_;
};

using ProgDescMap =
    std::unordered_map<ProgramDesc *, std::unique_ptr<ProgramDescBind>>;
static ProgDescMap *g_bind_map = nullptr;

class ProgramDescBind {
public:
  static ProgramDescBind &Instance(ProgramDesc *prog) {
    if (g_bind_map == nullptr) {
      g_bind_map = new ProgDescMap();
    }
    auto &map = *g_bind_map;
    auto &ptr = map[prog];

    if (ptr == nullptr) {
      ptr.reset(new ProgramDescBind(prog));
    }
    return *ptr;
  }

  BlockDescBind *AppendBlock(BlockDescBind *parent) {
    auto *b = prog_->add_blocks();
    std::cerr << "block ptr " << b << std::endl;
    std::cerr << "pass ptr " << parent << std::endl;
    b->set_parent_idx(parent->id());
    b->set_idx(prog_->blocks_size() - 1);
    blocks_.emplace_back(this, b);
    return &blocks_.back();
  }

  BlockDescBind *Root() { return &blocks_.front(); }

  BlockDescBind *Block(size_t idx) { return &blocks_[idx]; }

  std::string DebugString() { return Proto()->DebugString(); }

  size_t Size() const { return blocks_.size(); }

  ProgramDesc *Proto() {
    for (auto &block : blocks_) {
      block.Sync();
    }
    return prog_;
  }

private:
  explicit ProgramDescBind(ProgramDesc *prog) : prog_(prog) {
    for (auto &block : *prog->mutable_blocks()) {
      blocks_.emplace_back(this, &block);
    }
  }

  // Not owned
  ProgramDesc *prog_;

  std::vector<BlockDescBind> blocks_;
};

void BindProgramDesc(py::module &m) {
  py::class_<ProgramDescBind>(m, "ProgramDesc", "")
      .def_static("instance",
                  []() -> ProgramDescBind * {
                    return &ProgramDescBind::Instance(&GetProgramDesc());
                  },
                  py::return_value_policy::reference)
      .def_static("__create_program_desc__",
                  []() -> ProgramDescBind * {
                    // Only used for unit-test
                    auto *prog_desc = new ProgramDesc;
                    auto *block = prog_desc->mutable_blocks()->Add();
                    block->set_idx(0);
                    block->set_parent_idx(-1);
                    return &ProgramDescBind::Instance(prog_desc);
                  },
                  py::return_value_policy::reference)
      .def("append_block",
           &ProgramDescBind::AppendBlock,
           py::return_value_policy::reference)
      .def("root_block",
           &ProgramDescBind::Root,
           py::return_value_policy::reference)
      .def("block", &ProgramDescBind::Block, py::return_value_policy::reference)
      .def("__str__", &ProgramDescBind::DebugString)
      .def("num_blocks", &ProgramDescBind::Size);
}

void BindBlockDesc(py::module &m) {
  using namespace paddle::framework;  // NOLINT
  py::class_<BlockDescBind>(m, "BlockDesc", "")
      .def_property_readonly("id", &BlockDescBind::id)
      .def_property_readonly("parent", &BlockDescBind::Parent)
      .def("append_op",
           &BlockDescBind::AppendOp,
           py::return_value_policy::reference)
      .def("new_var",
           [](BlockDesc &self) { return self.add_vars(); },
           py::return_value_policy::reference);
}

void BindVarDsec(py::module &m) {
  py::class_<VarDesc>(m, "VarDesc", "");
  //  using namespace paddle::framework;  // NOLINT
  //  py::class_<VarDesc>(m, "VarDesc", "")
  //      .def(py::init<>())
  //      .def("set_name",
  //           [](VarDesc &self, const std::string &name) { self.set_name(name);
  //           })
  //      .def("set_shape",
  //           [](VarDesc &self, const std::vector<int64_t> &dims) {
  //             VectorToRepeated(dims,
  //             self.mutable_lod_tensor()->mutable_dims());
  //           })
  //      .def("set_data_type",
  //           [](VarDesc &self, int type_id) {
  //             LoDTensorDesc *lod_tensor_desc = self.mutable_lod_tensor();
  //             lod_tensor_desc->set_data_type(static_cast<DataType>(type_id));
  //           })
  //      .def("shape", [](VarDesc &self) {
  //        const LoDTensorDesc &lod_tensor_desc = self.lod_tensor();
  //        return RepeatedToVector(lod_tensor_desc.dims());
  //      });
}

void BindOpDesc(py::module &m) {
  //  auto op_desc_set_var = [](OpDesc::Var *var,
  //                            const std::string &parameter,
  //                            const std::vector<std::string> &arguments) {
  //    var->set_parameter(parameter);
  //    VectorToRepeated(arguments, var->mutable_arguments());
  //  };
  //
  //  auto op_desc_set_attr = [](OpDesc &desc, const std::string &name) {
  //    auto attr = desc.add_attrs();
  //    attr->set_name(name);
  //    return attr;
  //  };
  py::class_<OpDescBind>(m, "OpDesc", "");

  //      .def("type", [](OpDesc &op) { return op.type(); })
  //      .def("set_input",
  //           [op_desc_set_var](OpDesc &self,
  //                             const std::string &parameter,
  //                             const std::vector<std::string> &arguments) {
  //             auto ipt = self.add_inputs();
  //             op_desc_set_var(ipt, parameter, arguments);
  //           })
  //      .def("input_names",
  //           [](OpDesc &self) {
  //             std::vector<std::string> ret_val;
  //             ret_val.reserve(static_cast<size_t>(self.inputs().size()));
  //             std::transform(
  //                 self.inputs().begin(),
  //                 self.inputs().end(),
  //                 std::back_inserter(ret_val),
  //                 [](const OpDesc::Var &var) { return var.parameter(); });
  //             return ret_val;
  //           })
  //      .def("__str__", [](OpDesc &self) { return self.DebugString(); })
  //      .def("set_output",
  //           [op_desc_set_var](OpDesc &self,
  //                             const std::string &parameter,
  //                             const std::vector<std::string> &arguments) {
  //             auto opt = self.add_outputs();
  //             op_desc_set_var(opt, parameter, arguments);
  //           })
  //      .def("set_attr",
  //           [op_desc_set_attr](OpDesc &self, const std::string &name, int i)
  //           {
  //             op_desc_set_attr(self, name)->set_i(i);
  //           });
}
}  // namespace pybind
}  // namespace paddle
