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
#include <iostream>
#include "paddle/framework/attribute.h"

// Cast boost::variant for PyBind.
// Copy from
// https://github.com/pybind/pybind11/issues/576#issuecomment-269563199
namespace pybind11 {
namespace detail {

// Can be replaced by a generic lambda in C++14
struct variant_caster_visitor : public boost::static_visitor<handle> {
  return_value_policy policy;
  handle parent;

  variant_caster_visitor(return_value_policy policy, handle parent)
      : policy(policy), parent(parent) {}

  template <class T>
  handle operator()(T const &src) const {
    return make_caster<T>::cast(src, policy, parent);
  }
};

template <class Variant>
struct variant_caster;

template <template <class...> class V, class... Ts>
struct variant_caster<V<Ts...>> {
  using Type = V<Ts...>;

  template <class T>
  bool try_load(handle src, bool convert) {
    auto caster = make_caster<T>();
    if (!load_success_ && caster.load(src, convert)) {
      load_success_ = true;
      value = cast_op<T>(caster);
      return true;
    }
    return false;
  }

  bool load(handle src, bool convert) {
    auto unused = {false, try_load<Ts>(src, convert)...};
    (void)(unused);
    return load_success_;
  }

  static handle cast(Type const &src,
                     return_value_policy policy,
                     handle parent) {
    variant_caster_visitor visitor(policy, parent);
    return boost::apply_visitor(visitor, src);
  }

  PYBIND11_TYPE_CASTER(Type, _("Variant"));
  bool load_success_{false};
};

// Add specialization for concrete variant type
template <class... Args>
struct type_caster<boost::variant<Args...>>
    : variant_caster<boost::variant<Args...>> {};

}  // namespace detail
}  // namespace pybind11

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
  for (const auto &elem : vec) {
    *repeated_field->Add() = elem;
  }
}

template <typename RepeatedField>
inline void VectorToRepeated(const std::vector<bool> &vec,
                             RepeatedField *repeated_field) {
  repeated_field->Reserve(vec.size());
  for (auto elem : vec) {
    *repeated_field->Add() = elem;
  }
}

class ProgramDescBind;
class OpDescBind;
class BlockDescBind;
class VarDescBind;

class VarDescBind {
public:
  explicit VarDescBind(const std::string &name) { desc_.set_name(name); }

  VarDesc *Proto() { return &desc_; }

  py::bytes Name() { return desc_.name(); }

  void SetShape(const std::vector<int64_t> &dims) {
    VectorToRepeated(dims, desc_.mutable_lod_tensor()->mutable_dims());
  }

  void SetDataType(framework::DataType data_type) {
    desc_.mutable_lod_tensor()->set_data_type(data_type);
  }

  std::vector<int64_t> Shape() {
    return RepeatedToVector(desc_.lod_tensor().dims());
  }

  framework::DataType DataType() { return desc_.lod_tensor().data_type(); }

private:
  VarDesc desc_;
};

class OpDescBind {
public:
  OpDesc *Proto() {
    Sync();
    return &op_desc_;
  }

  std::string Type() const { return op_desc_.type(); }

  void SetType(const std::string &type) { op_desc_.set_type(type); }

  const std::vector<std::string> &Input(const std::string &name) const {
    auto it = inputs_.find(name);
    PADDLE_ENFORCE(
        it != inputs_.end(), "Input %s cannot be found in Op %s", name, Type());
    return it->second;
  }

  std::vector<std::string> InputNames() const {
    std::vector<std::string> retv;
    retv.reserve(this->inputs_.size());
    for (auto &ipt : this->inputs_) {
      retv.push_back(ipt.first);
    }
    return retv;
  }

  void SetInput(const std::string &param_name,
                const std::vector<std::string> &args) {
    need_update_ = true;
    inputs_[param_name] = args;
  }

  const std::vector<std::string> &Output(const std::string &name) const {
    auto it = outputs_.find(name);
    PADDLE_ENFORCE(it != outputs_.end(),
                   "Output %s cannot be found in Op %s",
                   name,
                   Type());
    return it->second;
  }

  std::vector<std::string> OutputNames() const {
    std::vector<std::string> retv;
    retv.reserve(this->outputs_.size());
    for (auto &ipt : this->outputs_) {
      retv.push_back(ipt.first);
    }
    return retv;
  }

  void SetOutput(const std::string &param_name,
                 const std::vector<std::string> &args) {
    need_update_ = true;
    this->outputs_[param_name] = args;
  }

  std::string DebugString() { return this->Proto()->DebugString(); }

  struct SetAttrDescVisitor : public boost::static_visitor<void> {
    explicit SetAttrDescVisitor(OpDesc::Attr *attr) : attr_(attr) {}
    mutable OpDesc::Attr *attr_;
    void operator()(int v) const { attr_->set_i(v); }
    void operator()(float v) const { attr_->set_f(v); }
    void operator()(const std::string &v) const { attr_->set_s(v); }
    void operator()(bool b) const { attr_->set_b(b); }

    void operator()(const std::vector<int> &v) const {
      VectorToRepeated(v, attr_->mutable_ints());
    }
    void operator()(const std::vector<float> &v) const {
      VectorToRepeated(v, attr_->mutable_floats());
    }
    void operator()(const std::vector<std::string> &v) const {
      VectorToRepeated(v, attr_->mutable_strings());
    }
    void operator()(const std::vector<bool> &v) const {
      VectorToRepeated(v, attr_->mutable_bools());
    }
    void operator()(BlockDesc *desc) const {
      attr_->set_block_idx(desc->idx());
    }
    void operator()(boost::blank) const { PADDLE_THROW("Unexpected branch"); }
  };

  void Sync() {
    if (need_update_) {
      this->op_desc_.mutable_inputs()->Clear();
      for (auto &ipt : inputs_) {
        auto *input = op_desc_.add_inputs();
        input->set_parameter(ipt.first);
        VectorToRepeated(ipt.second, input->mutable_arguments());
      }

      this->op_desc_.mutable_outputs()->Clear();
      for (auto &opt : outputs_) {
        auto *output = op_desc_.add_outputs();
        output->set_parameter(opt.first);
        VectorToRepeated(opt.second, output->mutable_arguments());
      }

      this->op_desc_.mutable_attrs()->Clear();
      for (auto &attr : attrs_) {
        auto *attr_desc = op_desc_.add_attrs();
        attr_desc->set_name(attr.first);
        attr_desc->set_type(
            static_cast<framework::AttrType>(attr.second.which() - 1));
        boost::apply_visitor(SetAttrDescVisitor(attr_desc), attr.second);
      }

      need_update_ = false;
    }
  }

  bool HasAttr(const std::string &name) const {
    return attrs_.find(name) != attrs_.end();
  }

  framework::AttrType GetAttrType(const std::string &name) const {
    auto it = attrs_.find(name);
    PADDLE_ENFORCE(it != attrs_.end(), "Attribute %s is not found", name);
    return static_cast<framework::AttrType>(it->second.which() - 1);
  }

  std::vector<std::string> AttrNames() const {
    std::vector<std::string> retv;
    retv.reserve(attrs_.size());
    for (auto &attr : attrs_) {
      retv.push_back(attr.first);
    }
    return retv;
  }

  void SetAttr(const std::string &name, const Attribute &v) {
    this->attrs_[name] = v;
  }

  void SetBlockAttr(const std::string &name, BlockDescBind &block);

  int GetBlockAttr(const std::string &name) const {
    auto it = attrs_.find(name);
    PADDLE_ENFORCE(it != attrs_.end(), "Attribute %s is not found", name);
    return boost::get<BlockDesc *>(it->second)->idx();
  }

  Attribute GetAttr(const std::string &name) const {
    auto it = attrs_.find(name);
    PADDLE_ENFORCE(it != attrs_.end(), "Attribute %s is not found", name);
    return it->second;
  }

private:
  OpDesc op_desc_;
  std::unordered_map<std::string, std::vector<std::string>> inputs_;
  std::unordered_map<std::string, std::vector<std::string>> outputs_;
  std::unordered_map<std::string, Attribute> attrs_;

  bool need_update_{false};
};

class BlockDescBind {
public:
  BlockDescBind(ProgramDescBind *prog, BlockDesc *desc)
      : prog_(prog), desc_(desc), need_update_(false) {}

  BlockDescBind(const BlockDescBind &o) = delete;
  BlockDescBind &operator=(const BlockDescBind &o) = delete;

  int32_t id() const { return desc_->idx(); }

  int32_t Parent() const { return desc_->parent_idx(); }

  VarDescBind *NewVar(py::bytes name_bytes) {
    std::string name = name_bytes;
    need_update_ = true;
    auto it = vars_.find(name);
    PADDLE_ENFORCE(it == vars_.end(), "Duplicated variable %s", name);
    auto var = new VarDescBind(name);
    vars_[name].reset(var);
    return var;
  }

  VarDescBind *Var(py::bytes name_bytes) const {
    std::string name = name_bytes;
    auto it = vars_.find(name);
    PADDLE_ENFORCE(
        it != vars_.end(), "Can not find variable %s in current block.", name);
    return it->second.get();
  }

  std::vector<VarDescBind *> AllVars() const {
    std::vector<VarDescBind *> res;
    for (const auto &p : vars_) {
      res.push_back(p.second.get());
    }
    return res;
  }

  BlockDescBind *ParentBlock() const;

  OpDescBind *AppendOp() {
    need_update_ = true;
    ops_.emplace_back(new OpDescBind());
    return ops_.back().get();
  }

  OpDescBind *PrependOp() {
    need_update_ = true;
    ops_.emplace_front(new OpDescBind());
    return ops_.front().get();
  }

  std::vector<OpDescBind *> AllOps() const {
    std::vector<OpDescBind *> res;
    for (const auto &op : ops_) {
      res.push_back(op.get());
    }
    return res;
  }

  void Sync() {
    if (need_update_) {
      auto &op_field = *this->desc_->mutable_ops();
      op_field.Clear();
      op_field.Reserve(static_cast<int>(ops_.size()));
      for (auto &op_desc : ops_) {
        op_field.AddAllocated(op_desc->Proto());
      }
      need_update_ = false;
    }
  }

  BlockDesc *RawPtr() { return desc_; }

private:
  ProgramDescBind *prog_;  // not_own
  BlockDesc *desc_;        // not_own
  bool need_update_;

  std::deque<std::unique_ptr<OpDescBind>> ops_;
  std::unordered_map<std::string, std::unique_ptr<VarDescBind>> vars_;
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
  ProgramDescBind(const ProgramDescBind &o) = delete;
  ProgramDescBind &operator=(const ProgramDescBind &o) = delete;

  BlockDescBind *AppendBlock(const BlockDescBind &parent) {
    auto *b = prog_->add_blocks();
    b->set_parent_idx(parent.id());
    b->set_idx(prog_->blocks_size() - 1);
    blocks_.emplace_back(new BlockDescBind(this, b));
    return blocks_.back().get();
  }

  BlockDescBind *Block(size_t idx) { return blocks_[idx].get(); }

  std::string DebugString() { return Proto()->DebugString(); }

  size_t Size() const { return blocks_.size(); }

  ProgramDesc *Proto() {
    for (auto &block : blocks_) {
      block->Sync();
    }
    return prog_;
  }

private:
  explicit ProgramDescBind(ProgramDesc *prog) : prog_(prog) {
    for (auto &block : *prog->mutable_blocks()) {
      blocks_.emplace_back(new BlockDescBind(this, &block));
    }
  }

  // Not owned
  ProgramDesc *prog_;

  std::vector<std::unique_ptr<BlockDescBind>> blocks_;
};

BlockDescBind *BlockDescBind::ParentBlock() const {
  if (this->desc_->parent_idx() == -1) {
    return nullptr;
  }
  return prog_->Block(static_cast<size_t>(this->desc_->parent_idx()));
}

void OpDescBind::SetBlockAttr(const std::string &name, BlockDescBind &block) {
  BlockDesc *desc = block.RawPtr();
  this->attrs_[name] = desc;
}

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
      .def("block", &ProgramDescBind::Block, py::return_value_policy::reference)
      .def("__str__", &ProgramDescBind::DebugString)
      .def("num_blocks", &ProgramDescBind::Size);
}

void BindBlockDesc(py::module &m) {
  py::class_<BlockDescBind>(m, "BlockDesc", "")
      .def_property_readonly("id", &BlockDescBind::id)
      .def_property_readonly("parent", &BlockDescBind::Parent)
      .def("append_op",
           &BlockDescBind::AppendOp,
           py::return_value_policy::reference)
      .def("prepend_op",
           &BlockDescBind::PrependOp,
           py::return_value_policy::reference)
      .def(
          "new_var", &BlockDescBind::NewVar, py::return_value_policy::reference)
      .def("var", &BlockDescBind::Var, py::return_value_policy::reference)
      .def("all_vars",
           &BlockDescBind::AllVars,
           py::return_value_policy::reference)
      .def("all_ops",
           &BlockDescBind::AllOps,
           py::return_value_policy::reference);
}

void BindVarDsec(py::module &m) {
  py::enum_<framework::DataType>(m, "DataType", "")
      .value("BOOL", DataType::BOOL)
      .value("INT16", DataType::INT16)
      .value("INT32", DataType::INT32)
      .value("INT64", DataType::INT64)
      .value("FP16", DataType::FP16)
      .value("FP32", DataType::FP32)
      .value("FP64", DataType::FP64);

  py::class_<VarDescBind>(m, "VarDesc", "")
      .def("name", &VarDescBind::Name, py::return_value_policy::reference)
      .def("set_shape", &VarDescBind::SetShape)
      .def("set_data_type", &VarDescBind::SetDataType)
      .def("shape", &VarDescBind::Shape, py::return_value_policy::reference)
      .def("data_type", &VarDescBind::DataType);
}

void BindOpDesc(py::module &m) {
  py::enum_<framework::AttrType>(m, "AttrType", "")
      .value("INT", AttrType::INT)
      .value("INTS", AttrType::INTS)
      .value("FLOAT", AttrType::FLOAT)
      .value("FLOATS", AttrType::FLOATS)
      .value("STRING", AttrType::STRING)
      .value("STRINGS", AttrType::STRINGS)
      .value("BOOL", AttrType::BOOLEAN)
      .value("BOOLS", AttrType::BOOLEANS)
      .value("BLOCK", AttrType::BLOCK);

  py::class_<OpDescBind> op_desc(m, "OpDesc", "");
  op_desc.def("type", &OpDescBind::Type)
      .def("set_type", &OpDescBind::SetType)
      .def("input", &OpDescBind::Input)
      .def("input_names", &OpDescBind::InputNames)
      .def("set_input", &OpDescBind::SetInput)
      .def("output", &OpDescBind::Output)
      .def("output_names", &OpDescBind::OutputNames)
      .def("set_output", &OpDescBind::SetOutput)
      .def("__str__", &OpDescBind::DebugString)
      .def("__repr__", &OpDescBind::DebugString)
      .def("has_attr", &OpDescBind::HasAttr)
      .def("attr_type", &OpDescBind::GetAttrType)
      .def("attr_names", &OpDescBind::AttrNames)
      .def("set_attr", &OpDescBind::SetAttr)
      .def("attr", &OpDescBind::GetAttr)
      .def("set_block_attr", &OpDescBind::SetBlockAttr)
      .def("get_block_attr", &OpDescBind::GetBlockAttr);
}

}  // namespace pybind
}  // namespace paddle
