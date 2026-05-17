/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once

#include <Python.h>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/pybind/sot/eval_frame_tools.h"
#include "paddle/fluid/pybind/sot/frame_proxy.h"
#include "paddle/fluid/pybind/sot/macros.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/utils/pybind.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"

namespace py = pybind11;
#define PYBIND11_DETAILED_ERROR_MESSAGES
#if SOT_IS_SUPPORTED

class CompiledGuardLookup;

class GuardBase {
 public:
  GuardBase() = default;
  bool check_pybind(py::handle value) { return check(value.ptr()); }

  virtual bool check(PyObject* value) = 0;
  virtual std::string get_guard_name() const = 0;
  virtual ~GuardBase() = default;
};

class LambdaGuard : public GuardBase {
 public:
  explicit LambdaGuard(PyObject* guard_check_fn)
      : guard_check_fn_(guard_check_fn) {}

  explicit LambdaGuard(const py::function& guard_check_fn)
      : guard_check_fn_(guard_check_fn.ptr()) {
    Py_INCREF(guard_check_fn_);
  }

  ~LambdaGuard() { Py_DECREF(guard_check_fn_); }

  bool check(PyObject* value) override;
  std::string get_guard_name() const override { return "LambdaGuard"; }

 private:
  PyObject* guard_check_fn_;
};

class GuardGroup : public GuardBase {
 public:
  explicit GuardGroup(const std::vector<std::shared_ptr<GuardBase>>& guards) {
    for (auto& guard : guards) {
      if (auto group = dynamic_cast<GuardGroup*>(guard.get())) {
        guards_.insert(
            guards_.end(), group->guards_.begin(), group->guards_.end());
      } else {
        guards_.push_back(std::move(guard));
      }
    }
  }
  bool check(PyObject* value) override;
  std::string get_guard_name() const override { return "GuardGroup"; }

 private:
  std::vector<std::shared_ptr<GuardBase>> guards_;
};

class TypeMatchGuard : public GuardBase {
 public:
  explicit TypeMatchGuard(PyTypeObject* type_ptr) : expected_(type_ptr) {}
  explicit TypeMatchGuard(PyObject* type_ptr)
      : expected_(reinterpret_cast<PyTypeObject*>(type_ptr)) {}
  explicit TypeMatchGuard(const py::type& py_type)
      : expected_(reinterpret_cast<PyTypeObject*>(py_type.ptr())) {}

  bool check(PyObject* value) override;
  std::string get_guard_name() const override { return "TypeMatchGuard"; }

 private:
  PyTypeObject* expected_;
};

class IdMatchGuard : public GuardBase {
 public:
  explicit IdMatchGuard(PyObject* obj_ptr)
      : expected_(reinterpret_cast<PyObject*>(obj_ptr)) {}
  explicit IdMatchGuard(const py::object& py_obj)
      : expected_(reinterpret_cast<PyObject*>(py_obj.ptr())) {}

  bool check(PyObject* value) override;
  std::string get_guard_name() const override { return "IdMatchGuard"; }

 private:
  PyObject* expected_;
};

class ValueMatchGuard : public GuardBase {
 public:
  explicit ValueMatchGuard(PyObject* value_ptr)
      : expected_value_(value_ptr), expected_type_(value_ptr->ob_type) {}

  explicit ValueMatchGuard(const py::object& py_value)
      : expected_value_(py_value.ptr()),
        expected_type_(Py_TYPE(py_value.ptr())) {
    Py_INCREF(expected_value_);
  }

  ~ValueMatchGuard() { Py_DECREF(expected_value_); }

  bool check(PyObject* value) override;
  std::string get_guard_name() const override { return "ValueMatchGuard"; }

 private:
  PyObject* expected_value_;
  PyTypeObject* expected_type_;
};

class LengthMatchGuard : public GuardBase {
 public:
  explicit LengthMatchGuard(const Py_ssize_t& length) : expected_(length) {}

  bool check(PyObject* value) override;
  std::string get_guard_name() const override { return "LengthMatchGuard"; }

 private:
  Py_ssize_t expected_;
};

class DtypeMatchGuard : public GuardBase {
 public:
  explicit DtypeMatchGuard(const paddle::framework::proto::VarType& dtype_ptr)
      : expected_(dtype_ptr.type()) {}

  explicit DtypeMatchGuard(const phi::DataType& dtype_ptr)
      : expected_(phi::TransToProtoVarType(dtype_ptr)) {}

  bool check(PyObject* value) override;
  std::string get_guard_name() const override { return "DtypeMatchGuard"; }

 private:
  int expected_;
};

class ShapeMatchGuard : public GuardBase {
 public:
  explicit ShapeMatchGuard(const std::vector<py::object>& shape,
                           int64_t min_non_specialized_number)
      : min_non_specialized_number_(min_non_specialized_number) {
    expected_.resize(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
      if (py::isinstance<py::int_>(shape[i]) && shape[i].cast<int64_t>() >= 0) {
        expected_[i] = std::make_optional(shape[i].cast<int64_t>());
      }
    }
  }

  bool check(PyObject* value) override;
  std::string get_guard_name() const override { return "ShapeMatchGuard"; }

 private:
  std::vector<std::optional<int64_t>> expected_;
  int64_t min_non_specialized_number_;
};

class AttributeMatchGuard : public GuardBase {
 public:
  AttributeMatchGuard(const py::object& obj, const std::string& attr_name)
      : attr_ptr_(PyObject_GetAttrString(obj.ptr(), attr_name.c_str())),
        attr_name_(attr_name) {}

  bool check(PyObject* value) override;
  std::string get_guard_name() const override { return "AttributeMatchGuard"; }

 private:
  PyObject* attr_ptr_;
  std::string attr_name_;
};

class LayerMatchGuard : public GuardBase {
 public:
  explicit LayerMatchGuard(const py::object& layer_obj)
      : layer_ptr_(layer_obj.ptr()),
        training_(layer_obj.attr("training").cast<bool>()) {}

  bool check(PyObject* value) override;
  std::string get_guard_name() const override { return "LayerMatchGuard"; }

 private:
  PyObject* layer_ptr_;
  bool training_;
};

class InstanceCheckGuard : public GuardBase {
 public:
  explicit InstanceCheckGuard(const py::object& py_type)
      : expected_(py_type.ptr()) {
    Py_INCREF(expected_);
  }

  ~InstanceCheckGuard() override { Py_DECREF(expected_); }

  bool check(PyObject* value) override;
  std::string get_guard_name() const override { return "InstanceCheckGuard"; }

 private:
  PyObject* expected_;
};

class NumPyDtypeMatchGuard : public GuardBase {
 public:
  explicit NumPyDtypeMatchGuard(const py::object& dtype)
      : expected_(dtype.ptr()) {
    Py_INCREF(expected_);
  }

  ~NumPyDtypeMatchGuard() override { Py_DECREF(expected_); }

  bool check(PyObject* value) override;
  std::string get_guard_name() const override { return "NumPyDtypeMatchGuard"; }

 private:
  PyObject* expected_;
};

class NumPyArrayValueMatchGuard : public GuardBase {
 public:
  explicit NumPyArrayValueMatchGuard(const py::object& array)
      : expected_(array.ptr()) {
    Py_INCREF(expected_);
  }

  ~NumPyArrayValueMatchGuard() override { Py_DECREF(expected_); }

  bool check(PyObject* value) override;
  std::string get_guard_name() const override {
    return "NumPyArrayValueMatchGuard";
  }

 private:
  PyObject* expected_;
};

class NumPyArrayShapeMatchGuard : public GuardBase {
 public:
  explicit NumPyArrayShapeMatchGuard(const std::vector<py::object>& shape,
                                     int64_t min_non_specialized_number)
      : min_non_specialized_number_(min_non_specialized_number) {
    expected_.resize(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
      if (py::isinstance<py::int_>(shape[i]) && shape[i].cast<int64_t>() >= 0) {
        expected_[i] = std::make_optional(shape[i].cast<int64_t>());
      }
    }
  }

  bool check(PyObject* value) override;
  std::string get_guard_name() const override {
    return "NumPyArrayShapeMatchGuard";
  }

 private:
  std::vector<std::optional<int64_t>> expected_;
  int64_t min_non_specialized_number_;
};

class WeakRefMatchGuard : public GuardBase {
 public:
  explicit WeakRefMatchGuard(const py::object& obj) {
    expected_ = PyWeakref_NewRef(obj.ptr(), nullptr);
  }

  ~WeakRefMatchGuard() override { PyObject_ClearWeakRefs(expected_); }

  bool check(PyObject* value) override;
  std::string get_guard_name() const override { return "WeakRefMatchGuard"; }

 private:
  PyObject* expected_;
};

class IsNotDenseTensorHoldAllocationMatchGuard : public GuardBase {
 public:
  bool check(PyObject* value) override;
  std::string get_guard_name() const override {
    return "IsNotDenseTensorHoldAllocationMatchGuard";
  }
};

enum class CompiledGuardAttrKind : uint8_t {
  GENERIC,
  TRAINING,
  SUB_LAYERS,
  FORWARD_PRE_HOOKS,
  FORWARD_POST_HOOKS,
  FUNC,
  CODE,
  GLOBALS,
  CALL,
  FORWARD,
  STOP_GRADIENT,
};

class CompiledGuard {
 public:
  explicit CompiledGuard(const py::list& specs);
  bool check_pybind(py::handle frame) {
    return check(reinterpret_cast<FrameProxy*>(frame.ptr()));
  }
  bool check(FrameProxy* frame);
  std::string stringify() const;

 private:
  friend class CompiledGuardLookup;

  enum class AccessKind {
    LOCAL,
    GLOBAL,
    BUILTIN,
    CONSTANT,
    ATTR,
    ITEM,
  };

  enum class OpKind {
    GRAD_ENABLED,
    TYPE_MATCH,
    INSTANCE_CHECK,
    ID_MATCH,
    VALUE_MATCH,
    LENGTH_MATCH,
    LAYER_MATCH,
    LAYER_MATCH_GROUP,
    TENSOR_SHAPE,
    TENSOR_DTYPE,
    TENSOR_IS_DIST,
    TENSOR_META,
    TENSOR_DIST_META,
    TENSOR_NOT_HOLD_ALLOCATION,
    NUMPY_DTYPE,
    NUMPY_SHAPE,
    WEAKREF_MATCH,
    EXPR_MATCH,
  };

  struct AccessStep {
    AccessKind kind;
    std::string name;
    CompiledGuardAttrKind attr_kind{CompiledGuardAttrKind::GENERIC};
    Py_hash_t key_hash{-1};
    py::object name_object;
    py::object value;
  };

  struct AccessNode {
    std::optional<size_t> parent;
    AccessStep step;
  };

  struct AccessCache {
    explicit AccessCache(CompiledGuard* guard);
    ~AccessCache();

    CompiledGuard* guard;
  };

  enum class ExprKind {
    CONSTANT,
    ACCESS,
    UNARY,
    BINARY,
  };

  struct GuardExpr {
    ExprKind kind;
    std::string op;
    py::object value;
    std::vector<AccessStep> access_path;
    std::shared_ptr<GuardExpr> lhs;
    std::shared_ptr<GuardExpr> rhs;
  };

  struct LayerMatchItem {
    struct AttrLengthCheck {
      size_t access_id{0};
      CompiledGuardAttrKind attr_kind{CompiledGuardAttrKind::GENERIC};
      py::object name_object;
      Py_hash_t name_hash{-1};
      Py_ssize_t expected_length{0};
      bool require_dict_length{false};
    };

    struct MethodCodeCheck {
      size_t access_id{0};
      size_t group_index{0};
      std::string method_name;
      py::object method_name_object;
      py::object expected_code;
      Py_hash_t method_name_hash{-1};
      bool expected_instance_override_absent{false};
    };

    struct SelfLengthCheck {
      size_t access_id{0};
      Py_ssize_t expected_length{0};
      bool require_dict_length{false};
    };

    struct AttrValueCheck {
      size_t access_id{0};
      CompiledGuardAttrKind attr_kind{CompiledGuardAttrKind::GENERIC};
      py::object name_object;
      Py_hash_t name_hash{-1};
      py::object expected;
      bool value_match_by_identity{false};
    };

    size_t access_id{0};
    py::object expected;
    PyObject** expected_layer_dict_ptr{nullptr};
    uint64_t expected_layer_dict_version{0};
    std::vector<AttrLengthCheck> attr_length_checks;
    std::vector<MethodCodeCheck> method_code_checks;
    bool use_sub_layer_tree{false};
    bool cache_value{false};
    bool expected_bool{false};
  };

  struct LayerMethodCodeGroup {
    std::string method_name;
    py::object method_name_object;
    py::object expected_code;
    PyTypeObject* expected_type{nullptr};
    Py_hash_t method_name_hash{-1};
  };

  struct LayerSubLayerTreeChild {
    py::object key;
    Py_hash_t key_hash{-1};
    size_t node_index{0};
  };

  struct LayerClassWeakRefCheck {
    size_t access_id{0};
    py::object name_object;
    Py_hash_t name_hash{-1};
    py::object expected;
  };

  struct LayerSubLayerTreeNode {
    std::optional<size_t> item_index;
    uint8_t check_flags{0};
    std::vector<LayerMatchItem::SelfLengthCheck> self_length_checks;
    std::vector<LayerMatchItem::AttrLengthCheck> attr_length_checks;
    std::vector<LayerMatchItem::AttrValueCheck> attr_value_checks;
    std::vector<LayerMatchItem::MethodCodeCheck> method_code_checks;
    std::vector<LayerClassWeakRefCheck> class_weakref_checks;
    std::vector<LayerSubLayerTreeChild> children;
  };

  struct LayerSubLayerTreeRoot {
    size_t base_access_id{0};
    size_t node_index{0};
  };

  struct GuardOp {
    OpKind kind;
    std::vector<AccessStep> access_path;
    std::shared_ptr<GuardExpr> expr;
    py::object expected;
    py::object expected_dist_mesh_shape;
    py::object expected_dist_process_ids;
    py::object expected_dist_dims_mapping;
    py::object expected_dist_local_shape;
    py::object expected_dist_info_from_tensor;
    PyTypeObject* expected_type{nullptr};
    std::vector<std::optional<int64_t>> expected_shape;
    std::vector<int64_t> expected_dist_mesh_shape_values;
    std::vector<int64_t> expected_dist_process_ids_values;
    std::vector<int64_t> expected_dist_dims_mapping_values;
    std::vector<int64_t> expected_dist_local_shape_values;
    size_t access_id{0};
    int expected_dtype{0};
    bool expected_bool{false};
    bool expected_is_dict{false};
    bool require_dict_length{false};
    bool value_match_by_identity{false};
    PyObject** expected_layer_dict_ptr{nullptr};
    uint64_t expected_layer_dict_version{0};
    std::vector<LayerMatchItem> layer_match_items;
    std::vector<LayerMethodCodeGroup> layer_method_code_groups;
    std::vector<LayerSubLayerTreeNode> layer_sub_layer_tree_nodes;
    std::vector<LayerSubLayerTreeRoot> layer_sub_layer_tree_roots;
    Py_ssize_t expected_length{0};
    int64_t min_non_specialized_number{0};
  };

  static std::vector<AccessStep> ParseAccessPath(py::handle access);
  static std::vector<std::optional<int64_t>> ParseShape(py::handle shape);
  static std::vector<int64_t> ParseInt64Vector(py::handle values,
                                               const std::string& name);
  static std::shared_ptr<GuardExpr> ParseExpr(py::handle expr);
  static std::string AccessStepKey(const AccessStep& step);
  static std::string GuardOpKey(const GuardOp& op);
  static bool CheckTensorDistMeta(PyObject* value, const GuardOp& op);
  size_t InternAccessPath(const std::vector<AccessStep>& access_path);
  void DeduplicateOps();
  void FuseTensorMetaOps();
  void FuseLayerMatchOps();
  std::string AccessPathKey(size_t access_id) const;
  std::string LookupGuardOpKey(const GuardOp& op) const;
  PyObject* EvalAccess(FrameProxy* frame,
                       const std::vector<AccessStep>& access_path) const;
  PyObject* EvalAccessNode(FrameProxy* frame,
                           size_t access_id,
                           AccessCache* cache) const;
  PyObject* EvalExpr(FrameProxy* frame, const GuardExpr& expr) const;
  bool CheckOp(FrameProxy* frame, const GuardOp& op, AccessCache* cache) const;

  std::vector<GuardOp> ops_;
  std::vector<AccessNode> access_nodes_;
  std::unordered_map<std::string, size_t> access_node_ids_;
  mutable std::vector<PyObject*> access_cache_values_;
  mutable std::vector<uint32_t> access_cache_evaluated_;
  mutable std::vector<unsigned char> access_cache_owned_;
  mutable uint32_t access_cache_generation_{0};
  mutable std::vector<size_t> access_cache_touched_;
  mutable std::vector<unsigned char> layer_method_code_required_;
  mutable std::vector<std::vector<PyTypeObject*>>
      layer_method_code_dynamic_required_;
};

class CompiledGuardLookup {
 public:
  CompiledGuardLookup();
  void add_guard(const std::shared_ptr<CompiledGuard>& guard, int cache_index);
  std::optional<int> lookup(FrameProxy* frame);
  std::string stringify() const;

 private:
  struct TrieEdge {
    std::string key;
    std::shared_ptr<CompiledGuard> guard;
    size_t op_index{0};
    size_t child{0};
  };

  struct TrieNode {
    std::vector<TrieEdge> edges;
    std::optional<int> return_cache_index;
  };

  struct LookupContext {
    CompiledGuard::AccessCache* GetCache(CompiledGuard* guard);
    std::unordered_map<CompiledGuard*,
                       std::unique_ptr<CompiledGuard::AccessCache>>
        caches;
  };

  std::optional<int> LookupNode(FrameProxy* frame,
                                size_t node_index,
                                LookupContext* context) const;

  std::vector<TrieNode> nodes_;
};

class GuardTreeNodeBase {
 public:
  virtual ~GuardTreeNodeBase() = default;
  virtual std::string stringify(int indent = 0) = 0;
};

class ExprNodeBase : public GuardTreeNodeBase,
                     public std::enable_shared_from_this<ExprNodeBase> {
 public:
  virtual PyObject* eval(FrameProxy* frame) = 0;
  virtual ~ExprNodeBase() = default;
};

class ConstantExprNode : public ExprNodeBase {
 public:
  explicit ConstantExprNode(PyObject* value_ptr) : value_ptr_(value_ptr) {}
  explicit ConstantExprNode(const py::object& value_obj)
      : value_ptr_(value_obj.ptr()) {
    Py_INCREF(value_ptr_);
  }
  ~ConstantExprNode() { Py_DECREF(value_ptr_); }
  PyObject* eval(FrameProxy* frame) override;
  std::string stringify(int indent = 0) override;

 private:
  PyObject* value_ptr_;
};

class ExternVarExprNode : public ExprNodeBase {
 public:
  explicit ExternVarExprNode(const std::string& var_name,
                             const py::object& value_obj)
      : value_ptr_(value_obj.ptr()), var_name_(var_name) {
    Py_INCREF(value_ptr_);
  }

  ~ExternVarExprNode() { Py_DECREF(value_ptr_); }
  PyObject* eval(FrameProxy* frame) override;
  std::string stringify(int indent = 0) override;

 private:
  PyObject* value_ptr_;
  std::string var_name_;
};

class LocalVarExprNode : public ExprNodeBase {
 public:
  explicit LocalVarExprNode(const std::string& var_name)
      : var_name_(var_name) {}

  PyObject* eval(FrameProxy* frame) override;
  std::string stringify(int indent = 0) override;

 private:
  std::string var_name_;
};

class GlobalVarExprNode : public ExprNodeBase {
 public:
  explicit GlobalVarExprNode(const std::string& var_name)
      : var_name_(var_name) {}

  PyObject* eval(FrameProxy* frame) override;
  std::string stringify(int indent = 0) override;

 private:
  std::string var_name_;
};

class AttributeExprNode : public ExprNodeBase {
 public:
  explicit AttributeExprNode(std::shared_ptr<ExprNodeBase> var_expr,
                             const std::string& attr_name)
      : var_expr_(var_expr), attr_name_(attr_name) {}

  PyObject* eval(FrameProxy* frame) override;
  std::string stringify(int indent = 0) override;

 private:
  std::shared_ptr<ExprNodeBase> var_expr_;
  std::string attr_name_;
};

class ItemExprNode : public ExprNodeBase {
 public:
  explicit ItemExprNode(std::shared_ptr<ExprNodeBase> var_expr,
                        std::shared_ptr<ExprNodeBase> key_expr)
      : var_expr_(var_expr), key_expr_(key_expr) {}

  PyObject* eval(FrameProxy* frame) override;
  std::string stringify(int indent = 0) override;

 private:
  std::shared_ptr<ExprNodeBase> var_expr_;
  std::shared_ptr<ExprNodeBase> key_expr_;
};

class BinaryExprNode : public ExprNodeBase {
 public:
  enum class OpType { COMPARE, NUMBER };

  static constexpr std::array<std::pair<const char*, std::pair<OpType, int>>,
                              18>
      kOpMap = {{
          {"<", {OpType::COMPARE, Py_LT}},
          {"<=", {OpType::COMPARE, Py_LE}},
          {"==", {OpType::COMPARE, Py_EQ}},
          {"!=", {OpType::COMPARE, Py_NE}},
          {">", {OpType::COMPARE, Py_GT}},
          {">=", {OpType::COMPARE, Py_GE}},
          {"+", {OpType::NUMBER, 0}},
          {"-", {OpType::NUMBER, 1}},
          {"*", {OpType::NUMBER, 2}},
          {"/", {OpType::NUMBER, 3}},
          {"//", {OpType::NUMBER, 4}},
          {"%", {OpType::NUMBER, 5}},
          {"**", {OpType::NUMBER, 6}},
          {"<<", {OpType::NUMBER, 7}},
          {">>", {OpType::NUMBER, 8}},
          {"&", {OpType::NUMBER, 9}},
          {"|", {OpType::NUMBER, 10}},
          {"^", {OpType::NUMBER, 11}},
      }};

  explicit BinaryExprNode(std::shared_ptr<ExprNodeBase> lhs,
                          std::shared_ptr<ExprNodeBase> rhs,
                          const std::string& op_str)
      : lhs_(lhs), rhs_(rhs) {
    auto it =
        std::find_if(kOpMap.begin(), kOpMap.end(), [&op_str](const auto& pair) {
          return std::string(pair.first) == op_str;
        });
    if (it == kOpMap.end()) {
      throw std::invalid_argument("Invalid operator: " + op_str);
    }
    op_str_ = it->first;
    op_type_ = it->second.first;
    op_code_ = it->second.second;
  }

  PyObject* eval(FrameProxy* frame) override;
  std::string stringify(int indent = 0) override;

 private:
  std::shared_ptr<ExprNodeBase> lhs_;
  std::shared_ptr<ExprNodeBase> rhs_;
  std::string op_str_;
  OpType op_type_;
  int op_code_;
};

class UnaryExprNode : public ExprNodeBase {
 public:
  enum class OpType { NUMBER, LOGICAL };

  static constexpr std::array<std::pair<const char*, std::pair<OpType, int>>, 6>
      kOpMap = {{{"+", {OpType::NUMBER, 0}},
                 {"-", {OpType::NUMBER, 1}},
                 {"~", {OpType::NUMBER, 2}},
                 {"not", {OpType::LOGICAL, 0}},
                 {"!", {OpType::LOGICAL, 0}},
                 {"bool", {OpType::LOGICAL, 1}}}};

  explicit UnaryExprNode(std::shared_ptr<ExprNodeBase> expr,
                         const std::string& op_str)
      : expr_(expr) {
    auto it =
        std::find_if(kOpMap.begin(), kOpMap.end(), [&op_str](const auto& pair) {
          return std::string(pair.first) == op_str;
        });
    if (it == kOpMap.end()) {
      throw std::invalid_argument("Invalid operator: " + op_str);
    }
    op_str_ = it->first;
    op_type_ = it->second.first;
    op_code_ = it->second.second;
  }

  PyObject* eval(FrameProxy* frame) override;
  std::string stringify(int indent = 0) override;

 private:
  std::shared_ptr<ExprNodeBase> expr_;
  std::string op_str_;
  OpType op_type_;
  int op_code_;
};

class GuardNodeBase : public GuardTreeNodeBase {
 public:
  std::vector<std::shared_ptr<GuardNodeBase>> next_guard_nodes;
  // return_cache_index is used to record the index of the guard list
  std::optional<int> return_cache_index;
  GuardNodeBase(std::vector<std::shared_ptr<GuardNodeBase>> next_guard_nodes,
                std::optional<int> return_cache_index)
      : next_guard_nodes(next_guard_nodes),
        return_cache_index(return_cache_index) {}
  virtual ~GuardNodeBase() = default;
  virtual std::optional<int> lookup(FrameProxy* frame) = 0;
  std::optional<int> lookup_next(FrameProxy* frame);
};

class DummyGuardNode : public GuardNodeBase {
 public:
  explicit DummyGuardNode(
      bool return_true,
      std::vector<std::shared_ptr<GuardNodeBase>> next_guard_nodes,
      std::optional<int> return_cache_index)
      : GuardNodeBase(next_guard_nodes, return_cache_index),
        return_true_(return_true) {}
  virtual ~DummyGuardNode() = default;
  std::string stringify(int indent = 0) override;
  std::optional<int> lookup(FrameProxy* frame) override;

 private:
  bool return_true_;
};

class ExprGuardNode : public GuardNodeBase {
 public:
  std::shared_ptr<ExprNodeBase> expr;
  explicit ExprGuardNode(
      std::shared_ptr<ExprNodeBase> expr,
      std::vector<std::shared_ptr<GuardNodeBase>> next_guard_nodes,
      std::optional<int> return_cache_index)
      : GuardNodeBase(next_guard_nodes, return_cache_index), expr(expr) {}

  std::string stringify(int indent = 0) override;
  std::optional<int> lookup(FrameProxy* frame) override;
};

template <size_t N>
class CheckGuardNode : public GuardNodeBase {
 public:
  std::array<std::shared_ptr<ExprNodeBase>, N> exprs;
  explicit CheckGuardNode(
      std::array<std::shared_ptr<ExprNodeBase>, N> exprs,
      std::vector<std::shared_ptr<GuardNodeBase>> next_guard_nodes,
      std::optional<int> return_cache_index)
      : GuardNodeBase(next_guard_nodes, return_cache_index), exprs(exprs) {}
  virtual ~CheckGuardNode() = default;
  virtual std::string get_guard_name() const = 0;
  virtual bool check(std::array<PyObject*, N> values) = 0;
  std::string stringify(int indent = 0) override {
    std::stringstream ss;
    ss << std::string(indent, ' ') << get_guard_name();
    ss << "(";
    for (size_t i = 0; i < N; ++i) {
      if (i > 0) {
        ss << " | ";
      }
      ss << exprs[i]->stringify();
    }
    ss << ")";
    if (!next_guard_nodes.empty()) {
      ss << " |" << std::endl;
      for (auto& next_guard_node : next_guard_nodes) {
        ss << std::string(indent + 2, ' ');
        ss << next_guard_node->stringify(indent + 2) << std::endl;
      }
    }
    return ss.str();
  }
  std::optional<int> lookup(FrameProxy* frame) override {
    std::array<PyObject*, N> values = {};
    for (size_t i = 0; i < N; ++i) {
      values[i] = exprs[i]->eval(frame);
      if (values[i]) {
        Py_INCREF(values[i]);
      }
    }
    std::optional<int> ret = std::nullopt;
    if (check(values)) {
      ret = lookup_next(frame);
    }
    for (size_t i = 0; i < N; ++i) {
      if (values[i]) {
        Py_DECREF(values[i]);
      }
    }
    return ret;
  }
};

class TensorDistMetaMatchGuardNode : public CheckGuardNode<2> {
 public:
  explicit TensorDistMetaMatchGuardNode(
      const py::object& obj,
      std::array<std::shared_ptr<ExprNodeBase>, 2> exprs,
      std::vector<std::shared_ptr<GuardNodeBase>> next_guard_nodes,
      std::optional<int> return_cache_index)
      : CheckGuardNode<2>(exprs, next_guard_nodes, return_cache_index) {
    if (!obj.is(py::none())) {
      mesh_shape_expected_ =
          obj.attr("mesh").attr("shape").cast<std::vector<int>>();
      mesh_process_ids_expected_ =
          obj.attr("mesh").attr("process_ids").cast<std::vector<int>>();
      dims_mapping_expected_ = obj.attr("dims_mapping").ptr();
      local_shape_expected_ = obj.attr("local_shape").ptr();

      is_dist_ = true;
      Py_INCREF(dims_mapping_expected_.value());
      Py_INCREF(local_shape_expected_.value());
    }
  }

  ~TensorDistMetaMatchGuardNode() override {
    if (is_dist_) {
      Py_DECREF(dims_mapping_expected_.value());
      Py_DECREF(local_shape_expected_.value());
    }
  }
  bool check(std::array<PyObject*, 2> values) override;
  std::string get_guard_name() const override {
    return "TensorDistMetaMatchGuard";
  }

 private:
  bool is_dist_ = false;
  std::optional<std::vector<int>> mesh_shape_expected_;
  std::optional<std::vector<int>> mesh_process_ids_expected_;
  std::optional<PyObject*> dims_mapping_expected_;
  std::optional<PyObject*> local_shape_expected_;
};

class LegacyGuardNode : public CheckGuardNode<1> {
 public:
  std::shared_ptr<GuardBase> guard;
  explicit LegacyGuardNode(
      std::shared_ptr<GuardBase> guard,
      std::array<std::shared_ptr<ExprNodeBase>, 1> exprs,
      std::vector<std::shared_ptr<GuardNodeBase>> next_guard_nodes,
      std::optional<int> return_cache_index)
      : CheckGuardNode<1>(exprs, next_guard_nodes, return_cache_index),
        guard(guard) {}
  virtual ~LegacyGuardNode() = default;
  std::string get_guard_name() const override {
    return guard->get_guard_name();
  };
  bool check(std::array<PyObject*, 1> values) override;
};

class IsGradEnabledGuardNode : public CheckGuardNode<0> {
 public:
  bool is_grad_enabled_;
  explicit IsGradEnabledGuardNode(
      bool is_grad_enabled,
      std::vector<std::shared_ptr<GuardNodeBase>> next_guard_nodes,
      std::optional<int> return_cache_index)
      : CheckGuardNode<0>({}, next_guard_nodes, return_cache_index) {
    is_grad_enabled_ = is_grad_enabled;
  }
  virtual ~IsGradEnabledGuardNode() = default;
  std::string get_guard_name() const override { return "IsGradEnabledGuard"; };
  bool check(std::array<PyObject*, 0> values) override;
};

class GuardTree {
 public:
  GuardTree(const std::vector<std::vector<std::shared_ptr<GuardNodeBase>>>&
                guard_chain_list) {
    for (size_t index = 0; index < guard_chain_list.size(); ++index) {
      add_guard_chain(guard_chain_list[index]);
    }
  }
  void add_guard_chain(
      const std::vector<std::shared_ptr<GuardNodeBase>>& guard_chain);
  std::string stringify();
  std::optional<int> lookup(FrameProxy* frame);
  std::vector<std::shared_ptr<GuardNodeBase>> get_guard_nodes() const;

 private:
  std::vector<std::shared_ptr<GuardNodeBase>> guard_nodes_;
};

std::string guard_tree_to_str(const GuardTree& guard_tree);

#endif
