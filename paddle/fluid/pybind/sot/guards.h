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

  enum class ExprUnaryOp {
    POSITIVE,
    NEGATIVE,
    BITWISE_NOT,
    LOGICAL_NOT,
    BOOL,
  };

  enum class ExprBinaryOp {
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE,
    ADD,
    SUB,
    MUL,
    TRUE_DIV,
    FLOOR_DIV,
    MOD,
    POW,
    LSHIFT,
    RSHIFT,
    BITWISE_AND,
    BITWISE_OR,
    BITWISE_XOR,
  };

  struct GuardExpr {
    ExprKind kind;
    ExprUnaryOp unary_op{ExprUnaryOp::POSITIVE};
    ExprBinaryOp binary_op{ExprBinaryOp::EQ};
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

#endif
