/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <vector>

#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/utils/any.h"
namespace paddle {
namespace framework {
class OpDesc;
class VarDesc;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {

// Node should only created by Graph::CreateXXXNode().
// 1. Every Node should be part of a graph. No dangling Node exists.
// 2. Node only contains members necessary for building graph structure.
//    It doesn't contain other unrelated members, such as device, etc.
//
// Sometimes, for specific usages, Node needs to have additional members,
// such as device_placement, version in order to be executed. It is suggested
// to use composition pattern.
//
// class RunnableOp {
//    RunnableOp(ir::Node* n) : n_(n) { n_.WrappedBy(this); }
//
//    int any_thing_;
// }
//
// RunnableOp is owned by the ir::Node that composes it. In other words.
// ir::Node will be responsible for deleting RunnableOp, say, when ir::Node
// is deleted from the graph.
class Node {
 public:
  virtual ~Node() {
    if (!wrapper_.empty()) {
      VLOG(10) << "ir::Node deleting a wrapper node " << Name();
      wrapper_deleter_();
    }
  }

  enum class Type { kOperation, kVariable };
  enum class Dep { kSame = 0, kBefore = 1, kAfter = 2, kNoDep = 3 };
#if !defined(_WIN32)  // msvc not support constexpr correctly.
  static constexpr char kControlDepVarName[] = "__control_var";
#else
  static const char kControlDepVarName[];
#endif

  Type NodeType() const { return type_; }

  std::string Name() const { return name_; }

  VarDesc* Var() const {
    PADDLE_ENFORCE_EQ(IsVar(), true,
                      platform::errors::InvalidArgument(
                          "Node(%s) must be kVariable type, but not %d.", name_,
                          static_cast<int>(type_)));
    return var_desc_.get();
  }

  OpDesc* Op() const {
    PADDLE_ENFORCE_EQ(IsOp(), true,
                      platform::errors::InvalidArgument(
                          "Node(%s) must be kOperation type, but not %d.",
                          name_, static_cast<int>(type_)));
    return op_desc_.get();
  }

  // Set the `wrapper` that wraps the Node. `wrapper` is owned by Node.
  template <typename T>
  void WrappedBy(T* wrapper) {
    if (!wrapper_.empty()) {
      wrapper_deleter_();
    }
    wrapper_ = wrapper;
    wrapper_deleter_ = [wrapper]() { delete wrapper; };
    wrapper_type_ = std::type_index(typeid(T));
  }

  // Return a reference to the `wrapper`.
  template <typename T>
  T& Wrapper() {
    try {
      return *paddle::any_cast<T*>(wrapper_);
    } catch (paddle::bad_any_cast&) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Invalid wrapper type error, expected %s, actual %s.",
          typeid(T).name(), wrapper_type_.name()));
    }
  }

  // Test if the Node is wrapped by type T.
  template <typename T>
  bool IsWrappedBy() const {
    return std::type_index(typeid(T)) == wrapper_type_;
  }

  // Please don't use this API!
  int id() const { return id_; }

  // Only use this for auto parallel.
  // A node does not have original desc if the return is zero.
  uint64_t OriginalDescId() const { return original_desc_id_; }

  bool IsOp() const { return type_ == Type::kOperation; }
  bool IsVar() const { return type_ == Type::kVariable; }
  bool IsCtrlVar() const {
    return type_ == Type::kVariable &&
           Name().find(ir::Node::kControlDepVarName) != std::string::npos;
  }

  void RenameVar(const std::string& new_name) {
    PADDLE_ENFORCE_EQ(
        type_ == Type::kVariable && var_desc_, true,
        platform::errors::InvalidArgument("Node must be type of variable."));
    name_ = new_name;
    var_desc_->SetName(new_name);
  }

  int DescOrder() const { return desc_order_; }

  int GetVarNodeBlockId() const {
    PADDLE_ENFORCE_EQ(
        type_ == Type::kVariable && var_desc_, true,
        platform::errors::InvalidArgument("Node must be type of variable."));
    return block_id_;
  }

  const std::string ToString() const {
    if (IsOp()) {
      std::string op_str(Name());

      const auto& op = Op();
      if (op == nullptr) {
        // Node is an Op but hasn't OpDesc (often create by CreateEmptyNode),
        // like ScaleLossGradOp, it's type is OpHandle, which created by Pass
        // and then inserted into graph.
        // For OpHandle, we have to use Node's input and output for sorting.
        std::vector<Node*> sorted_inputs(inputs);
        std::vector<Node*> sorted_outputs(outputs);

        auto comparator = [](Node* a, Node* b) {
          return a->Name() > b->Name();
        };
        std::stable_sort(sorted_inputs.begin(), sorted_inputs.end(),
                         comparator);
        std::stable_sort(sorted_outputs.begin(), sorted_outputs.end(),
                         comparator);

        std::string out_str = "{";
        std::string pre_str = "";
        for (const auto& output : sorted_outputs) {
          out_str.append(pre_str + output->Name());
          pre_str = ", ";
        }
        out_str.append("} = ");

        std::string in_str = "(";
        pre_str = "";
        for (const auto& input : sorted_inputs) {
          in_str.append(pre_str + input->Name());
          pre_str = ", ";
        }
        in_str.append(")");
        op_str = out_str + op_str + in_str;
      } else {
        // A normal Op, has OpDesc, create from ProgramDesc
        std::string out_str = "{";
        std::string outer_pre_str = "";
        for (const auto& output : op->OutputNames()) {
          out_str.append(outer_pre_str + output + "=[");
          std::string inner_pre_str = "";
          for (const auto& arg : op->Output(output)) {
            out_str.append(inner_pre_str + arg);
            inner_pre_str = " ,";
          }
          outer_pre_str = ", ";
          out_str.append("]");
        }
        out_str.append("} = ");

        std::string in_str = "(";
        outer_pre_str = "";
        for (const auto& input : op->InputNames()) {
          in_str.append(outer_pre_str + input + "=[");
          std::string inner_pre_str = "";
          for (const auto& arg : op->Input(input)) {
            in_str.append(inner_pre_str + arg);
            inner_pre_str = " ,";
          }
          outer_pre_str = " ,";
          in_str.append("]");
        }
        in_str.append(")");
        op_str = out_str + op_str + in_str;
      }

      return op_str;
    }
    return Name();
  }

  std::vector<Node*> inputs;
  std::vector<Node*> outputs;

  // Because NO_DESC_ORDER is a constexpr number,
  // no one can change it, meanwhile, we need
  // check whether the DescOrder invalid sometime,
  // so expose it is a good idea
  static constexpr int NO_DESC_ORDER = INT_MAX;

 protected:
  std::string name_;
  std::unique_ptr<VarDesc> var_desc_;
  std::unique_ptr<OpDesc> op_desc_;
  Type type_;
  int id_;

  int desc_order_;
  int block_id_{-1};

  // Store the original id of var desc or op desc.
  // Only use this for auto parallel.
  uint64_t original_desc_id_{0};

 private:
  // ID can only set by a Graph.
  void SetId(int id) { id_ = id; }

  // desc_order can only set by a Graph when constructing a Graph from a
  // BlockDesc.
  void SetDescOrder(int desc_order) { desc_order_ = desc_order; }

  friend class Graph;
  friend std::unique_ptr<Node> CreateNodeForTest(const std::string& name,
                                                 Node::Type type);
  friend std::unique_ptr<Node> CreateNodeForTest(VarDesc* var_desc);
  friend std::unique_ptr<Node> CreateNodeForTest(OpDesc* op_desc);

  explicit Node(const std::string& name, Type type, int block_id = 0)
      : name_(name),
        var_desc_(nullptr),
        op_desc_(nullptr),
        type_(type),
        desc_order_(NO_DESC_ORDER),
        block_id_(block_id) {}

  explicit Node(VarDesc* var_desc, int block_id)
      : name_(var_desc->Name()),
        var_desc_(new VarDesc(*var_desc)),
        op_desc_(nullptr),
        type_(Type::kVariable),
        desc_order_(NO_DESC_ORDER),
        block_id_(block_id),
        original_desc_id_(var_desc->OriginalId()) {}

  explicit Node(OpDesc* op_desc)
      : name_(op_desc->Type()),
        var_desc_(nullptr),
        op_desc_(new OpDesc(*op_desc, op_desc->Block())),
        type_(Type::kOperation),
        desc_order_(NO_DESC_ORDER),
        original_desc_id_(op_desc->OriginalId()) {}

  Node() = delete;

  paddle::any wrapper_;
  std::function<void(void)> wrapper_deleter_;
  std::type_index wrapper_type_ = std::type_index(typeid(void));

  DISABLE_COPY_AND_ASSIGN(Node);
};

std::unique_ptr<Node> CreateNodeForTest(const std::string& name,
                                        Node::Type type);
std::unique_ptr<Node> CreateNodeForTest(VarDesc* var_desc);

std::unique_ptr<Node> CreateNodeForTest(OpDesc* op_desc);
}  // namespace ir
}  // namespace framework
}  // namespace paddle
