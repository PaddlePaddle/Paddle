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

/*
 * This file defines the Node class and its subclasses. A Node is the basis
 * analysis element in a computation graph.
 * There are basically two kinds of nodes, the function node and value node.
 */
#pragma once

#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/inference/analysis/device.h"
#include "paddle/fluid/inference/analysis/dot.h"
#include "paddle/fluid/inference/analysis/helper.h"

namespace paddle {
namespace inference {
namespace analysis {

class NodeMap;

/*
 * Node Representation.
 *
 * This is a very important class for analysis. It is the base class of all
 * nodes computed by a program that may be used as operands to other nodes.
 * Node is the super class of other important classes such as Function and
 * Value, some nodes can have a name.
 */
class Node {
 public:
  // Node type. NOTE the new node types should add here.
  enum class Type { kNone = -1, kFunction, kValue, kFunctionBlock };

  Node() = default;

  struct Attr;

  // Cast to a subclass type, Function for example.
  template <typename Subclass>
  Subclass &As() {
    return *dynamic_cast<Subclass *>(this);
  }

  // Formatted representation of this Node.
  virtual std::string repr() const {
    return name() + "(" + std::to_string(id()) + ")";
  }

  // DOT node representation. One Node type can customize its own node
  // representation.
  virtual std::vector<Dot::Attr> dot_attrs() const {
    return std::vector<Dot::Attr>({Dot::Attr("style", "filled")});
  }

  // Get an additional attribute and convert it to T data type. NOTE this will
  // silently create a new attribute if not exists.
  Attr &attr(const std::string &name) const { return attrs_[name]; }

  int id() const { return id_; }

  // The Protobuf description is set/get with a void* to decouple Node interface
  // from a specific kind of Protobuf message.
  void SetPbDesc(void *pb) { attr("pb_desc").Pointer() = pb; }
  void *pb_desc() const { return attr("pb_desc").Pointer(); }

  void SetDeleted() { deleted_ = true; }
  bool deleted() const { return deleted_; }

  void SetName(const std::string &name) { name_ = name; }
  const std::string &name() const { return name_; }

  void SetType(Type type) { type_ = type; }
  Type type() const { return type_; }

  // Input links.
  std::vector<Node *> inlinks;
  // Output links.
  std::vector<Node *> outlinks;

  // A helper class to maintain the status from Pass.
  struct Attr {
    // NOTE T should be a primary type or a struct combined by several primary
    // types.
    // NOTE the STL containers should not use here.
    // Some usages
    //   Attr attr;
    //   attr.Bool() = true;

    bool &Bool() { return As<bool>(); }
    float &Float() { return As<float>(); }
    int32_t &Int32() { return As<int32_t>(); }
    int64_t &Int64() { return As<int64_t>(); }
    void *&Pointer() { return As<void *>(); }

   private:
    template <typename T>
    T &As() {
      // init storage in the first usage.
      if (data_.empty()) {
        VLOG(4) << "resize data to " << sizeof(T);
        type_hash_ = typeid(T).hash_code();
        data_.resize(sizeof(T));
      }
      PADDLE_ENFORCE(type_hash_ == typeid(T).hash_code(),
                     "type not matched, origin is %s, want %s",
                     DataTypeNamer::Global().repr(type_hash_),
                     DataTypeNamer::Global().repr<T>());
      PADDLE_ENFORCE_EQ(data_.size(), sizeof(T), "Node attr type recast error");
      return *reinterpret_cast<T *>(&data_[0]);
    }

   private:
    std::string data_;
    size_t type_hash_{std::numeric_limits<size_t>::max()};
  };

  // Type checks.
  bool IsFunction() const { return type_ == Node::Type::kFunction; }
  bool IsValue() const { return type_ == Node::Type::kValue; }
  bool IsFunctionBlock() const { return type_ == Node::Type::kFunctionBlock; }

  virtual ~Node() {}

  friend class NodeMap;

  PADDLE_DISALLOW_COPY_AND_ASSIGN(Node);

 protected:
  // The id number not the name is a node's unique identifier in the computation
  // graph.
  int id_{-1};
  std::string name_;
  Type type_{Type::kNone};
  // Mark this node is deleted by some pass.
  bool deleted_{false};
  mutable std::unordered_map<std::string, Attr> attrs_;
};

class Function;
/*
 * Value represents a value node, it has some attributes including dims, data
 * type and so on.
 */
class Value : public Node {
 public:
  enum class DataType { kInt32, kInt64, kFloat32, kFloat64 };
  using Dims = std::vector<int>;

  void SetDataType(DataType data_type) { data_type_ = data_type; }
  DataType data_type() const { return data_type_; }

  void SetDims(const Dims &dims) { dims_ = dims; }
  const Dims &dims() const { return dims_; }

  Device device() const { return device_; }
  void SetDevice(Device device) { device_ = device; }

  std::vector<Dot::Attr> dot_attrs() const override;

  PADDLE_DISALLOW_COPY_AND_ASSIGN(Value);

 protected:
  Value() { SetType(Node::Type::kValue); }
  friend class NodeMap;

 private:
  DataType data_type_;
  Dims dims_;
  Device device_;
};

/*
 * Function represents any kind of executable concepts that takes several Values
 * as input, and outputs several Values.
 */
class Function : public Node {
 public:
  std::vector<Dot::Attr> dot_attrs() const override;

  // Get the operator's type from Desc.
  const std::string &func_type() const { return func_type_; }
  // Set the operator's type.
  void SetFuncType(const std::string &func_type) { func_type_ = func_type; }

  PADDLE_DISALLOW_COPY_AND_ASSIGN(Function);

 protected:
  std::string func_type_;
  Function() { SetType(Node::Type::kFunction); }
  friend class NodeMap;
};

/*
 * FunctionBlock is a Node that contains a sub-graph multiple Node.
 */
struct FunctionBlock : public Node {
  std::string repr() const override { return "block-" + std::to_string(id()); }
  std::vector<Node *> subgraph;
};

class NodeMap {
 public:
  // Create a new node with type.
  Node *Create(Node::Type type);

  // Get a node by its id.
  Node *GetMutable(size_t id);

  const Node &Get(size_t id) const;

  void Delete(size_t id);

  const std::vector<std::unique_ptr<Node>> &nodes() { return nodes_; }

  size_t size() const { return nodes_.size(); }

 private:
  std::vector<std::unique_ptr<Node>> nodes_;
  std::unordered_map<std::string, Node *> map_;
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
