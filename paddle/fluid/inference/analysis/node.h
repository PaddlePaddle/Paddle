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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/inference/analysis/device.h"
#include "paddle/fluid/inference/analysis/helper.h"

namespace paddle {
namespace inference {
namespace analysis {

/*
 * Node Representation.
 *
 * This is a very important class for analysis. It is the base class of all
 * nodes computed by a program
 * that may be used as operands to other nodes.
 * Node is the super class of other important classes such as Function and
 * Value, some nodes can have
 * a name.
 */
class Node {
 public:
  // Node type. NOTE the new node types should add here.
  enum class Type { kNone = -1, kFunction, kValue, kFunctionBlock };

  Node() : id_(counter_), extra_info_(nullptr) { ++counter_; }

  // Cast to a subclass type, Function for example.
  template <typename Subclass>
  Subclass &As() {
    return *reinterpret_cast<Subclass *>(this);
  }

  // Formatted representation of this Node.
  virtual std::string repr() const = 0;

  template <typename T>
  T &NewAttr(const std::string &name) {
    auto it = attrs_.find(name);
    PADDLE_ENFORCE(it == attrs_.end(), "set duplicate attribute %s", name);
    return it->second.As<T>();
  }

  size_t id() const { return id_; }

  bool deleted() const { return deleted_; }
  void SetDeleted() { deleted_ = true; }

  void SetName(const std::string &name) { name_ = name; }
  const std::string &name() const { return name_; }

  void SetType(Type type) { type_ = type; }
  Type type() const { return type_; }

  void *extra_info() const { return extra_info_; }
  void SetExtraInfo(void *extra_info) { extra_info_ = extra_info; }

  static unsigned int counter() { return counter_; }

  // Input links.
  std::vector<Node *> inlinks;
  // Output links.
  std::vector<Node *> outlinks;

  // A helper class to maintain the status from Pass.
  // TODO(superjomn) add a checker here to ensure the T is primary.
  struct Attr {
    // NOTE T should be a primary type or a struct combined by several primary
    // types.
    // NOTE the STL containers should not use here.
    // Some usages
    // Attr attr;
    // T data;
    // attr.data.assign((char*)data, sizeof(data));
    template <typename T>
    T &As() {
      // init storage in the first usage.
      if (data.empty()) data.resize(sizeof(T));
      PADDLE_ENFORCE_EQ(data.size(), sizeof(T), "Node attr type recast error");
      return *reinterpret_cast<T *>(&data[0]);
    }

   private:
    std::string data;
  };

  virtual ~Node() {}

  PADDLE_DISALLOW_COPY_AND_ASSIGN(Node);

 protected:
  // The id number not the name is a node's unique identifier in the computation
  // graph.
  size_t id_;
  std::string name_;
  Type type_{Type::kNone};
  // Mark this node is deleted by some pass.
  bool deleted_{false};

  void *extra_info_;

  mutable std::unordered_map<std::string, Attr> attrs_;

 private:
  static unsigned counter_;
};

class Function;
/*
 * Value represents a value node, it has some attributes including dims, data
 * type and so on.
 */
class Value : public Node {
 public:
  enum class DataType { kInt32, kInt64, kFloat32, kFloat64 };
  using Dims = std::vector<short>;

  void SetDataType(DataType data_type) { data_type_ = data_type; }
  DataType data_type() const { return data_type_; }

  void SetDims(const Dims &dims) { dims_ = dims; }
  const Dims &dims() const { return dims_; }

  Device device() const { return device_; }
  void SetDevice(Device device) { device_ = device; }

  std::string repr() const override;

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
  std::string repr() const override;

  PADDLE_DISALLOW_COPY_AND_ASSIGN(Function);

 protected:
  Function() { SetType(Node::Type::kFunction); }
  friend class NodeMap;
};

/*
 * FunctionBlock is a Node that contains a sub-graph multiple Node.
 */
struct FunctionBlock : public Node {
  std::string repr() const override;
  std::vector<Node *> subgraph;
};

class NodeMap {
 public:
  // Create a new node with type.
  Node *Create(Node::Type type);

  // Get a node by its id.
  Node *Get(size_t id);

  void Delete(size_t id);

  size_t size() const { return nodes_.size(); }

 private:
  std::vector<std::unique_ptr<Node>> nodes_;
  std::unordered_map<std::string, Node *> map_;
};

}  // namespace analysis
}  // namespace inference
}  // namespace analysis
