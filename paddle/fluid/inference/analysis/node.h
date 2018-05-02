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

  Node() : id_(counter_) { ++counter_; }

  // Cast to a subclass type, Function for example.
  template <typename Subclass>
  Subclass &As() {
    return *reinterpret_cast<Subclass *>(this);
  }

  // Formatted representation of this Node.
  virtual std::string repr() const = 0;

  size_t id() const { return id_; }
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


 protected:
  // The id number not the name is a node's unique identifier in the computation
  // graph.
  size_t id_;
  std::string name_;
  Type type_{Type::kNone};
  PADDLE_DISALLOW_COPY_AND_ASSIGN(Node);

  void* extra_info_;

 private:
  static unsigned counter_;
};

struct Function;
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
  Dims &dims() const { return dims_; }

  Device device() const { return device_; }
  void SetDevice(Device device) { device_ = device; }

  std::string repr() const override;

 protected:
  Value() { SetType(Node::Type::kValue); }
  friend class NodeMap;
  PADDLE_DISALLOW_COPY_AND_ASSIGN(Value);

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
  virtual std::string repr() const override;

 protected:
  Function() { SetType(Node::Type::kFunction); }
  friend class NodeMap;
  PADDLE_DISALLOW_COPY_AND_ASSIGN(Function);
};

/*
 * FunctionBlock is a Node that contains a sub-graph multiple Node.
 */
struct FunctionBlock : public Node {};

class NodeMap {
 public:
  // Create a new node with type.
  Node *Create(Node::Type type);

  // Get a node by its id.
  Node *Get(size_t id);

  size_t size() const { return nodes_.size(); }

 private:
  std::vector<std::unique_ptr<Node>> nodes_;
  std::unordered_map<std::string, Node *> map_;
};

}  // namespace analysis
}  // namespace inference
}  // namespace analysis
