// Copyright (c) 2021 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

namespace cinn {
namespace utils {

void ResetDotCounters();

struct DotNode;
struct DotCluster;
struct DotEdge;
struct DotAttr;

/*
 * A Dot template that helps to build a DOT graph definition.
 */
class DotLang {
 public:
  DotLang() = default;

  explicit DotLang(const std::vector<DotAttr>& attrs) : attrs_(attrs) {}

  /**
   * Add a node to the DOT graph.
   * @param id Unique ID for this node.
   * @param attrs DOT attributes.
   * @param label Name of the node.
   */
  void AddNode(const std::string& id,
               const std::vector<DotAttr>& attrs,
               std::string label = "",
               std::string cluster_id = "",
               bool allow_duplicate = false);

  /**
   * Add a subgraph to the DOT graph.
   * @param id Unique ID for this subgraph.
   * @param attrs DOT attributes.
   */
  void AddCluster(const std::string& id, const std::vector<DotAttr>& attrs);

  /**
   * Add an edge to the DOT graph.
   * @param source The id of the source of the edge.
   * @param target The id of the sink of the edge.
   * @param attrs The attributes of the edge.
   */
  void AddEdge(const std::string& source,
               const std::string& target,
               const std::vector<DotAttr>& attrs);

  std::string operator()() const { return Build(); }

 private:
  // Compile to DOT language codes.
  std::string Build() const;

  std::map<std::string, DotNode> nodes_;
  std::map<std::string, DotCluster> clusters_;
  std::vector<DotEdge> edges_;
  std::vector<DotAttr> attrs_;
};

struct DotAttr {
  std::string key;
  std::string value;

  DotAttr(const std::string& key, const std::string& value)
      : key(key), value(value) {}

  std::string repr() const;
};

struct DotNode {
  std::string name;
  std::vector<DotAttr> attrs;

  DotNode() = default;
  DotNode(const std::string& name,
          const std::vector<DotAttr>& attrs,
          const std::string& cluster_id);

  std::string id() const { return id_; }
  std::string cluster_id() const { return cluster_id_; }

  std::string repr() const;

 private:
  std::string id_;
  std::string cluster_id_;
};

struct DotCluster {
  std::string name;
  std::vector<DotAttr> attrs;

  DotCluster() = default;
  DotCluster(const std::string& name, const std::vector<DotAttr>& attrs);

  void Insert(DotNode* node) { nodes_.insert(node); }

  std::string id() const { return id_; }
  std::set<DotNode*> nodes() const { return nodes_; }

 private:
  std::string id_;
  std::set<DotNode*> nodes_;  // Not owned
};

struct DotEdge {
  std::string source;
  std::string target;
  std::vector<DotAttr> attrs;

  DotEdge(const std::string& source,
          const std::string& target,
          const std::vector<DotAttr>& attrs)
      : source(source), target(target), attrs(attrs) {}

  std::string repr() const;
};

}  // namespace utils
}  // namespace cinn
