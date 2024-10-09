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

#include "paddle/cinn/utils/dot_lang.h"

#include <glog/logging.h>

#include <sstream>
#include "paddle/common/enforce.h"

namespace cinn {
namespace utils {

size_t dot_node_counter{0};
size_t dot_cluster_counter{0};

void ResetDotCounters() {
  dot_node_counter = 0;
  dot_cluster_counter = 0;
}

std::string DotAttr::repr() const {
  std::stringstream ss;
  ss << key << "=" << '"' << value << '"';
  return ss.str();
}

DotNode::DotNode(const std::string& name,
                 const std::vector<DotAttr>& attrs,
                 const std::string& cluster_id)
    : name(name), attrs(attrs), cluster_id_(cluster_id) {
  std::stringstream ss;
  ss << "node_" << dot_node_counter++;
  id_ = ss.str();
}

std::string DotNode::repr() const {
  std::stringstream ss;
  PADDLE_ENFORCE_NE(name.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The name of DotNode is empty! Please check."));
  ss << id_;
  if (attrs.empty()) {
    ss << "[label=" << '"' << name << '"' << "]";
    return ss.str();
  }
  for (size_t i = 0; i < attrs.size(); i++) {
    if (i == 0) {
      ss << "[label=" << '"' << name << '"' << " ";
    }
    ss << attrs[i].repr();
    ss << ((i < attrs.size() - 1) ? " " : "]");
  }
  return ss.str();
}

DotCluster::DotCluster(const std::string& name,
                       const std::vector<DotAttr>& attrs)
    : name(name), attrs(attrs) {
  std::stringstream ss;
  ss << "cluster_" << dot_cluster_counter++;
  id_ = ss.str();
}

std::string DotEdge::repr() const {
  std::stringstream ss;
  PADDLE_ENFORCE_NE(source.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The source of DotEdge is empty! Please check."));
  PADDLE_ENFORCE_NE(target.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The target of DotEdge is empty! Please check."));
  ss << source << "->" << target;
  for (size_t i = 0; i < attrs.size(); i++) {
    if (i == 0) {
      ss << "[";
    }
    ss << attrs[i].repr();
    ss << ((i < attrs.size() - 1) ? " " : "]");
  }
  return ss.str();
}

void DotLang::AddNode(const std::string& id,
                      const std::vector<DotAttr>& attrs,
                      std::string label,
                      std::string cluster_id,
                      bool allow_duplicate) {
  if (!allow_duplicate) {
    PADDLE_ENFORCE_EQ(!nodes_.count(id),
                      true,
                      ::common::errors::InvalidArgument(
                          "Duplicate Node %s, please check.", id));
  }
  if (!nodes_.count(id)) {
    if (label.empty()) {
      label = id;
    }
    nodes_.emplace(id, DotNode{label, attrs, cluster_id});
    if (!cluster_id.empty()) {
      PADDLE_ENFORCE_GE(clusters_.count(cluster_id),
                        1,
                        ::common::errors::InvalidArgument(
                            "Cluster %s is not existed.", cluster_id));
      clusters_[cluster_id].Insert(&nodes_[id]);
    }
  }
}

void DotLang::AddCluster(const std::string& id,
                         const std::vector<DotAttr>& attrs) {
  PADDLE_ENFORCE_EQ(!clusters_.count(id),
                    true,
                    ::common::errors::InvalidArgument(
                        "Duplicate Cluster %s, please check.", id));
  clusters_.emplace(id, DotCluster{id, attrs});
}

void DotLang::AddEdge(const std::string& source,
                      const std::string& target,
                      const std::vector<DotAttr>& attrs) {
  PADDLE_ENFORCE_NE(source.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The source of AddEdge is empty! Please check."));
  PADDLE_ENFORCE_NE(target.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The target of AddEdge is empty! Please check."));
  PADDLE_ENFORCE_EQ(nodes_.find(source) != nodes_.end(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Call AddNote to add %s to dot first.", source));
  PADDLE_ENFORCE_EQ(nodes_.find(target) != nodes_.end(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Call AddNote to add %s to dot first.", target));
  auto sid = nodes_.at(source).id();
  auto tid = nodes_.at(target).id();
  edges_.emplace_back(sid, tid, attrs);
}

std::string DotLang::Build() const {
  std::stringstream ss;
  const std::string indent = "   ";
  ss << "digraph G {" << '\n';

  // Add graph attrs
  for (const auto& attr : attrs_) {
    ss << indent << attr.repr() << '\n';
  }
  // add clusters
  for (auto& item : clusters_) {
    const auto& cluster = item.second;
    ss << indent << "subgraph " << cluster.id() << " {\n";
    ss << indent << indent << "label=\"" << item.first << "\"\n";
    if (!cluster.attrs.empty()) {
      for (size_t i = 0; i < cluster.attrs.size(); i++) {
        ss << indent << indent << cluster.attrs[i].repr() << "\n";
      }
    }
    for (auto* node : cluster.nodes()) {
      ss << indent << indent << node->repr() << "\n";
    }
    ss << indent << "}\n";
  }
  // add nodes
  for (auto& item : nodes_) {
    if (item.second.cluster_id().empty()) {
      ss << indent << item.second.repr() << '\n';
    }
  }
  // add edges
  for (auto& edge : edges_) {
    ss << indent << edge.repr() << '\n';
  }
  ss << "} // end G";
  return ss.str();
}

}  // namespace utils
}  // namespace cinn
