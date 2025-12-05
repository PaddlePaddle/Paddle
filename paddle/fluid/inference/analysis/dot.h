//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

/*
 * This file implements some helper classes and methods for DOT programming
 * support. It will give a visualization of the graph and that helps to debug
 * the logics of each Pass.
 */
#pragma once

#include <glog/logging.h>

#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
namespace paddle {
namespace inference {
namespace analysis {

static size_t dot_node_counter{0};

/*
 * A Dot template that helps to build a DOT graph definition.
 */
class Dot {
 public:
  struct Attr {
    std::string key;
    std::string value;

    Attr(const std::string& key, const std::string& value)
        : key(key), value(value) {}

    std::string repr() const {
      std::stringstream ss;
      ss << key << "=" << '"' << value << '"';
      return ss.str();
    }
  };

  struct Node {
    std::string name;
    std::vector<Attr> attrs;
    std::string comments;

    Node(const std::string& name,
         const std::vector<Attr>& attrs,
         std::string comments)
        : name(name),
          attrs(attrs),
          comments(comments),
          id_("node_" + std::to_string(dot_node_counter++)) {}

    Node(const std::string& name,
         const std::vector<Attr>& attrs,
         size_t id,
         std::string comments)
        : name(name),
          attrs(attrs),
          comments(comments),
          id_("node_" + std::to_string(id)) {}

    std::string id() const { return id_; }

    std::string repr() const {
      std::stringstream ss;
      PADDLE_ENFORCE_EQ(
          !name.empty(),
          true,
          common::errors::InvalidArgument("Sorry,but name is empty"));
      if (comments != "") {
        ss << "#" << std::regex_replace(comments, std::regex("\n"), "\n\t#")
           << "\n\t";
      }
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

   private:
    std::string id_;
  };

  struct Edge {
    std::string source;
    std::string target;
    std::vector<Attr> attrs;
    std::string label;

    Edge(const std::string& source,
         const std::string& target,
         const std::vector<Attr>& attrs,
         const std::string label = "")
        : source(source), target(target), attrs(attrs), label(label) {}

    std::string repr() const {
      std::stringstream ss;
      PADDLE_ENFORCE_EQ(
          !source.empty(),
          true,
          common::errors::InvalidArgument("Sorry,but source is empty"));
      PADDLE_ENFORCE_EQ(
          !target.empty(),
          true,
          common::errors::InvalidArgument("Sorry,but target is empty"));
      ss << source << "->" << target;
      if (attrs.empty() && label != "") {
        ss << "[label=" << '"' << label << '"' << "]";
        return ss.str();
      }
      for (size_t i = 0; i < attrs.size(); i++) {
        if (i == 0) {
          ss << "[label=" << '"' << label << '"' << " ";
        }
        ss << attrs[i].repr();
        ss << ((i < attrs.size() - 1) ? " " : "]");
      }
      return ss.str();
    }
  };

  Dot() = default;

  explicit Dot(const std::vector<Attr>& attrs) : attrs_(attrs) {}

  void AddNode(const std::string& id,
               const std::vector<Attr>& attrs,
               std::string label = "",
               bool use_local_id = false,
               std::string comments = "") {
    PADDLE_ENFORCE_EQ(
        !nodes_.count(id),
        true,
        common::errors::InvalidArgument("Sorry,but duplicate Node"));
    if (label.empty()) label = id;
    if (use_local_id) {
      nodes_.emplace(id, Node{label, attrs, local_node_counter_++, comments});
    } else {
      nodes_.emplace(id, Node{label, attrs, comments});
    }
  }
  bool ContainsNode(const std::string& id) const { return nodes_.count(id); }
  // If the graph has no nodes, then it is empty graph
  bool IsEmpty() const { return nodes_.empty(); }

  void AddEdge(const std::string& source,
               const std::string& target,
               const std::vector<Attr>& attrs,
               const std::string& label = "") {
    PADDLE_ENFORCE_EQ(
        !source.empty(),
        true,
        common::errors::InvalidArgument("Sorry,but source is empty"));
    PADDLE_ENFORCE_EQ(
        !target.empty(),
        true,
        common::errors::InvalidArgument("Sorry,but target is empty"));
    auto sid = nodes_.at(source).id();
    auto tid = nodes_.at(target).id();
    edges_.emplace_back(sid, tid, attrs, label);
  }

  // Compile to DOT language codes.
  std::string Build() const {
    std::stringstream ss;
    const std::string indent = "\t";
    ss << "digraph G {" << '\n';

    // Add graph attrs
    for (const auto& attr : attrs_) {
      ss << indent << attr.repr() << '\n';
    }
    // add nodes
    for (auto& item : nodes_) {
      ss << indent << item.second.repr() << '\n';
    }
    // add edges
    for (auto& edge : edges_) {
      ss << indent << edge.repr() << '\n';
    }
    ss << "} // end G";
    return ss.str();
  }

 private:
  std::unordered_map<std::string, Node> nodes_;
  std::vector<Edge> edges_;
  std::vector<Attr> attrs_;

  size_t local_node_counter_{0};
};
// Some attributes settings for reference
const std::vector<Dot::Attr> grey_box_attrs({
    Dot::Attr("style", "rounded,filled,bold"),  //
    Dot::Attr("shape", "box"),                  //
    Dot::Attr("color", "#999999"),              //
    Dot::Attr("fontcolor", "#ffffff"),          //
    Dot::Attr("width", "1.3"),                  //
    Dot::Attr("height", "0.84"),                //
    Dot::Attr("fontname", "Arial"),             //
});
const std::vector<Dot::Attr> teal_box_attrs({
    Dot::Attr("shape", "box"),                  //
    Dot::Attr("style", "rounded,filled,bold"),  //
    Dot::Attr("fontname", "Arial"),             //
    Dot::Attr("color", "#148b97"),              //
    Dot::Attr("fontcolor", "#ffffff"),          //
});
const std::vector<Dot::Attr> orange_box_attrs({
    Dot::Attr("style", "rounded,filled,bold"),  //
    Dot::Attr("shape", "box"),                  //
    Dot::Attr("color", "#FFE4B5"),              //
    Dot::Attr("fontcolor", "#ffffff"),          //
    Dot::Attr("width", "1.3"),                  //
    Dot::Attr("height", "0.84"),                //
    Dot::Attr("fontname", "Arial"),             //
});

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
