/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/profiler/event_node.h"

#include <limits.h>
#include <algorithm>
#include <deque>
#include <set>
#include <stack>

namespace paddle {
namespace platform {

HostTraceEventNode::~HostTraceEventNode() {
  // delete all runtime nodes and recursive delete children
  for (auto it = runtime_node_ptrs_.begin(); it != runtime_node_ptrs_.end();
       ++it) {
    delete *it;
  }
  for (auto it = children_.begin(); it != children_.end(); ++it) {
    delete *it;
  }
}

CudaRuntimeTraceEventNode::~CudaRuntimeTraceEventNode() {
  // delete all device nodes
  for (auto it = device_node_ptrs_.begin(); it != device_node_ptrs_.end();
       ++it) {
    delete *it;
  }
}

NodeTrees::~NodeTrees() {
  // delete all root nodes
  for (auto it = thread_event_trees_map_.begin();
       it != thread_event_trees_map_.end(); ++it) {
    delete it->second;
  }
}

void NodeTrees::BuildTrees(
    const std::vector<HostTraceEventNode*>& host_event_nodes,
    std::vector<CudaRuntimeTraceEventNode*>& runtime_event_nodes,
    const std::vector<DeviceTraceEventNode*>& device_event_nodes) {
  // seperate Host Event Nodes into different threads
  std::map<uint64_t, std::vector<HostTraceEventNode*>>
      thread2host_event_nodes;  // used to store HostTraceEventNodes per thread
  std::map<uint64_t, std::vector<CudaRuntimeTraceEventNode*>>
      thread2runtime_event_nodes;  // used to store CudaRuntimeTraceEventNode
                                   // per
                                   // thread
  std::map<uint32_t, CudaRuntimeTraceEventNode*>
      correlation_id2runtime_event_node;  // used to store the relation between
                                          // correlation id and runtime node
  // construct thread2host_event_nodes
  for (auto it = host_event_nodes.begin(); it != host_event_nodes.end(); ++it) {
    thread2host_event_nodes[(*it)->ThreadId()].push_back(*it);
  }
  // construct thread2runtime_event_nodes and
  // correlation_id2runtime_event_node
  for (auto it = runtime_event_nodes.begin(); it != runtime_event_nodes.end();
       ++it) {
    thread2runtime_event_nodes[(*it)->ThreadId()].push_back(*it);
    correlation_id2runtime_event_node[(*it)->CorrelationId()] = *it;
  }
  // associate CudaRuntimeTraceEventNode and DeviceTraceEventNode
  // construct correlation_id2device_event_nodes
  for (auto it = device_event_nodes.begin(); it != device_event_nodes.end();
       ++it) {
    auto dst_iter =
        correlation_id2runtime_event_node.find((*it)->CorrelationId());
    PADDLE_ENFORCE_NE(
        dst_iter, correlation_id2runtime_event_node.end(),
        platform::errors::NotFound("Unknown device events, "
                                   "no corresponding cuda runtime events"));
    dst_iter->second->AddDeviceTraceEventNode(*it);
  }
  // sort host event nodes and runtime event nodes according to start_ns and
  // end_ns
  // the smaller start_ns is, the further ahead position is.
  // when start_ns of two nodes are equal, the one with bigger end_ns should be
  // ahead.
  for (auto it = thread2host_event_nodes.begin();
       it != thread2host_event_nodes.end(); ++it) {
    std::sort(it->second.begin(), it->second.end(),
              [](HostTraceEventNode* node1, HostTraceEventNode* node2) {
                if (node1->StartNs() < node2->StartNs()) {
                  return true;
                }
                if ((node1->StartNs() == node2->StartNs()) &&
                    (node1->EndNs() > node2->EndNs())) {
                  return true;
                }
                return false;
              });
  }
  for (auto it = thread2runtime_event_nodes.begin();
       it != thread2runtime_event_nodes.end(); ++it) {
    std::sort(
        it->second.begin(), it->second.end(),
        [](CudaRuntimeTraceEventNode* node1, CudaRuntimeTraceEventNode* node2) {
          if (node1->StartNs() < node2->StartNs()) {
            return true;
          }
          if ((node1->StartNs() == node2->StartNs()) &&
              (node1->EndNs() > node2->EndNs())) {
            return true;
          }
          return false;
        });
  }

  // construct trees
  std::set<uint64_t> thread_set;
  for (auto it = thread2host_event_nodes.begin();
       it != thread2host_event_nodes.end(); ++it) {
    thread_set.insert(it->first);
  }

  for (auto it = thread2runtime_event_nodes.begin();
       it != thread2runtime_event_nodes.end(); ++it) {
    thread_set.insert(it->first);
  }

  for (auto it = thread_set.begin(); it != thread_set.end(); ++it) {
    thread_event_trees_map_[*it] = BuildTreeRelationship(
        thread2host_event_nodes[*it], thread2runtime_event_nodes[*it]);
  }
}

HostTraceEventNode* NodeTrees::BuildTreeRelationship(
    std::vector<HostTraceEventNode*> host_event_nodes,
    std::vector<CudaRuntimeTraceEventNode*> runtime_event_nodes) {
  // a stack used for analyse relationship
  auto node_stack = std::vector<HostTraceEventNode*>();
  // root node, top level
  auto root_node = new HostTraceEventNode(
      HostTraceEvent(std::string("root node"), TracerEventType::UserDefined, 0,
                     ULLONG_MAX, 0, 0));
  // push root node into node_stack
  node_stack.push_back(root_node);
  // handle host_event_nodes
  for (auto it = host_event_nodes.begin(); it != host_event_nodes.end(); ++it) {
    while (true) {
      auto stack_top_node = node_stack.back();
      if ((*it)->StartNs() < stack_top_node->EndNs()) {
        // current node is the child of stack_top_node
        PADDLE_ENFORCE_LE(
            (*it)->EndNs(), stack_top_node->EndNs(),
            platform::errors::Fatal(
                "should not have time range intersection within one thread"));
        stack_top_node->AddChild(*it);
        node_stack.push_back(*it);
        break;
      } else {
        node_stack.pop_back();
        // insert runtime node
        // select runtime nodes which time range within stack_top_node
        std::vector<CudaRuntimeTraceEventNode*>::iterator firstposition;
        std::vector<CudaRuntimeTraceEventNode*>::iterator lastposition =
            runtime_event_nodes.end();
        bool hasenter = false;
        for (auto runtimenode = runtime_event_nodes.begin();
             runtimenode != runtime_event_nodes.end(); ++runtimenode) {
          if (((*runtimenode)->StartNs() >= stack_top_node->StartNs()) &&
              ((*runtimenode)->EndNs() <= stack_top_node->EndNs())) {
            if (!hasenter) {
              firstposition = runtimenode;
              hasenter = true;
            }
            stack_top_node->AddCudaRuntimeNode(*runtimenode);
          } else {
            // from this runtime node, not within stack_top_node, erase the
            // nodes from runtime_event_nodes
            if ((*runtimenode)->StartNs() > stack_top_node->EndNs()) {
              lastposition = runtimenode;
              break;
            }
          }
        }
        if (hasenter) {
          runtime_event_nodes.erase(firstposition, lastposition);
        }
      }
    }
  }
  // to insert left runtimenode into host_event_nodes
  while (!node_stack.empty()) {
    auto stack_top_node = node_stack.back();
    // insert runtime node
    // select runtime nodes which time range within stack_top_node
    std::vector<CudaRuntimeTraceEventNode*>::iterator firstposition;
    std::vector<CudaRuntimeTraceEventNode*>::iterator lastposition =
        runtime_event_nodes.end();
    bool hasenter = false;
    for (auto runtimenode = runtime_event_nodes.begin();
         runtimenode != runtime_event_nodes.end(); ++runtimenode) {
      if (((*runtimenode)->StartNs() >= stack_top_node->StartNs()) &&
          ((*runtimenode)->EndNs() <= stack_top_node->EndNs())) {
        if (!hasenter) {
          firstposition = runtimenode;
          hasenter = true;
        }
        stack_top_node->AddCudaRuntimeNode(*runtimenode);
      } else {
        // from this runtime node, not within stack_top_node, erase the
        // nodes from runtime_event_nodes
        if ((*runtimenode)->StartNs() > stack_top_node->EndNs()) {
          lastposition = runtimenode;
          break;
        }
      }
    }
    if (hasenter) {
      runtime_event_nodes.erase(firstposition, lastposition);
    }
    node_stack.pop_back();
  }
  return root_node;
}

std::map<uint64_t, std::vector<HostTraceEventNode*>> NodeTrees::Traverse(
    bool bfs) const {
  // traverse the tree, provide two methods: bfs(breadth first search) or
  // dfs(depth first search)
  std::map<uint64_t, std::vector<HostTraceEventNode*>> thread2host_event_nodes;
  if (bfs == true) {
    for (auto it = thread_event_trees_map_.begin();
         it != thread_event_trees_map_.end(); ++it) {
      auto deque = std::deque<HostTraceEventNode*>();
      uint64_t thread_id = it->first;
      auto root_node = it->second;
      deque.push_back(root_node);
      while (!deque.empty()) {
        auto current_node = deque.front();
        deque.pop_front();
        thread2host_event_nodes[thread_id].push_back(current_node);
        for (auto child = current_node->GetChildren().begin();
             child != current_node->GetChildren().end(); ++child) {
          deque.push_back(*child);
        }
      }
    }

  } else {
    for (auto it = thread_event_trees_map_.begin();
         it != thread_event_trees_map_.end(); ++it) {
      auto stack = std::stack<HostTraceEventNode*>();
      uint64_t thread_id = it->first;
      auto root_node = it->second;
      stack.push(root_node);
      while (!stack.empty()) {
        auto current_node = stack.top();
        stack.pop();
        thread2host_event_nodes[thread_id].push_back(current_node);
        for (auto child = current_node->GetChildren().begin();
             child != current_node->GetChildren().end(); ++child) {
          stack.push(*child);
        }
      }
    }
  }
  return thread2host_event_nodes;
}

void NodeTrees::LogMe(BaseLogger* logger) { logger->LogNodeTrees(*this); }

void NodeTrees::HandleTrees(
    std::function<void(HostTraceEventNode*)> host_event_node_handle,
    std::function<void(CudaRuntimeTraceEventNode*)> runtime_event_node_handle,
    std::function<void(DeviceTraceEventNode*)> device_event_node_handle) {
  // using different user-defined function to handle different nodes
  const std::map<uint64_t, std::vector<HostTraceEventNode*>>
      thread2host_event_nodes = Traverse(true);
  for (auto it = thread2host_event_nodes.begin();
       it != thread2host_event_nodes.end(); ++it) {
    for (auto hostnode = it->second.begin(); hostnode != it->second.end();
         ++hostnode) {
      if (hostnode != it->second.begin()) {  // skip root node
        host_event_node_handle(*hostnode);
      }
      for (auto runtimenode = (*hostnode)->GetRuntimeTraceEventNodes().begin();
           runtimenode != (*hostnode)->GetRuntimeTraceEventNodes().end();
           ++runtimenode) {
        runtime_event_node_handle(*runtimenode);
        for (auto devicenode =
                 (*runtimenode)->GetDeviceTraceEventNodes().begin();
             devicenode != (*runtimenode)->GetDeviceTraceEventNodes().end();
             ++devicenode) {
          device_event_node_handle(*devicenode);
        }
      }
    }
  }
}
}  // namespace platform
}  // namespace paddle
