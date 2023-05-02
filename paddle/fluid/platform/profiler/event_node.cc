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

#include "paddle/fluid/platform/profiler/utils.h"

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
       it != thread_event_trees_map_.end();
       ++it) {
    delete it->second;
  }
}

void NodeTrees::BuildTrees(
    const std::vector<HostTraceEventNode*>& host_event_nodes,
    const std::vector<CudaRuntimeTraceEventNode*>& runtime_event_nodes,
    const std::vector<DeviceTraceEventNode*>& device_event_nodes,
    const std::vector<MemTraceEventNode*>& mem_event_nodes,
    const std::vector<OperatorSupplementEventNode*>& op_supplement_events) {
  // separate Host Event Nodes into different threads
  std::map<uint64_t, std::vector<HostTraceEventNode*>>
      thread2host_event_nodes;  // used to store HostTraceEventNodes per thread
  std::map<uint64_t, std::vector<CudaRuntimeTraceEventNode*>>
      thread2runtime_event_nodes;  // used to store CudaRuntimeTraceEventNode
                                   // per
                                   // thread
  std::map<uint64_t, std::vector<MemTraceEventNode*>>
      thread2mem_event_nodes;  // used to store MemTraceEventNode
                               // per
                               // thread
  std::map<uint64_t, std::vector<OperatorSupplementEventNode*>>
      thread2op_supplement_event_nodes;  // used to store
                                         // OperatorSupplementEventNode
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
    if (dst_iter == correlation_id2runtime_event_node.end()) {
      continue;
    }
    dst_iter->second->AddDeviceTraceEventNode(*it);
  }
  // construct thread2mem_event_nodes
  for (auto it = mem_event_nodes.begin(); it != mem_event_nodes.end(); ++it) {
    thread2mem_event_nodes[(*it)->ThreadId()].push_back(*it);
  }
  // construct thread2op_supplement_event_nodes
  for (auto it = op_supplement_events.begin(); it != op_supplement_events.end();
       ++it) {
    thread2op_supplement_event_nodes[(*it)->ThreadId()].push_back(*it);
  }
  // sort host event nodes and runtime event nodes according to start_ns and
  // end_ns
  // the smaller start_ns is, the further ahead position is.
  // when start_ns of two nodes are equal, the one with bigger end_ns should be
  // ahead.
  for (auto it = thread2host_event_nodes.begin();
       it != thread2host_event_nodes.end();
       ++it) {
    std::sort(it->second.begin(),
              it->second.end(),
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
       it != thread2runtime_event_nodes.end();
       ++it) {
    std::sort(
        it->second.begin(),
        it->second.end(),
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
  // sort mem event nodes and operator supplement event nodes
  for (auto it = thread2mem_event_nodes.begin();
       it != thread2mem_event_nodes.end();
       ++it) {
    std::sort(it->second.begin(),
              it->second.end(),
              [](MemTraceEventNode* node1, MemTraceEventNode* node2) {
                if (node1->TimeStampNs() <= node2->TimeStampNs()) {
                  return true;
                }
                return false;
              });
  }

  for (auto it = thread2op_supplement_event_nodes.begin();
       it != thread2op_supplement_event_nodes.end();
       ++it) {
    std::sort(it->second.begin(),
              it->second.end(),
              [](OperatorSupplementEventNode* node1,
                 OperatorSupplementEventNode* node2) {
                if (node1->TimeStampNs() <= node2->TimeStampNs()) {
                  return true;
                }
                return false;
              });
  }

  // construct trees
  std::set<uint64_t> thread_set;
  for (auto it = thread2host_event_nodes.begin();
       it != thread2host_event_nodes.end();
       ++it) {
    thread_set.insert(it->first);
  }

  for (auto it = thread2runtime_event_nodes.begin();
       it != thread2runtime_event_nodes.end();
       ++it) {
    thread_set.insert(it->first);
  }
  for (auto it = thread2mem_event_nodes.begin();
       it != thread2mem_event_nodes.end();
       ++it) {
    thread_set.insert(it->first);
  }
  for (auto it = thread2op_supplement_event_nodes.begin();
       it != thread2op_supplement_event_nodes.end();
       ++it) {
    thread_set.insert(it->first);
  }

  for (auto it = thread_set.begin(); it != thread_set.end(); ++it) {
    thread_event_trees_map_[*it] =
        BuildTreeRelationship(thread2host_event_nodes[*it],
                              thread2runtime_event_nodes[*it],
                              thread2mem_event_nodes[*it],
                              thread2op_supplement_event_nodes[*it]);
  }
}

HostTraceEventNode* NodeTrees::BuildTreeRelationship(
    std::vector<HostTraceEventNode*> host_event_nodes,
    std::vector<CudaRuntimeTraceEventNode*> runtime_event_nodes,
    std::vector<MemTraceEventNode*> mem_event_nodes,
    std::vector<OperatorSupplementEventNode*> op_supplement_events) {
  // a stack used for analyse relationship
  auto node_stack = std::vector<HostTraceEventNode*>();
  // root node, top level
  auto root_node =
      new HostTraceEventNode(HostTraceEvent(std::string("root node"),
                                            TracerEventType::UserDefined,
                                            0,
                                            ULLONG_MAX,
                                            0,
                                            0));
  // push root node into node_stack
  node_stack.push_back(root_node);
  // handle host_event_nodes
  for (auto it = host_event_nodes.begin(); it != host_event_nodes.end(); ++it) {
    while (true) {
      auto stack_top_node = node_stack.back();
      if ((*it)->StartNs() < stack_top_node->EndNs()) {
        // current node is the child of stack_top_node
        PADDLE_ENFORCE_LE(
            (*it)->EndNs(),
            stack_top_node->EndNs(),
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
             runtimenode != runtime_event_nodes.end();
             ++runtimenode) {
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
         runtimenode != runtime_event_nodes.end();
         ++runtimenode) {
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

  // build relationship between host event node and mem event node
  // First, post-order traverse the tree. Then, insert the memory and op
  // supplement node into correct host nodes.
  auto stack = std::stack<HostTraceEventNode*>();
  auto flag_stack = std::stack<int32_t>();
  auto post_order_nodes = std::vector<HostTraceEventNode*>();
  stack.push(root_node);
  flag_stack.push(0);
  while (!stack.empty()) {
    auto current_node = stack.top();
    stack.pop();
    auto flag = flag_stack.top();
    flag_stack.pop();
    if (flag == 0) {
      stack.push(current_node);
      flag_stack.push(1);
      for (auto child = current_node->GetChildren().rbegin();
           child != current_node->GetChildren().rend();
           ++child) {
        stack.push(*child);
        flag_stack.push(0);
      }
    } else {
      post_order_nodes.push_back(current_node);
    }
  }

  for (auto it = post_order_nodes.begin(); it < post_order_nodes.end(); ++it) {
    bool hasenter = false;
    std::vector<MemTraceEventNode*>::iterator firstposition;
    std::vector<MemTraceEventNode*>::iterator lastposition =
        mem_event_nodes.end();
    for (auto mem_it = mem_event_nodes.begin(); mem_it < mem_event_nodes.end();
         ++mem_it) {
      if ((*mem_it)->TimeStampNs() >= (*it)->StartNs() &&
          (*mem_it)->TimeStampNs() <= (*it)->EndNs()) {
        (*it)->AddMemNode(*mem_it);
        if (!hasenter) {
          firstposition = mem_it;
          hasenter = true;
        }
      } else {
        if ((*mem_it)->TimeStampNs() > (*it)->EndNs()) {
          lastposition = mem_it;
          break;
        }
      }
    }
    if (hasenter) {
      mem_event_nodes.erase(firstposition, lastposition);
    }
  }

  // build relationship between host event node and op supplement node
  for (auto it = post_order_nodes.begin(); it < post_order_nodes.end(); ++it) {
    int op_supplement_count = 0;
    bool hasenter = false;
    std::vector<OperatorSupplementEventNode*>::iterator firstposition;
    std::vector<OperatorSupplementEventNode*>::iterator lastposition =
        op_supplement_events.end();
    for (auto op_supplement_it = op_supplement_events.begin();
         op_supplement_it < op_supplement_events.end();
         ++op_supplement_it) {
      if ((*op_supplement_it)->TimeStampNs() >= (*it)->StartNs() &&
          (*op_supplement_it)->TimeStampNs() <= (*it)->EndNs()) {
        if (!hasenter) {
          firstposition = op_supplement_it;
          hasenter = true;
        }
        (*it)->SetOperatorSupplementNode(*op_supplement_it);
        op_supplement_count += 1;
      } else {
        if ((*op_supplement_it)->TimeStampNs() > (*it)->EndNs()) {
          lastposition = op_supplement_it;
          break;
        }
      }
    }
    if (hasenter) {
      op_supplement_events.erase(firstposition, lastposition);
    }
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
         it != thread_event_trees_map_.end();
         ++it) {
      auto deque = std::deque<HostTraceEventNode*>();
      uint64_t thread_id = it->first;
      auto root_node = it->second;
      deque.push_back(root_node);
      while (!deque.empty()) {
        auto current_node = deque.front();
        deque.pop_front();
        thread2host_event_nodes[thread_id].push_back(current_node);
        for (auto child = current_node->GetChildren().begin();
             child != current_node->GetChildren().end();
             ++child) {
          deque.push_back(*child);
        }
      }
    }

  } else {
    for (auto it = thread_event_trees_map_.begin();
         it != thread_event_trees_map_.end();
         ++it) {
      auto stack = std::stack<HostTraceEventNode*>();
      uint64_t thread_id = it->first;
      auto root_node = it->second;
      stack.push(root_node);
      while (!stack.empty()) {
        auto current_node = stack.top();
        stack.pop();
        thread2host_event_nodes[thread_id].push_back(current_node);
        for (auto child = current_node->GetChildren().rbegin();
             child != current_node->GetChildren().rend();
             ++child) {
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
    std::function<void(DeviceTraceEventNode*)> device_event_node_handle,
    std::function<void(MemTraceEventNode*)> mem_event_node_handle,
    std::function<void(OperatorSupplementEventNode*)>
        op_supplement_node_handle) {
  // using different user-defined function to handle different nodes
  const std::map<uint64_t, std::vector<HostTraceEventNode*>>
      thread2host_event_nodes = Traverse(true);
  for (auto it = thread2host_event_nodes.begin();
       it != thread2host_event_nodes.end();
       ++it) {
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
      for (auto memeventnode = (*hostnode)->GetMemTraceEventNodes().begin();
           memeventnode != (*hostnode)->GetMemTraceEventNodes().end();
           ++memeventnode) {
        mem_event_node_handle(*memeventnode);
      }
      if ((*hostnode)->GetOperatorSupplementEventNode()) {
        op_supplement_node_handle(
            (*hostnode)->GetOperatorSupplementEventNode());
      }
    }
  }
}
}  // namespace platform
}  // namespace paddle
