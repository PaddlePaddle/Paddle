/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <limits.h>

#include "paddle/fluid/platform/profiler/event_node.h"

namespace paddle{
namespace platform {

void NodeTrees::BuildTrees(const std::vector<HostRecordNode*>& host_record_nodes, 
                           const std::vector<CudaRuntimeRecordNode*>& runtime_record_nodes,
                           const std::vector<DeviceRecordNode*>& device_record_nodes){
  // seperate Host Record Nodes into different threads
  std::map<uint64_t, vector<HostRecordNode*>> thread2host_record_nodes; //used to store HostRecordNodes per thread
  std::map<uint64_t, vector<CudaRuntimeRecordNode*>> thread2runtime_record_nodes; //used to store CudaRuntimeRecordNode per thread
  std::map<uint32_t, CudaRuntimeRecordNode*> correlation_id2runtime_record_node; // used to store the relation between correlation id and runtime node
  // construct thread2host_record_nodes
  for(auto it = host_record_nodes.begin(); it != host_record_nodes.end(); ++it){
      thread2host_record_nodes[(*it)->thread_id()].push_back(*it);
  }
  // construct thread2runtime_record_nodes and correlation_id2runtime_record_node
  for(auto it = runtime_record_nodes.begin(); it != runtime_record_nodes.end(); ++it){
      thread2runtime_record_nodes[(*it)->thread_id()].push_back(*it);
      correlation_id2runtime_record_node[(*it)->correlation_id()] = *it;
  }
  // associate CudaRuntimeRecordNode and DeviceRecordNode
  // construct correlation_id2device_record_nodes
  for(auto it = device_record_nodes.begin(); it != device_record_nodes.end(); ++it){
      auto dst_iter = correlation_id2runtime_record_node.find((*it)->correlation_id());
      PADDLE_ENFORCE_NE(dst_iter, correlation_id2runtime_record_node.end(), 
                        platform::errors::NotFound("Unknown device records, \
                                            no corresponding cuda runtime records")
                       );    
      dst_iter->second->AddDeviceRecordNode(*it);
  }
  // sort host record nodes and runtime record nodes according to start_ns and end_ns
  // the smaller start_ns is, the further ahead position is. 
  // when start_ns of two nodes are equal, the one with bigger end_ns should be ahead.
  for(auto it = thread2host_record_nodes.begin(); it != thread2host_record_nodes.end(); ++it){
      std::sort(it->second.begin(), it->second.end(), [](HostRecordNode*node1, HostRecordNode*node2){
          if(node1->start_ns() < node2->start_ns()){
              return true;
          }
          if((node1->start_ns()==node2->start_ns())&&(node1->end_ns()>node2->end_ns())){
              return true;
          }
          return false;

      });
  }
  for(auto it = thread2runtime_record_nodes.begin(); it != thread2runtime_record_nodes.end(); ++it){
      std::sort(it->second.begin(), it->second.end(), [](CudaRuntimeRecordNode*node1, CudaRuntimeRecordNode*node2){
          if(node1->start_ns() < node2->start_ns()){
              return true;
          }
          if((node1->start_ns()==node2->start_ns())&&(node1->end_ns()>node2->end_ns())){
              return true;
          }
          return false;

      });
  }

  // construct trees
  for(auto it = thread2host_record_nodes.begin(); it != thread2host_record_nodes.end(); ++it){
     thread_record_trees_map_[it->first] = BuildTreeRelationship(it->second, thread2runtime_record_nodes[it->first]);
  }
}

HostRecordNode* NodeTrees::BuildTreeRelationship(std::vector<HostRecordNode*>& host_record_nodes, 
                                      std::vector<CudaRuntimeRecordNode*>& runtime_record_nodes){
  // a stack used for analyse relationship
  auto node_stack = std::vector<HostRecordNode*>;
  // root node, top level 
  auto root_node = new HostRecordNode(std::string("root node"), TracerEventType::Operator, LLONG_MIN, LLONG_MAX);
  // push root node into node_stack
  node_stack.push_back(root_node);
  // handle host_record_nodes
  for(auto it = host_record_nodes.begin(); it != host_record_nodes.end(); ++it){
      while(true){
          auto stack_top_node = node_stack.back();

          if(*it->start_ns() < stack_top_node->end_ns()){
              // current node is the child of stack_top_node
              PADDLE_ENFORCE_LE(*it->end_ns(), stack_top_node->end_ns(), 
                                platform::errors::Fatal("should not have time range intersection within one thread")
                               ); 
              stack_top_node->AddChild(*it);
              node_stack.push_back(*it);
              break;
          }else{
              node_stack.pop_back();
              // insert runtime node
              // judge if time range of runtime node within stack_top_node
              for(auto runtimenode = runtime_record_nodes.begin(); runtimenode != runtime_record_nodes.end(); ++runtimenode){
                  if((runtimenode->start_ns() >= stack_top_node->start_ns()) && (runtimenode->end_ns() <= stack_top_node->end_ns())){
                     stack_top_node->AddCudaRuntimeNode(runtimenode);                    
                  }else{
                      // from this runtime node, not within stack_top_node, erase the nodes within stack_top_node from runtime_record_nodes
                      runtime_record_nodes.erase(runtime_record_nodes.begin(), runtimenode);
                      break;
                  }
              }
          }
      }
  }
  return root_node;

}

} // namespace platform
} // namespace paddle