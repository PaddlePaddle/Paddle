// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/backward.h"

#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/grad_node_info.h"

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"

#include "paddle/top/core/dense_tensor.h"

#include "glog/logging.h"

#include <queue>

namespace egr {

std::unordered_map<GradNodeBase*, int> getInDegreeMap(const std::queue<GradNodeBase*>& init_queue) {
    // Calculate in_degree for each node
    // We can completely remove this pass, if in_degree were set during forward pass
    std::unordered_map<GradNodeBase*, int> node_in_degree_map;

    // Copy nodes
    std::queue<GradNodeBase*> queue = init_queue;
    std::unordered_set<GradNodeBase*> visited;

    // Visit each node exactly once in any order
    while(!queue.empty()) {
        GradNodeBase* node = queue.front();
        queue.pop();

        if(visited.count(node))
            continue;
        visited.insert(node);
       
        // Find and append next nodes
        const std::vector<Edge>& edges = node->GetEdges();
        for(const Edge& edge : edges) {
            GradNodeBase* next_node = edge.GetMutableGradNode().get();
            // Update in_degree
            if(!node_in_degree_map.count(next_node))
                node_in_degree_map[next_node] = 0;
            node_in_degree_map[next_node]++;
            queue.push(next_node);
        }
    }

    return node_in_degree_map;
}

void RunBackward(std::vector<pt::Tensor>& tensors,
                 const std::vector<pt::Tensor>& grad_tensors,
                 bool retain_graph) {
    
    // *Gradient Hook should happen at node-level
    // *Inplace version check should perform at node-level
    // *Cross-batch accumulation happens at forward pass
    
    /* --- Initialization --- */
    // 1. Init queue with starting nodes
    // 2. Prepare initial input buffers
    std::queue<GradNodeBase*> queue;
    std::unordered_map<GradNodeBase*, std::unique_ptr<InputBuffer>> node_input_buffers_dict;
    for(size_t i = 0; i < tensors.size(); i++) {
        pt::Tensor& tensor = tensors[i];
        
        AutogradMeta* auto_grad_meta = EagerUtils::autograd_meta(tensor);

        PADDLE_ENFORCE(auto_grad_meta, 
                paddle::platform::errors::Fatal("Detected NULL auto_grad_meta during backward execution"));

        GradNodeBase* grad_node = auto_grad_meta->GetMutableGradNode().get();
        
        // Prepare InputBuffer
        if(!node_input_buffers_dict.count(grad_node)) {
            node_input_buffers_dict[grad_node] = std::make_unique<InputBuffer>(grad_node->InputNum());
        }
        
        if(grad_tensors.size() > 0) {
            PADDLE_ENFORCE(grad_tensors.size() == tensors.size(), 
                paddle::platform::errors::Fatal("grad_tensors should either have size = 0 or same size as tensors"));
            
            node_input_buffers_dict[grad_node]->add(auto_grad_meta->OutRank(), grad_tensors[i]);

        } else {
            // Initialize tensor with 1.0
            // Forward Tensor "tensor" is passed to indicate tensortype, datatype and dims
            // InputBuffer will initialize another tensor with same tensortype, datatype and dims but filled with 1.0
            node_input_buffers_dict[grad_node]->add(auto_grad_meta->OutRank(), tensor, true /*fill_one=true*/);
        }

        // Prepare queue
        queue.push(grad_node);
    }
    
    // 3. Compute in_degree for each node
    std::unordered_map<GradNodeBase*, int> node_in_degree_map = getInDegreeMap(queue);

    /* --- Topological Visit --- */
    // 1. Pop queue
    // 2. Run node
    //    |- node(grads)
    //    |- Prepare for next node
    // 3. Update queue
    while(!queue.empty()) {
        GradNodeBase* node = queue.front();
        queue.pop();

        // Run node: This is where Hook happens
        PADDLE_ENFORCE(node_input_buffers_dict.count(node), 
            paddle::platform::errors::Fatal("Trying to run Node without configuring its InputBuffer"));

        std::unique_ptr<InputBuffer> node_input_buffer = std::move(node_input_buffers_dict[node]);
        std::vector<pt::Tensor> grad_output_tensors = (*node)(node_input_buffer->Buffers());
        
        node_input_buffers_dict.erase(node);

        // Prepare InputBuffer for next node
        const std::vector<Edge>& edges = node->GetEdges();
        PADDLE_ENFORCE(edges.size() == grad_output_tensors.size() || edges.size() == 0, 
            paddle::platform::errors::Fatal("Number of edges should be either zero or the same as number of output grad tensors"));
        
        for(size_t i = 0; i < edges.size(); i++) {
            const Edge& edge = edges[i];
            pt::Tensor& grad_output_tensor = grad_output_tensors[i];
            
            GradNodeBase* next_node = edge.GetMutableGradNode().get();

            if(!node_input_buffers_dict.count(next_node))
                node_input_buffers_dict[next_node] = std::make_unique<InputBuffer>(next_node->InputNum());
            node_input_buffers_dict[next_node]->add(edge.GetInputRank(), grad_output_tensor);

            // Update queue
            node_in_degree_map[next_node]--;
            PADDLE_ENFORCE(node_in_degree_map[next_node] >= 0, 
                paddle::platform::errors::Fatal("Node's in-degree cannot be negative"));
            if(node_in_degree_map[next_node] == 0)
                queue.emplace(std::move(next_node));
        }
    }
}

} // namespace egr
