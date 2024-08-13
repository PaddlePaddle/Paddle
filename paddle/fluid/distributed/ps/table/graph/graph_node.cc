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

#include "paddle/fluid/distributed/ps/table/graph/graph_node.h"

#include <cstring>
namespace paddle::distributed {

GraphNode::~GraphNode() {
  if (sampler != nullptr) {
    delete sampler;
    sampler = nullptr;
  }
  if (edges != nullptr) {
    delete edges;
    edges = nullptr;
  }
}

int Node::weight_size = sizeof(float);
int Node::id_size = sizeof(uint64_t);
int Node::int_size = sizeof(int);

int Node::get_size(bool need_feature) { return id_size + int_size; }

void Node::to_buffer(char* buffer, bool need_feature) {
  memcpy(buffer, &id, id_size);
  buffer += id_size;

  int feat_num = 0;
  memcpy(buffer, &feat_num, sizeof(int));
}

void Node::recover_from_buffer(char* buffer) { memcpy(&id, buffer, id_size); }

int FeatureNode::get_size(bool need_feature) {
  int size = id_size + int_size;  // id, feat_num
  if (need_feature) {
    size += feature.size() * int_size;
    for (const std::string& fea : feature) {
      size += fea.size();
    }
  }
  return size;
}

void GraphNode::build_edges(bool is_weighted) {
  if (edges == nullptr) {
    if (is_weighted == true) {
      edges = new WeightedGraphEdgeBlob();
    } else {
      edges = new GraphEdgeBlob();
    }
  }
}
void GraphNode::build_sampler(std::string sample_type) {
  if (sampler != nullptr) {
    return;
  }
  if (sample_type == "random") {
    sampler = new RandomSampler();
  } else if (sample_type == "weighted") {
    sampler = new WeightedSampler();
  }
  if (sampler != nullptr) {
    sampler->build(edges);
  } else {
    throw std::runtime_error("Failed to create a sampler of type: " +
                             sample_type);
  }
}
void FeatureNode::to_buffer(char* buffer, bool need_feature) {
  memcpy(buffer, &id, id_size);
  buffer += id_size;

  int feat_num = 0;
  int feat_len;
  if (need_feature) {
    feat_num += feature.size();
    memcpy(buffer, &feat_num, sizeof(int));
    buffer += sizeof(int);
    for (int i = 0; i < feat_num; ++i) {
      feat_len = feature[i].size();
      memcpy(buffer, &feat_len, sizeof(int));
      buffer += sizeof(int);
      memcpy(buffer, feature[i].c_str(), feature[i].size());
      buffer += feature[i].size();
    }
  } else {
    memcpy(buffer, &feat_num, sizeof(int));
  }
}
void FeatureNode::recover_from_buffer(char* buffer) {
  int feat_num, feat_len;
  memcpy(&id, buffer, id_size);
  buffer += id_size;

  memcpy(&feat_num, buffer, sizeof(int));
  buffer += sizeof(int);

  feature.clear();
  for (int i = 0; i < feat_num; ++i) {
    memcpy(&feat_len, buffer, sizeof(int));
    buffer += sizeof(int);

    char str[feat_len + 1];  // NOLINT
    memcpy(str, buffer, feat_len);
    buffer += feat_len;
    str[feat_len] = '\0';
    feature.push_back(str);  // NOLINT
  }
}
}  // namespace paddle::distributed
