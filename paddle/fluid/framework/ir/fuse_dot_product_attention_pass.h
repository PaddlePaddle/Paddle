// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

/*
 * Fuse the ElewiseAdd and activation
 */
class Graph;
class Node;

enum class AttentionType { kSelfAttention, kCrossAttention };

class SequenceMetaData {
 public:
  ir::Node *q_actual_seqlen_node;
  ir::Node *kv_actual_seqlen_node;
};
class QKVMetaData {
 public:
  ir::Node *qkv_node = nullptr;
  ir::Node *q_node = nullptr;
  ir::Node *kv_node = nullptr;
};

template <typename T, char const *NAME>
class OpCache {
 public:
  OpCache() {}
  OpCache(const OpCache &) = delete;
  void operator=(const OpCache &) = delete;

  bool Exist(const std::string &key) const {
    std::lock_guard<std::mutex> lock(mtx_);
    return map_.count(key);
  }

  T Get(const std::string &key) const {
    std::lock_guard<std::mutex> lock(mtx_);
    return map_.find(key)->second;
  }

  void Insert(const std::string &key, const T &value) {
    std::lock_guard<std::mutex> lock(mtx_);
    map_[key] = value;
  }

  void Erase(const std::string &key) {
    std::lock_guard<std::mutex> lock(mtx_);
    map_.erase(key);
  }

 private:
  std::unordered_map<std::string, T> map_;
  mutable std::mutex mtx_;
};

const char NAME1[] = "SequenceMetaData";
const char NAME2[] = "SoftmaxOutput";
const char NAME3[] = "QKVMetaData";
using SequenceMetaCache = OpCache<SequenceMetaData, NAME1>;
using SoftmaxOutputCache = OpCache<ir::Node *, NAME2>;
using QKVCache = OpCache<QKVMetaData, NAME3>;

class FuseDotProductAttentionPass : public FusePassBase {
 public:
  virtual ~FuseDotProductAttentionPass() {}

 protected:
  void ApplyImpl(ir::Graph *graph) const override;

  ir::Graph *FuseDotProductAttentionFwd(
      ir::Graph *graph,
      AttentionType attn_type,
      bool with_dropout,
      SequenceMetaCache *seq_meta_cache,
      SoftmaxOutputCache *softmax_output_cache,
      QKVCache *qkv_cache,
      std::unordered_set<const Node *> *nodes_to_remove) const;
  ir::Graph *FuseDotProductAttentionBwd(
      ir::Graph *graph,
      AttentionType attn_type,
      bool with_dropout,
      bool share_attn_mask,
      SequenceMetaCache *seq_meta_cache,
      SoftmaxOutputCache *softmax_output_cache,
      QKVCache *qkv_cache,
      std::unordered_set<const Node *> *nodes_to_remove) const;

 private:
  SequenceMetaData InsertActualSeqlenOp_(ir::Graph *graph,
                                         ir::Node *attn_mask,
                                         BlockDesc *block,
                                         Attribute op_role) const;
  std::string GenerateMetaKey_(const std::string &q_name,
                               const std::string &k_name,
                               const std::string &v_name) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
