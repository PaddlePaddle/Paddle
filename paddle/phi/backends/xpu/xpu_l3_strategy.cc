/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/backends/xpu/xpu_l3_strategy.h"
#include "glog/logging.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"

namespace phi {

void XPUL3CacheBlock::Set(void* addr, size_t size) {
  if (addr == nullptr || size == 0) {
    PADDLE_THROW(
        common::errors::InvalidArgument("Set XPUL3CacheBlock Size as Zero"));
  }
  addr_ = addr;
  size_ = size;
}

// return true means success, false means Autotune L3 fail
bool XPUL3Planner::RunAutotune(
    const std::vector<XPUL3CacheBlock*>& l3_block_dict, size_t l3_size) {
  if (l3_block_dict.size() == 0 || l3_size <= 0 || !plan_.empty()) {
    return false;
  }
  VLOG(3) << "AutoTune XPU L3 Cache Block Start.";
  struct node {
    size_t weights = 0;
    size_t scores = 0;
    std::vector<size_t> choices{0};
  };
  std::vector<std::vector<node>> records;
  std::vector<size_t> record_map;
  size_t total_scores = 0;
  for (size_t block_idx = 0; block_idx < l3_block_dict.size(); block_idx++) {
    XPUL3CacheBlock* cur_block = l3_block_dict[block_idx];
    std::vector<size_t>& history = cur_block->history_;
    auto history_size = history.size();
    size_t score = 0;
    VLOG(3) << "Block Idx is " << block_idx;
    if (history_size > 1) {
      std::vector<node> block_nodes{node()};
      std::sort(history.begin(), history.end());
      for (size_t i = 0; i < history_size; i++) {
        VLOG(3) << "Size History : " << i << " is " << history[i];
        if (history[i] > l3_size) {
          break;
        }
        score += history[i];
        if (i == history_size - 1 || history[i + 1] != history[i]) {
          node cur_node;
          cur_node.weights = history[i];
          cur_node.choices = {history[i]};
          cur_node.scores = score;
          block_nodes.push_back(cur_node);
          VLOG(3) << "Node Weights is:" << cur_node.weights
                  << ", Node Scores is: " << score;
        }
      }
      total_scores += score;
      records.push_back(block_nodes);
      record_map.push_back(block_idx);
    }
  }
  if (records.size() <= 0) {
    VLOG(3) << "No blocks to reuse!";
    return false;
  }
  std::vector<node> res(records[0]);
  for (size_t block_idx = 1; block_idx < records.size(); block_idx++) {
    std::vector<node> new_nodes;
    for (size_t node_idx = 0; node_idx < records[block_idx].size();
         node_idx++) {
      for (size_t res_idx = 0; res_idx < res.size(); res_idx++) {
        node cur_node;
        size_t cur_weights =
            records[block_idx][node_idx].weights + res[res_idx].weights;
        if (cur_weights > l3_size) {
          break;
        }
        cur_node.scores =
            records[block_idx][node_idx].scores + res[res_idx].scores;
        cur_node.weights = cur_weights;
        cur_node.choices = res[res_idx].choices;
        cur_node.choices.push_back(records[block_idx][node_idx].choices[0]);
        new_nodes.push_back(cur_node);
      }
    }
    struct {
      bool operator()(node a, node b) const {
        if (a.weights < b.weights) {
          return true;
        } else if (a.weights == b.weights) {
          return a.scores > b.scores;
        } else {
          return false;
        }
      }
    } customLess;

    std::sort(new_nodes.begin(), new_nodes.end(), customLess);
    std::vector<bool> stay(new_nodes.size(), true);
    for (int i = new_nodes.size() - 1; i >= 0; i--) {
      for (int j = i - 1; j >= 0; j--) {
        if (new_nodes[j].scores >= new_nodes[i].scores) {
          stay[i] = false;
          break;
        }
      }
    }
    res.clear();
    for (size_t i = 0; i < new_nodes.size(); i++) {
      if (stay[i] == true) {
        res.push_back(new_nodes[i]);
      }
    }
    VLOG(3) << "XPU L3 Block IDX is " << block_idx
            << ", Choices before filter are " << new_nodes.size()
            << ", Choices after filter are " << res.size();
  }
  // final result: res.back().choices
  //               std::vector<size_t> record_map;
  for (size_t i = 0; i < res.back().choices.size(); i++) {
    VLOG(3) << "BLOCK IDX is " << i << ", Acquired L3 Size is "
            << res.back().choices[i];
  }
  double l3_global_ratio = static_cast<double>(res.back().scores) /
                           static_cast<double>(total_scores);
  VLOG(3) << "Tensor Space in L3 / Tensor Space in Global :"
          << l3_global_ratio * 100 << " %";

  size_t block_l3_size =
      std::accumulate(res.back().choices.begin(), res.back().choices.end(), 0);
  size_t xdnn_ctx_l3_size = (l3_size - block_l3_size) / 64 * 64;

  VLOG(3) << "Block L3 Size : " << block_l3_size
          << ", XDNN Ctx L3 Size : " << xdnn_ctx_l3_size;

  plan_.resize(l3_block_dict.size() + 1, 0);
  for (size_t i = 0; i < res.back().choices.size(); i++) {
    plan_[record_map[i]] = res.back().choices[i];
  }
  plan_[l3_block_dict.size()] = xdnn_ctx_l3_size;
  VLOG(3) << "AutoTune XPU L3 Cache Block End.";
  return true;
}

}  // namespace phi
