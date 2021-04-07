/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <stdio.h>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include "paddle/fluid/distributed/index_dataset/index_wrapper.h"

namespace paddle {
namespace distributed {

std::shared_ptr<IndexWrapper> IndexWrapper::s_instance_(nullptr);

int TreeIndex::load(const std::string filename) {
  FILE* fp = fopen(filename.c_str(), "rb");
  if (fp == NULL) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Can not open file: %s. Please check whether the file exists.",
        filename));
    return -1;
  }

  int num = 0;
  max_id_ = 0;
  size_t ret = fread(&num, sizeof(num), 1, fp);
  while (ret == 1 && num > 0) {
    std::string content(num, '\0');
    size_t read_num = fread(const_cast<char*>(content.data()), 1, num, fp);
    PADDLE_ENFORCE_EQ(
        read_num, static_cast<size_t>(num),
        platform::errors::InvalidArgument(
            "Read from file: %s failed. Valid Format is "
            "an integer representing the length of the following string, "
            "and the string itself.We got an iteger[% d], "
            "but the following string's length is [%d].",
            filename, num, read_num));

    KVItem item;
    PADDLE_ENFORCE_EQ(
        item.ParseFromString(content), true,
        platform::errors::InvalidArgument("Parse from file: %s failed. It's "
                                          "content can't be parsed by KVItem.",
                                          filename));

    if (item.key() == ".tree_meta") {
      meta_.ParseFromString(item.value());
    } else {
      auto code = boost::lexical_cast<uint64_t>(item.key());
      Node node;
      node.ParseFromString(item.value());
      if (node.is_leaf()) {
        id_codes_map_[node.id()] = code;
      }
      data_[code] = node;
      if (node.id() > max_id_) {
        max_id_ = node.id();
      }
    }
    ret = fread(&num, sizeof(num), 1, fp);
  }
  fclose(fp);
  total_nodes_num_ = data_.size();
  return 0;
}

std::vector<uint64_t> TreeIndex::get_nodes_given_level(int level,
                                                       bool ret_code) {
  uint64_t level_num = static_cast<uint64_t>(std::pow(meta_.branch(), level));
  uint64_t level_offset = level_num - 1;

  std::vector<uint64_t> res;
  res.reserve(level_num);
  for (uint64_t i = 0; i < level_num; i++) {
    auto code = level_offset + i;
    if (data_.find(code) != data_.end()) {
      if (ret_code) {
        res.push_back(code);
      } else {
        res.push_back(data_[code].id());
      }
    }
  }

  return res;
}

std::vector<std::vector<uint64_t>> TreeIndex::get_parent_path(
    const std::vector<uint64_t>& ids, int start_level, bool ret_code) {
  std::vector<std::vector<uint64_t>> res(ids.size(), std::vector<uint64_t>());
  for (size_t i = 0; i < ids.size(); i++) {
    auto code = id_codes_map_[ids[i]];
    int level = meta_.height() - 1;

    while (level >= start_level) {
      if (ret_code) {
        res[i].push_back(code);
      } else {
        res[i].push_back(data_[code].id());
      }
      code = (code - 1) / meta_.branch();
      level--;
    }
  }
  return res;
}

std::vector<uint64_t> TreeIndex::get_ancestor_given_level(
    const std::vector<uint64_t>& ids, int level, bool ret_code) {
  std::vector<uint64_t> res;
  res.reserve(ids.size());
  int cur_level;
  for (size_t i = 0; i < ids.size(); i++) {
    if (ids[i] == 0 || id_codes_map_.find(ids[i]) == id_codes_map_.end()) {
      res.push_back(0);
      continue;
    }
    auto code = id_codes_map_[ids[i]];
    cur_level = meta_.height() - 1;
    while (cur_level > level) {
      code = (code - 1) / meta_.branch();
      cur_level--;
    }
    if (ret_code) {
      res.push_back(code);
    } else {
      res.push_back(data_[code].id());
    }
  }
  return res;
}

std::vector<uint64_t> TreeIndex::get_all_items() {
  std::vector<uint64_t> ids;
  ids.reserve(id_codes_map_.size());
  for (auto& ite : id_codes_map_) {
    ids.push_back(ite.first);
  }
  return ids;
}

std::unordered_map<uint64_t, uint64_t> TreeIndex::get_relation(
    int level, const std::vector<uint64_t>& ids) {
  std::unordered_map<uint64_t, uint64_t> pi_new;

  for (auto& id : ids) {
    auto code = id_codes_map_[id];
    auto cur_level = meta_.height() - 1;
    while (cur_level > level) {
      code = (code - 1) / meta_.branch();
      cur_level--;
    }
    pi_new[id] = code;
  }
  return pi_new;
}

std::vector<uint64_t> TreeIndex::get_children_given_ancestor_and_level(
    uint64_t ancestor, int level, bool ret_code = true) {
  auto level_code_num = static_cast<uint64_t>(std::pow(meta_.branch(), level));
  auto code_min = level_code_num - 1;
  auto code_max = level * level_code_num - 1;

  std::vector<uint64_t> parent;
  parent.push_back(ancestor);
  std::vector<uint64_t> res;
  size_t p_idx = 0;
  while (true) {
    size_t p_size = parent.size();
    for (; p_idx < p_size; p_idx++) {
      for (int i = 0; i < meta_.branch(); i++) {
        auto code = parent[p_idx] * meta_.branch() + i + 1;
        if (data_.find(code) != data_.end()) parent.push_back(code);
      }
    }
    if ((code_min <= parent[p_idx]) && (parent[p_idx] < code_max)) {
      break;
    }
  }

  res = std::vector<uint64_t>(parent.begin() + p_idx, parent.end());
  if (ret_code == false) {
    for (size_t i = 0; i < res.size(); i++) {
      res[i] = data_[res[i]].id();
    }
  }
  return res;
}

std::vector<uint64_t> TreeIndex::get_travel_path(uint64_t child,
                                                 uint64_t ancestor) {
  std::vector<uint64_t> res;
  while (child > ancestor) {
    res.push_back(data_[child].id());
    child = (child - 1) / meta_.branch();
  }
  return res;
}
}  // end namespace distributed
}  // end namespace paddle
