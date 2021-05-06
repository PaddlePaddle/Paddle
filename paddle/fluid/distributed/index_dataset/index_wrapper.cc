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

#include <math.h>
#include <memory>
#include <string>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/io/fs.h"

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include "paddle/fluid/distributed/index_dataset/index_wrapper.h"

namespace paddle {
namespace distributed {

std::shared_ptr<IndexWrapper> IndexWrapper::s_instance_(nullptr);

int TreeIndex::Load(const std::string filename) {
  int err_no;
  auto fp = paddle::framework::fs_open_read(filename, &err_no, "");
  PADDLE_ENFORCE_NE(
      fp, nullptr,
      platform::errors::InvalidArgument(
          "Open file %s failed. Please check whether the file exists.",
          filename));

  int num = 0;
  max_id_ = 0;
  fake_node_.set_id(0);
  fake_node_.set_is_leaf(false);
  fake_node_.set_probability(0.0);
  max_code_ = 0;
  size_t ret = fread(&num, sizeof(num), 1, fp.get());
  while (ret == 1 && num > 0) {
    std::string content(num, '\0');
    size_t read_num =
        fread(const_cast<char*>(content.data()), 1, num, fp.get());
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
      IndexNode node;
      node.ParseFromString(item.value());
      PADDLE_ENFORCE_NE(node.id(), 0,
                        platform::errors::InvalidArgument(
                            "Node'id should not be equel to zero."));
      if (node.is_leaf()) {
        id_codes_map_[node.id()] = code;
      }
      data_[code] = node;
      if (node.id() > max_id_) {
        max_id_ = node.id();
      }
      if (code > max_code_) {
        max_code_ = code;
      }
    }
    ret = fread(&num, sizeof(num), 1, fp.get());
  }
  total_nodes_num_ = data_.size();
  max_code_ += 1;
  return 0;
}

std::vector<IndexNode> TreeIndex::GetNodes(const std::vector<uint64_t>& codes) {
  std::vector<IndexNode> nodes;
  nodes.reserve(codes.size());
  for (size_t i = 0; i < codes.size(); i++) {
    if (CheckIsValid(codes[i])) {
      nodes.push_back(data_.at(codes[i]));
    } else {
      nodes.push_back(fake_node_);
    }
  }
  return nodes;
}

std::vector<uint64_t> TreeIndex::GetLayerCodes(int level) {
  uint64_t level_num = static_cast<uint64_t>(std::pow(meta_.branch(), level));
  uint64_t level_offset = level_num - 1;

  std::vector<uint64_t> res;
  res.reserve(level_num);
  for (uint64_t i = 0; i < level_num; i++) {
    auto code = level_offset + i;
    if (CheckIsValid(code)) {
      res.push_back(code);
    }
  }
  return res;
}

std::vector<uint64_t> TreeIndex::GetAncestorCodes(
    const std::vector<uint64_t>& ids, int level) {
  std::vector<uint64_t> res;
  res.reserve(ids.size());

  int cur_level;
  for (size_t i = 0; i < ids.size(); i++) {
    if (id_codes_map_.find(ids[i]) == id_codes_map_.end()) {
      res.push_back(max_code_);
    } else {
      auto code = id_codes_map_.at(ids[i]);
      cur_level = meta_.height() - 1;

      while (level >= 0 && cur_level > level) {
        code = (code - 1) / meta_.branch();
        cur_level--;
      }
      res.push_back(code);
    }
  }
  return res;
}

std::vector<uint64_t> TreeIndex::GetChildrenCodes(uint64_t ancestor,
                                                  int level) {
  auto level_code_num = static_cast<uint64_t>(std::pow(meta_.branch(), level));
  auto code_min = level_code_num - 1;
  auto code_max = meta_.branch() * level_code_num - 1;

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

  return std::vector<uint64_t>(parent.begin() + p_idx, parent.end());
}

std::vector<uint64_t> TreeIndex::GetTravelCodes(uint64_t id, int start_level) {
  std::vector<uint64_t> res;
  PADDLE_ENFORCE_NE(id_codes_map_.find(id), id_codes_map_.end(),
                    paddle::platform::errors::InvalidArgument(
                        "id = %d doesn't exist in Tree.", id));
  auto code = id_codes_map_.at(id);
  int level = meta_.height() - 1;

  while (level >= start_level) {
    res.push_back(code);
    code = (code - 1) / meta_.branch();
    level--;
  }
  return res;
}

std::vector<IndexNode> TreeIndex::GetAllLeafs() {
  std::vector<IndexNode> res;
  res.reserve(id_codes_map_.size());
  for (auto& ite : id_codes_map_) {
    auto code = ite.second;
    res.push_back(data_.at(code));
  }
  return res;
}

int GraphIndex::save(std::string filename) {
  std::function<int(FILE*, KVItem&)> writeToFile = [](FILE* fp, KVItem& item) {
    std::string output;
    item.SerializeToString(&output);
    int len = output.size();
    if (fwrite(&len, sizeof(int), 1, fp) != 1) {
      VLOG(0) << "write len failed";
      return -1;
    }
    if (fwrite(output.data(), 1, len, fp) != (size_t)len) {
      VLOG(0) << "write data failed";
      return -1;
    }
    return 0;
  };
  VLOG(0) << " in save height = " << height() << " width = " << width();
  FILE* fp = fopen(filename.c_str(), "wb");
  if (fp == NULL) {
    fprintf(stderr, "Can not open file: %s\n", filename.c_str());
    return -1;
  }
  KVItem item;
  item.set_key(".graph_meta");
  std::string output;
  meta_.SerializeToString(&output);
  GraphMeta test;
  test.ParseFromString(output);
  VLOG(0) << "test height = " << test.height() << " " << test.width();
  item.set_value(output);
  if (writeToFile(fp, item) != 0) {
    fprintf(stderr, "fail to write file: %s\n", filename.c_str());
    fclose(fp);
    return -1;
  }
  for (auto p : item_path_dict_) {
    item.set_key(std::to_string(p.first));
    GraphItem graph_item;
    graph_item.set_item_id(p.first);
    for (auto path_id : p.second) graph_item.add_path_id(path_id);
    std::string graph_serialized;
    graph_item.SerializeToString(&graph_serialized);
    item.set_value(graph_serialized);
    if (writeToFile(fp, item) != 0) {
      fprintf(stderr, "fail to write file: %s\n", filename.c_str());
      fclose(fp);
      return -1;
    }
  }
  fclose(fp);
  return 0;
}
int GraphIndex::load(std::string filename) {
  FILE* fp = fopen(filename.c_str(), "rb");
  if (fp == NULL) {
    fprintf(stderr, "Can not open file: %s\n", filename.c_str());
    return -1;
  }

  int num = 0;
  item_path_dict_.clear();
  path_item_set_dict_.clear();
  size_t ret = fread(&num, sizeof(num), 1, fp);
  while (ret == 1 && num > 0) {
    std::string content(num, '\0');
    char buffer[num + 1];
    memset(buffer, '\0', sizeof(buffer));
    if (fread(buffer, 1, num, fp) != static_cast<size_t>(num)) {
      fprintf(stderr, "Read from file: %s failed, invalid format.\n",
              filename.c_str());
      break;
    }
    content = buffer;
    KVItem item;
    if (!item.ParseFromString(content)) {
      fprintf(stderr, "Parse from file: %s failed.\n", filename.c_str());
      break;
    }
    if (item.key() == ".graph_meta") {
      meta_.ParseFromString(item.value());
      path_item_set_dict_.reserve(std::pow(meta_.height(), meta_.width()));
    } else {
      GraphItem graph_item;
      graph_item.ParseFromString(item.value());

      uint64_t item_id = graph_item.item_id();

      if (item_path_dict_.find(item_id) == item_path_dict_.end()) {
        std::vector<uint32_t> path_ids;
        for (int i = 0; i < graph_item.path_id_size(); i++) {
          path_ids.push_back(graph_item.path_id(i));
          VLOG(0) << "Graph insert item: " << item_id
                  << " path: " << graph_item.path_id(i);
        }
        item_path_dict_[item_id] = path_ids;

        for (auto& path_id : path_ids) {
          if (path_item_set_dict_.find(path_id) == path_item_set_dict_.end()) {
            std::unordered_set<uint64_t> path_set;
            path_item_set_dict_[path_id] = path_set;
          }
          path_item_set_dict_[path_id].insert(item_id);
        }
      }
    }
    ret = fread(&num, sizeof(num), 1, fp);
  }
  fclose(fp);
  VLOG(0) << "Graph Load Success.";
  return 0;
}

std::vector<uint32_t> GraphIndex::create_path(uint64_t item_id) {
  if (item_path_dict_.find(item_id) == item_path_dict_.end()) {
    item_path_dict_[item_id] = generate_random_path();
    for (auto path : item_path_dict_[item_id]) {
      path_item_set_dict_[path].insert(item_id);
    }
  }
  return item_path_dict_[item_id];
}

std::vector<uint32_t> GraphIndex::generate_random_path() {
  int h = height(), w = width(), path_nums = item_path_nums();
  uint32_t total_num = pow(h, w) + 0.2;
  std::vector<uint32_t> vec(path_nums, 0);
  std::unordered_set<uint32_t> path_set;
  for (int i = 0; i < path_nums; i++) {
    uint32_t path_id;
    do {
      path_id = rand() % total_num;
    } while (path_set.find(path_id) != path_set.end());
    vec[i] = path_id;
    path_set.insert(path_id);
  }
  return vec;
}

void GraphIndex::add_item(uint64_t item_id, std::vector<uint32_t> vec) {
  if (item_path_dict_.find(item_id) != item_path_dict_.end()) {
    auto path_vec = item_path_dict_[item_id];
    for (auto path : path_vec) {
      auto iter = path_item_set_dict_.find(path);
      if (iter != path_item_set_dict_.end()) {
        iter->second.erase(item_id);
      }
    }
    item_path_dict_[item_id].clear();
  }
  for (auto p : vec) {
    item_path_dict_[item_id].push_back(p);
    path_item_set_dict_[p].insert(item_id);
  }
}

std::vector<std::vector<uint32_t>> GraphIndex::get_path_of_item(
    std::vector<uint64_t>& items) {
  std::vector<std::vector<uint32_t>> result;
  for (auto& item : items) {
    // result.push_back(item_path_dict_[item]);

    result.push_back(create_path(item));
  }
  return result;
}

std::vector<std::vector<uint64_t>> GraphIndex::get_item_of_path(
    std::vector<uint32_t>& paths) {
  std::vector<std::vector<uint64_t>> result;
  for (auto& path : paths) {
    result.push_back(std::vector<uint64_t>(path_item_set_dict_[path].begin(),
                                           path_item_set_dict_[path].end()));
  }
  return result;
}

int GraphIndex::update_Jpath_of_item(
    std::map<uint64_t, std::vector<std::string>>& item_paths, const int T,
    const int J, const double lamd, const int factor) {
  return 0;
}

}  // end namespace distributed
}  // end namespace paddle
