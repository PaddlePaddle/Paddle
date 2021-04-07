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

#include <boost/lexical_cast.hpp>
#include "paddle/fluid/framework/fleet/index_wrapper.h"


namespace paddle {
namespace framework {

std::shared_ptr<IndexWrapper> IndexWrapper::s_instance_(nullptr);

int TreeIndex::load(const std::string filename) {
  FILE* fp = fopen(filename.c_str(), "rb");
  if (fp == NULL) {
    fprintf(stderr, "Can not open file: %s\n", filename.c_str());
    return -1;
  }

  int num = 0;
  max_id_ = 0;
  size_t ret = fread(&num, sizeof(num), 1, fp);
  while (ret == 1 && num > 0) {
    std::string content(num, '\0');
    if (fread(const_cast<char*>(content.data()), 1, num, fp) !=
        static_cast<size_t>(num)) {
      fprintf(stderr, "Read from file: %s failed, invalid format.\n",
              filename.c_str());
      break;
    }
    KVItem item;
    if (!item.ParseFromString(content)) {
      fprintf(stderr, "Parse from file: %s failed.\n", filename.c_str());
      break;
    }
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
    std::vector<uint64_t>& ids, int start_level, bool ret_code) {
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
    std::vector<uint64_t>& ids, int level, bool ret_code) {
  std::vector<uint64_t> res;
  res.reserve(ids.size());
  int cur_level;
  for (size_t i = 0; i < ids.size(); i++) {
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

int GraphIndex::load(std::string filename) {
  FILE* fp = fopen(filename.c_str(), "rb");
  if (fp == NULL) {
    fprintf(stderr, "Can not open file: %s\n", filename.c_str());
    return -1;
  }

  int num = 0;
  size_t ret = fread(&num, sizeof(num), 1, fp);
  while (ret == 1 && num > 0) {
    std::string content(num, '\0');
    VLOG(0) << "content: " << content;
    if (fread(const_cast<char*>(content.data()), 1, num, fp) !=
        static_cast<size_t>(num)) {
      fprintf(stderr, "Read from file: %s failed, invalid format.\n",
              filename.c_str());
      break;
    }
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
        std::vector<int64_t> path_ids;
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
          } else {
            path_item_set_dict_[path_id].insert(item_id);
          }
        }
      }
    }
    ret = fread(&num, sizeof(num), 1, fp);
  }
  fclose(fp);
  VLOG(0) << "Graph Load Success.";
  return 0;
}

std::vector<std::vector<int64_t>> GraphIndex::get_path_of_item(
    std::vector<uint64_t>& items) {
  std::vector<std::vector<int64_t>> result;
  for (auto& item : items) {
    result.push_back(item_path_dict_[item]);
  }
  return result;
}

std::vector<std::vector<uint64_t>> GraphIndex::get_item_of_path(
    std::vector<int64_t>& paths) {
  std::vector<std::vector<uint64_t>> result;
  for (auto& path : paths) {
    result.push_back(std::vector<uint64_t>(path_item_set_dict_[path].begin(),
                                           path_item_set_dict_[path].end()));
  }
  return result;
}

int GraphIndex::update_Jpath_of_item(
  std::map<uint64_t, std::vector<std::string>>& item_paths, const int T, const int J, const double lamd, const int factor){
    std::map<uint64_t, std::vector<std::string>> ::iterator item_path; 
    std::unordered_map<uint64_t, std::vector<int64_t>> temp_item_path;  
    std::unordered_map<int64_t, std::unordered_set<uint64_t>> temp_path_item;
    
    std::pair<int64_t, double> top_path_pro(-1,-1); 
    std::vector<std::pair<int64_t, double>> tmp_path_pro; 
    int64_t path_id;
    double probility=0;
    double item_probility=0;

    for(int t=0; t<T; t++){
      int path_cnt=0;
      // printf("####### t_th:%d ####### \n",t);
      for (item_path = item_paths.begin(); item_path != item_paths.end(); item_path++){
        double sum=0;
        uint64_t item_id=item_path->first;
        // printf("====== item_id:%lu ====== \n",item_id);

        for (int j=0; j<J; j++){
          if(t>1){
            path_id=temp_item_path.find(item_id)->second[j];
            path_cnt = temp_path_item.find(path_id)->second.size();
            // printf("****** last_top_jth_path: path_id:%lu, path_cnt:%d ***** \n", path_id, path_cnt);
          }

          for (auto& path_pro_i : item_path->second){ 
            std::string top_path=path_pro_i;
            size_t pos = path_pro_i.find(":");
            probility = stof(path_pro_i.substr(pos+1,path_pro_i.size()));
            item_probility=item_probility+probility;
          }
          // printf("item_id:%lu, item_paths_prob:%f \n",item_id, item_probility);

          for (auto& path_pro_i : item_path->second){ 
            std::string top_path=path_pro_i;
            size_t pos = path_pro_i.find(":");
            path_id = stoi(path_pro_i.substr(0,pos));
            // path_id=tmp_path_pro[ii].first;
            // probility=tmp_path_pro[ii].second;
            int flag=0;
            if(temp_item_path.find(item_id)==temp_item_path.end()){
              temp_item_path[item_id].push_back(path_id);
              probility=log(item_probility+sum)-lamd*((pow(path_cnt+1,factor)/factor)-(pow(path_cnt,factor)/factor));
              // printf("path_id:%lu, - effProb:%f, - path_cnt:%d \n",path_id,probility,path_cnt);
              top_path_pro.first=path_id;
              top_path_pro.second=probility;
            }else{
              for(int i=0; i<j; i++){
                if(temp_item_path[item_id][i]==path_id){
                  flag=1;
                }
              }
              if(flag==0){
                probility=log(item_probility+sum)-lamd*((pow(path_cnt+1,factor)/factor)-(pow(path_cnt,factor)/factor));
                // printf("path_id:%lu, - effProb:%f, - path_cnt:%d \n",path_id,probility,path_cnt);

                if (probility > (top_path_pro.second)){
                  top_path_pro.first=path_id;
                  top_path_pro.second=probility;
                }
                // printf("toper_path_id:%lu, - toper_path_pro=:%f \n",top_path_pro.first, top_path_pro.second);
              }
            }
          }
          temp_item_path[item_id][j]=top_path_pro.first;
          sum=sum+top_path_pro.second;
          temp_path_item[top_path_pro.first].insert(item_id);
          // printf("****** item:%lu, top_jth_path:%lu ****** \n",item_id, top_path_pro.first);
        }
      }
    }
    //update path_item_graph
    path_item_set_dict_=temp_path_item;
    item_path_dict_=temp_item_path;
    return 0;
  }

}  // end namespace framework
}  // end namespace paddle