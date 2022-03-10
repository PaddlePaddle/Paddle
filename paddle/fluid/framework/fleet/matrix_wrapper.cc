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

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include "paddle/fluid/framework/fleet/matrix_wrapper.h"
#include "paddle/fluid/framework/io/fs.h"

namespace paddle {
namespace framework {

std::shared_ptr<MatrixWrapper> MatrixWrapper::s_instance_(nullptr);

int Nodes::load(std::string path) {
  uint64_t linenum = 0;
  //size_t idx = 0;
  std::vector<std::string> lines;
  std::vector<std::string> strs;
  std::vector<std::string> items;

  int err_no;
  std::shared_ptr<FILE> fp_ = fs_open_read(path, &err_no, "");
  string::LineFileReader reader;
  while (reader.getline(&*(fp_.get()))) {
    auto line = std::string(reader.get());
    strs.clear();
    boost::split(strs, line, boost::is_any_of("\t"));
    //if (0 == linenum) {
    //  if (0 == _total_node_num) {
    //    //_total_node_num = boost::lexical_cast<size_t>(strs[0]);
    //    _nodes = new Node[_total_node_num];
    //    ++linenum;
    //    continue;
    //  }
    //}
    //if (strs.size() < 4) {
    //  LOG(WARNING) << "each line must has more than field";
    //  return -1;
    //}
    
    //Node& node = _nodes[idx];
    // id
    ++linenum;
    uint64_t nid = boost::lexical_cast<uint64_t>(strs[0]);
    //if (_node_map.find(nid) != _node_map.end()){
    //  continue; 
    //}
    std::shared_ptr<Node> node = std::make_shared<Node>();
    node->id = nid;
    // embedding
    items.clear();
    if (!strs[1].empty()) {
      boost::split(items, strs[1], boost::is_any_of(" "));
      for (size_t i = 0; i != items.size(); ++i) {
        node->paths.emplace_back(boost::lexical_cast<uint32_t>(items[i]));
      }
    }
    _node_map[node->id] = node;
    ++_total_node_num;
    //++idx;
  }
  LOG(INFO) << "all lines:" << linenum << ", " << ", total_node_num:" << _total_node_num;
  return 0;
}
void MatrixWrapper::sample(const uint16_t sample_slot, const std::vector<uint16_t> path_slots,
          std::vector<Record>* src_datas,
          std::vector<Record>* sample_results, const uint16_t path_num) {
  const uint16_t K_NUM = 8;
  const uint16_t D_NUM = 4;

  if (NULL == _nodes){
      _nodes = std::make_shared<Nodes>();
      std::cout << "create nodes\n";
  }
  uint64_t table_id = boost::lexical_cast<uint64_t>(0);
  std::shared_ptr<Node> node = std::make_shared<Node>();
  node->id = boost::lexical_cast<uint64_t>("10019922723212237434");
  node->paths = {1,2,3,4};
  _nodes->_node_map[node->id] = node;


  // push path
  std::vector<uint64_t> fea_keys_push;
  fea_keys_push.emplace_back(boost::lexical_cast<uint64_t>("10019922723212237434"));
  std::vector<float*> push_g_vec;
  std::vector<float> n_vec = {4.0, 3.0, 2.0, 1.0};
  float* n_ptr = n_vec.data(); 
  push_g_vec.push_back(n_ptr);
  auto fleet_ptr = FleetWrapper::GetInstance();

  auto status_push = fleet_ptr->pslib_ptr_->_worker_ptr->push_path(
                table_id, fea_keys_push.data(), (const float**)push_g_vec.data(),
                      fea_keys_push.size());
  std::vector<::std::future<int32_t>> push_sparse_status;
  push_sparse_status.resize(0);
  push_sparse_status.push_back(std::move(status_push));
  for (auto& t : push_sparse_status) {
    t.wait();
  }
  std::cout<< "end push path" << std::endl;

  // pull path
  std::vector<uint64_t> fea_keys;
  std::vector<float*> pull_result_ptr;
  fea_keys.emplace_back(boost::lexical_cast<uint64_t>("10019922723212237434"));
  pull_result_ptr.push_back(node->paths.data());
  std::vector<::std::future<int32_t>> pull_sparse_status;
  pull_sparse_status.resize(0);
  auto status = fleet_ptr->pslib_ptr_->_worker_ptr->pull_path(
     pull_result_ptr.data(), table_id, fea_keys.data(), fea_keys.size());
  pull_sparse_status.push_back(std::move(status));
  for (auto& t : pull_sparse_status) {
    t.wait();
    auto status = t.get();
    if (status != 0) {
      LOG(ERROR) << "fleet pull sparse failed, status[" << status << "]";
      sleep(300);
      exit(-1);
    }
  }
  std::cout << "pull keys:" << fea_keys[0] << ", pull path:" << node->paths[0] << ", " << node->paths[1];

  sample_results->clear();
  for (auto& data : *src_datas) {
    VLOG(1) << "src record";
    //data.Print();
    uint64_t start_idx = sample_results->size();
    VLOG(1) << "before sample, sample_results.size = " << start_idx;
    uint64_t sample_feasign_idx = -1, path_first_feasign_idx = -1;
    bool sample_sign = false, path_sign = false; 
    for (uint64_t i = 0; i < data.uint64_feasigns_.size(); i++) {
      if (data.uint64_feasigns_[i].slot() == sample_slot) {
        sample_sign = true;
        sample_feasign_idx = i;
      }
      // trick for Path for first slot
      if (data.uint64_feasigns_[i].slot() == path_slots[0]) {
        path_sign = true;
        path_first_feasign_idx = i;
      }
      if (sample_sign && path_sign)
        break;
    }
    if (!path_sign){
      VLOG(1) << "could not find path slot :" << path_slots[0];
      continue;
    }

    VLOG(1) << "sample_feasign_idx: " << sample_feasign_idx
            << "; path_first_feasign_idx: " << path_first_feasign_idx;
    // why > 0?
    if (sample_sign) {
      if (_nodes->_node_map.find(data.uint64_feasigns_[sample_feasign_idx].sign().uint64_feasign_) == _nodes->_node_map.end()){
        //std::cout<<"could not find this feasign :" << data.uint64_feasigns_[sample_feasign_idx].sign().uint64_feasign_ <<"\n";
        continue;
      } else {
        Record instance(data);
        std::shared_ptr<Node> node_now = _nodes->_node_map.at(data.uint64_feasigns_[sample_feasign_idx].sign().uint64_feasign_);  
        //std::cout << "key id:" << data.uint64_feasigns_[sample_feasign_idx].sign().uint64_feasign_ << ", node id:" << node_now->id << "\n"; 
        for (uint16_t p_num = 0; p_num < path_num; p_num++){
          for (uint16_t d_num = 0; d_num < D_NUM; d_num++) {
            // path id from 1
            instance.uint64_feasigns_[path_first_feasign_idx + p_num * D_NUM + d_num].sign().uint64_feasign_ = static_cast<uint64_t>(node_now->paths[p_num * D_NUM + d_num]) + d_num * K_NUM + 1;
          }
        }
        sample_results->push_back(instance);
      }
    }
  }
  return;
  }

}  // end namespace framework
}  // end namespace paddle
