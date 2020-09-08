#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/data_feed.h"

namespace paddle {
namespace framework {

struct Node {
  Node::Node() : parent_node(NULL), id(0), height(0) {}
  ~Node(){};
  std::vector<Node*> sub_nodes;
  // uint32_t sub_node_num;
  Node* parent_node;
  uint64_t id;
  std::vector<float> embedding;
  int16_t height;  //层级
};

class Tree {
 public:
  Tree() : _nodes(NULL), _head(NULL) {}
  ~Tree() {
    if (_nodes) {
      delete[] _nodes;
      _nodes = NULL;
    }
  }

  void print_tree();
  int dump_tree(const uint64_t table_id, int fea_value_dim,
                const std::string tree_path);
  //采样：从叶节点回溯到根节点
  void trace_back(uint64_t id, std::vector<std::pair<uint64_t, uint32_t>>& ids);
  int load(std::string path);
  Node* get_node();
  size_t get_total_node_num();

 private:
  // tree data info
  Node* _nodes{nullptr};
  // head pointer
  Node* _head{nullptr};
  // total number of nodes
  size_t _total_node_num{0};
  // leaf node map
  std::unordered_map<uint64_t, Node*> _leaf_node_map;
  // version
  std::string _version{""};
  //树的高度
  int16_t _tree_height{0};
};

using TreePtr = std::shared_ptr<Tree>;
class TreeWrapper {
 public:
  virtual ~TreeWrapper() {}
  TreeWrapper() {}

  // TreeWrapper singleton
  static std::shared_ptr<TreeWrapper> GetInstance() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::framework::TreeWrapper());
    }
    return s_instance_;
  }

  void clear() { tree_map.clear(); }

  void insert(std::string name, std::string tree_path) {
    if (tree_map.find(name) != tree_map.end()) {
      return;
    }
    TreePtr tree = new Tree();
    tree.load(tree_path);
    tree_map.insert(std::pair<std::string, TreePtr>{name, tree});
  }

  void dump(std::string name, const uint64_t table_id, int fea_value_dim,
            const std::string tree_path) {
    if (tree_map.find(name) == tree_map.end()) {
      return;
    }
    tree_map.at(name)->dump_tree(table_id, fea_value_dim, tree_path);
  }

  void sample(const uint16_t sample_slot, const uint64_t type_slot,
              std::vector<Record>& src_datas,
              std::vector<Record>& sample_results) {
    sample_results.clear();
    for (auto& data : src_datas) {
      uint64_t sample_feasign_idx = -1, type_feasign_idx = -1;
      for (auto i = 0; i < data.uint64_feasigns_.size(); i++) {
        if (data.uint64_feasigns_[i].slot() == sample_slot) {
          sample_feasign_idx = i;
        }
        if (data.uint64_feasigns_.slot() == type_slot) {
          type_feasign_idx = i;
        }
      }
      if (sample_feasign_idx > 0) {
        std::vector<std::pair<uint64_t, uint32_t>> trace_ids;
        for (auto name : tree_map) {
          bool in_tree = tree_map.at(name)->trace_back(
              data.uint64_feasigns_[sample_feasign_idx].sign().uint64_feasign_,
              trace_ids);
          if (in_tree) {
            break;
          } else {
            PADDLE_ENFORCE_EQ(trace_ids.size(), 0, "");
          }
        }
        for (auto i = 0; i < trace_ids.size(); i++) {
          Record instance(data);
          instance.uint64_feasigns_[sample_feasign_idx].sign().uint64_feasign_ =
              trace_ids[i].first;
          if (type_feasign_idx > 0)
            instance.uint64_feasigns_[type_feasign_idx]
                .sign()
                .uint64_feasign_ += trace_ids[i].second * 100;
          sample_results.push_back(instance);
        }
      }
    }
    return;
  }

 public:
  std::unordered_map<std::string, TreePtr> tree_map;

 private:
  static std::shared_ptr<TreeWrapper> s_instance_;
};

}  // end namespace framework
}  // end namespace paddle
