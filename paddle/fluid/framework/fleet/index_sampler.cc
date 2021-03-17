#include "paddle/fluid/framework/fleet/index_sampler.h"

namespace paddle {
namespace framework {
    
std::vector<std::vector<uint64_t>> LayerWiseSampler::sample(std::vector<std::vector<uint64_t>>& user_inputs, std::vector<uint64_t>& target_ids) {
    std::vector<std::vector<uint64_t>> outputs;
    return outputs;
    //auto ids_num = target_ids->size();
    //(ids_num * layer_counts_sum_, std::vector<uint64_t>(inputs[0].size() + 1));

//     std::vector<std::vector<uint64_t>> ancestors(ids_num);
//     tree_->Ancestors(*input_ids, ancestors);
//     int i = 0;
//     for (auto it = ancestors.begin(); it != ancestors.end(); ++it) {
//     auto& ancs = *it;
//     if (!ancs.empty()) {
//       if (ancs.size() > layer_counts_.size()) {
//         ancs.resize(layer_counts_.size());
//       }

//       auto level = tree_->tree_height();
//       for (size_t j = 0; j < ancs.size(); ++j) {
//         level --;
//         // sample +
//         outputs->at(i).first = ancs[j];
//         outputs->at(i).second = 1;
//         i ++;

//         // sample: -
//         auto layer_nodes_info = tree_->layer_nodes(level);

//         size_t cur_layer_count = layer_counts_.at(level);
//         std::unordered_set<int> neighbor_indices_set;
//         std::vector<int> neighbor_indices;
//         neighbor_indices_set.reserve(cur_layer_count);
//         neighbor_indices.reserve(cur_layer_count);
//         auto neighbors_count = layer_nodes_info.second;

//         static __thread std::hash<std::thread::id> hasher;
//         static __thread std::mt19937 rng(
//             clock() + hasher(std::this_thread::get_id()));
//         std::uniform_int_distribution<int> distrib(0, neighbors_count);
         
//         while (neighbor_indices_set.size() < cur_layer_count) {
//           int q = distrib(rng);
//           auto id = layer_nodes_info.first[q].id;
//           if (neighbor_indices_set.find(q) != neighbor_indices_set.end() || id == ancs[j]) {
//             continue;
//           }

//           neighbor_indices_set.insert(id);
//           neighbor_indices.push_back(id);
//         }
//         for (size_t k = 0; k < cur_layer_count; ++k) {
//           outputs->at(i).first = neighbor_indices[k];
//           outputs->at(i).second = 0;
//           i++;
//         }
//       }
//     }
//   }
//   return;
}

std::vector<std::vector<uint64_t>> BeamSearchSampler::sample(std::vector<std::vector<uint64_t>>& user_inputs, std::vector<uint64_t>& target_ids) {
    std::vector<std::vector<uint64_t>> outputs;
    return outputs;
}

}  // end namespace framework
}  // end namespace paddle