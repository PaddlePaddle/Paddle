#include "paddle/fluid/framework/fleet/index_sampler.h"

namespace paddle {
namespace framework {
    
std::vector<std::vector<uint64_t>> LayerWiseSampler::sample(std::vector<std::vector<uint64_t>>& user_inputs, std::vector<uint64_t>& target_ids, bool with_hierarchy) {
    std::vector<std::vector<uint64_t>> outputs;
    return outputs;
}

std::vector<std::vector<uint64_t>> BeamSearchSampler::sample(std::vector<std::vector<uint64_t>>& user_inputs, std::vector<uint64_t>& target_ids, bool with_hierarchy) {
    std::vector<std::vector<uint64_t>> outputs;
    return outputs;
}

}  // end namespace framework
}  // end namespace paddle