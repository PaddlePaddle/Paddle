#include <vector>
#include "paddle/fluid/framework/fleet/index_wrapper.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace framework {

class Sampler {
 public:
  virtual ~Sampler() {}
  Sampler() {}

  template <typename T>
  static std::shared_ptr<Sampler> Init(const std::string& name) {
    std::shared_ptr<Sampler> instance = nullptr;
    instance.reset(new T(name));
    return instance;
  }

  virtual void init_layerwise_conf(const std::vector<int64_t> &layer_sample_counts) {};
  virtual void init_beamsearch_conf(const int64_t k) {};
  virtual std::vector<std::vector<uint64_t>> sample(std::vector<std::vector<uint64_t>>& user_inputs, std::vector<uint64_t>& input_targets, bool with_hierarchy=false) = 0;
};

class LayerWiseSampler : public Sampler {
 public:
  virtual ~LayerWiseSampler() {}
  LayerWiseSampler(const std::string& name) {tree_ = IndexWrapper::GetInstance()->GetTreeIndex(name);}

  void init_layerwise_conf(const std::vector<int64_t> &layer_sample_counts) override {
      for (int i = 0; i < tree_->height(); ++i) {
          layer_counts_sum_ += layer_sample_counts[i] + 1;
          layer_counts_.push_back(layer_sample_counts[i]);
          VLOG(0) << "[INFO] level " << i << " layer_counts.push_back: " << layer_sample_counts[i];
      }
  }
  std::vector<std::vector<uint64_t>> sample(std::vector<std::vector<uint64_t>>& user_inputs, std::vector<uint64_t>& target_ids, bool with_hierarchy) override;
 
 private:
  std::vector<int64_t> layer_counts_;
  int64_t layer_counts_sum_;
  TreePtr tree_{nullptr};
};

class BeamSearchSampler : public Sampler{
 public:
  virtual ~BeamSearchSampler() {}
  BeamSearchSampler(const std::string& name) {tree_ = IndexWrapper::GetInstance()->GetTreeIndex(name);}

  void init_beamsearch_conf(const int64_t k) override {
    k_ = k;
    return;
  }
  std::vector<std::vector<uint64_t>> sample(std::vector<std::vector<uint64_t>>& user_inputs, std::vector<uint64_t>& target_ids, bool with_hierarchy) override;
 
 private:
  int64_t k_;
  TreePtr tree_{nullptr};
};

}  // end namespace framework
}  // end namespace paddle