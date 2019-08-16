/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

// Just a abstract class and a stub implementation now
// AiBox will support the actual function
// Will be ready in the end of September

#ifdef PADDLE_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include <cstring>
#include <map>
#include <mutex>  // NOLINT
#include <string>
#include <vector>

namespace paddle {
namespace boxps {

struct FeatureValue {
  FeatureValue() {
    slot = 0;
    show = 0.;
    clk = 0.;
    embed_w = 0.;
    embed_g2sum = 0.;
    delta_score = 0.;
    embedding_size = 0;
    memset(embedx, 0, sizeof(embedx));
    ssd_ptr = nullptr;
  }
  int slot;
  float show;
  float clk;
  float embedx[9];
  float embed_w;
  float embed_g2sum;
  uint32_t embedding_size;
  float delta_score;
  uint64_t *ssd_ptr;
};

struct FeaturePushValue {
  FeaturePushValue() {
    slot = 0;
    show = 0.;
    clk = 0.;
    embed_g = 0.;
    memset(embedx_g, 0, sizeof(embedx_g));
  }
  int slot;
  float show;
  float clk;
  float embedx_g[9];
  float embed_g;
};

class BoxPSBase {
 public:
  BoxPSBase() {}
  virtual ~BoxPSBase() {}

  virtual int FeedPass(int date,
                       const std::vector<uint64_t> &keys_of_one_pass) = 0;
  virtual int BeginPass() = 0;
  virtual int EndPass() = 0;

  virtual int InitializeCPU(const char *conf_file, int minibatch_size) = 0;
  virtual int PullSparseCPU(const uint64_t *keys, FeatureValue **vals,
                            int fea_num) = 0;
  virtual int PushSparseCPU(const uint64_t *keys,
                            const FeaturePushValue *push_vals, int fea_num) = 0;
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
  virtual int InitializeGPU(const char *conf_file, int minibatch_size,
                            const std::vector<cudaStream_t *> &stream_list) = 0;
  virtual int PullSparseGPU(const uint64_t *d_keys, FeatureValue **d_vals,
                            int fea_num, int stream_idx) = 0;
  virtual int PushSparseGPU(const uint64_t *d_keys,
                            const FeaturePushValue *d_push_vals, int fea_num,
                            int stream_idx) = 0;
#endif
  // mode = 0, load all feature
  // mode = 1, laod delta feature, which means load diff
  virtual int LoadModel(const std::string &path, const int mode) = 0;

  // mode = 0, save all feature
  // mode = 1, save delta feature, which means save diff
  virtual int SaveModel(const std::string &path, const int mode) = 0;
};

class FakeBoxPS : public BoxPSBase {
 public:
  FakeBoxPS() {}
  virtual ~FakeBoxPS() {}

  int FeedPass(int date,
               const std::vector<uint64_t> &keys_of_one_pass) override;
  int BeginPass() override;
  int EndPass() override;

  int InitializeCPU(const char *conf_file, int minibatch_size) override;
  int PullSparseCPU(const uint64_t *keys, FeatureValue **vals,
                    int fea_num) override;
  int PushSparseCPU(const uint64_t *keys, const FeaturePushValue *push_vals,
                    int fea_num) override;
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
  int InitializeGPU(const char *conf_file, int minibatch_size,
                    const std::vector<cudaStream_t *> &stream_list) override;
  int PullSparseGPU(const uint64_t *d_keys, FeatureValue **d_vals, int fea_num,
                    int stream_idx) override;
  int PushSparseGPU(const uint64_t *d_keys, const FeaturePushValue *d_push_vals,
                    int fea_num, int stream_idx) override;
#endif
  // mode = 0, load all feature
  // mode = 1, laod delta feature, which means load diff
  int LoadModel(const std::string &path, const int mode) override;

  // mode = 0, save all feature
  // mode = 1, save delta feature, which means save diff
  int SaveModel(const std::string &path, const int mode) override;

 private:
  std::map<uint64_t, std::vector<float>> emb_;
  int hidden_size_ = 11;  // should be read in config file, hard-cord now
  float learning_rate_ = 0.01;
  void PrintAllEmb() const;
  void DebugPrintKey(const uint64_t *d, int len, const std::string &info) const;
  std::mutex map_mutex;
};
}  // namespace boxps
}  // namespace paddle
