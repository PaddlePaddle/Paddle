/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include <vector>
#include <random>
#include <utility>
#include <mutex>
#include <memory>
#include <map>

namespace paddle {
namespace operators {
namespace data {

using AspectRatioRange = std::pair<float, float>;
using AreaRange = std::pair<float, float>;

struct ROI {
  // left top coordination (x, y)
  int64_t x;
  int64_t y;
  // width/height of crop window (w, h)
  int64_t w;
  int64_t h;
};

class RandomROIGenerator {
  public:
    explicit RandomROIGenerator(
        AspectRatioRange aspect_ratio_range, AreaRange area_range,
        int64_t seed = time(0), int64_t num_attempts = 10);

    void GenerateRandomROI(const int64_t width, const int64_t height, ROI* roi);

  private:

    AspectRatioRange aspect_ratio_range_;
    AreaRange area_range_;

    std::uniform_real_distribution<float> aspect_ratio_distribution_;
    std::uniform_real_distribution<float> area_distribution_;
    std::mt19937 random_generator_;

    int64_t seed_;
    int64_t num_attempts_;
};

class GeneratorManager {
  using Generators = std::vector<std::unique_ptr<RandomROIGenerator>>;

  private:
    // DISABLE_COPY_AND_ASSIGN(GeneratorManager);

    static GeneratorManager* gm_instance_ptr_;
    static std::mutex m_;

    std::map<int64_t, std::unique_ptr<Generators>> prog_id_to_generators_;

  public:
    static GeneratorManager *Instance() {
      if (gm_instance_ptr_ == nullptr) {
        std::lock_guard<std::mutex> lk(m_);
        if (gm_instance_ptr_ == nullptr) {
          gm_instance_ptr_ = new GeneratorManager;
        }
      }
      return gm_instance_ptr_;
    }

    Generators* GetGenerators(const int64_t program_id, const int batch_size,
                      const AspectRatioRange aspect_ratio_range,
                      const AreaRange area_range) {
      auto iter = prog_id_to_generators_.find(program_id);
      if (iter == prog_id_to_generators_.end()) {
        prog_id_to_generators_[program_id] = std::unique_ptr<Generators>(
                                             new Generators(batch_size));

        std::seed_seq rand_seq{static_cast<int64_t>(time(0))};
        std::vector<int> rands(batch_size);
        rand_seq.generate(rands.begin(), rands.end());

        for (int i = 0; i < batch_size; i++) {
          prog_id_to_generators_[program_id]->at(i).reset(
              new RandomROIGenerator(aspect_ratio_range,
                                      area_range, rands[i]));
        }
      }
      return prog_id_to_generators_[program_id].get();
    } 
};

}  // namespace data
}  // namespace operators
}  // namespace paddle
