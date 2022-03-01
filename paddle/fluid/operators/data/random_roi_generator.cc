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

#include "paddle/fluid/operators/data/random_roi_generator.h"

namespace paddle {
namespace operators {
namespace data {

RandomROIGenerator::RandomROIGenerator(
    AspectRatioRange aspect_ratio_range, AreaRange area_range,
    int64_t seed, int64_t num_attempts)
    : aspect_ratio_range_(aspect_ratio_range),
      area_range_(area_range),
      random_generator_(seed),
      seed_(seed),
      num_attempts_(num_attempts) {}

void RandomROIGenerator::GenerateRandomROI(
    const int64_t width, const int64_t height, ROI* roi) {
  if (width <= 0 || height <= 0) return;
	
	float min_wh_ratio = aspect_ratio_range_.first;
  float max_wh_ratio = aspect_ratio_range_.second;
  float max_hw_ratio = 1 / aspect_ratio_range_.first;
  float min_area = width * height * area_distribution_.a();
  auto max_width = std::max<int64_t>(1, height * max_wh_ratio);
  auto max_height = std::max<int64_t>(1, width * max_hw_ratio);

  // process max_width/height cannot satisfy min_area restriction firstly
  if (height * max_width < min_area) {
    roi->w = max_width;
    roi->h = height;
  } else if (width * max_height < min_area) {
    roi->w = width;
    roi->h = max_height;
  } else {
    int64_t attempts = num_attempts_;
    while (attempts-- > 0) {
      // calc ROI area
      float scale = area_distribution_(random_generator_);
      float roi_area = scale * height * width;

      // calc ROI width/height
      float ratio = std::exp(
                    aspect_ratio_distribution_(random_generator_));
      auto w = static_cast<int64_t>(
              std::roundf(sqrtf(roi_area * ratio)));
      auto h = static_cast<int64_t>(
              std::roundf(sqrtf(roi_area / ratio)));
      w = std::max<int64_t>(1, w);
      h = std::max<int64_t>(1, h);

      // check restrictions
      ratio = static_cast<float>(w) / h;
      if (w <= width && h <= height
          && ratio >= min_wh_ratio && ratio <= max_hw_ratio) {
        roi->w = w;
        roi->h = h;
        break;
      }
    }

    if (attempts <= 0) {
      float max_area = area_distribution_.b() * width * height;
      float ratio = static_cast<float>(width) / height;
      int64_t w, h;
      if (ratio > max_wh_ratio) {
        w = max_width;
        h = height;
      } else if (ratio < min_wh_ratio) {
        w = width;
        h = max_height;
      } else {
        w = width;
        h = height;
      }
      float scale = std::min(1.f, max_area / (w * h));
      roi->w = std::max<int64_t>(1, w * sqrtf(scale));
      roi->h = std::max<int64_t>(1, h * sqrtf(scale));
    }

    // generate random left top coordination x, y
    roi->x = std::uniform_int_distribution<int64_t>(
                  0, width - roi->w)(random_generator_);
    roi->y = std::uniform_int_distribution<int64_t>(
                  0, height - roi->h)(random_generator_);
  }
}

// initialization static variables out of GeneratorManager
GeneratorManager* GeneratorManager::gm_instance_ptr_ = nullptr;
std::mutex GeneratorManager::m_;

}  // namespace data
}  // namespace operators
}  // namespace paddle
