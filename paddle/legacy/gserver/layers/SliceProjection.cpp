/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "Projection.h"

namespace paddle {

/**
 * SliceProjection can slice the input value into multiple parts,
 * and then select some of them to merge into a new output.
 *
 * First, calculate the slices that need to be merged into the output.
 * slices = input.slices().for_output()
 *
 * Second, merge each slice into the output.
 * for(auto slice: slices) {
 *   out.addAtOffset(slice, offset);
 * }
 *
 * Input slices as output: s0, s1, ...:
 *   -----------------------
 *   |///|   |//////|      |
 *   |/s0|   |//s1//|      |
 *   |///|   |//////|      |
 *   -----------------------
 * Output, merge s0, s1, ... into one output:
 *   ----------------
 *   |///|//////|   |
 *   |/s0|//s1//|...|
 *   |///|//////|   |
 *   ----------------
 *
 * The config file api is slice_projection.
 */
class SliceProjection : public Projection {
 public:
  SliceProjection(const ProjectionConfig& config,
                  const ParameterPtr& parameter,
                  bool useGpu);
  virtual void forward();
  virtual void backward(const UpdateCallback& callback);

 protected:
  std::vector<std::pair<size_t, size_t>> slices_;
};

REGISTER_PROJECTION(slice, SliceProjection);

/**
 * Constructed function.
 * @note SliceProjection should not have any parameter.
 */
SliceProjection::SliceProjection(const ProjectionConfig& config,
                                 const ParameterPtr& parameter,
                                 bool useGpu)
    : Projection(config, parameter, useGpu) {
  CHECK(!parameter) << "'slice' projection should not have any parameter";

  slices_.reserve(config.slices_size());
  for (const auto& slice : config.slices()) {
    slices_.push_back(std::make_pair(slice.start(), slice.end()));
  }
}

void SliceProjection::forward() {
  size_t offset = 0;
  for (auto& slice : slices_) {
    auto slice_out = in_->value->subColMatrix(slice.first, slice.second);
    out_->value->addAtOffset(*slice_out, offset);
    offset += slice_out->getWidth();
  }
}

void SliceProjection::backward(const UpdateCallback& callback) {
  if (in_->grad) {
    size_t offset = 0;
    for (auto& slice : slices_) {
      auto slice_out = in_->grad->subColMatrix(slice.first, slice.second);
      slice_out->addAtOffset(*out_->grad, offset);
      offset += slice_out->getWidth();
    }
  }
}

}  // namespace paddle
