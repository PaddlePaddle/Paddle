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

#pragma once

#include "Projection.h"

namespace paddle {

/**
 * @brief Context projection concatenate features in adjacent time steps in
 * a sequence. The i-th row of the output is the concatenation of
 * context_length rows of the input. The context_length rows are the
 * consecutive rows from the i+shift_start row.
 *
 * For example, assumed input (x) has 4 words and the dimension of each word
 * representation is 2. If we use zero to pad instead of learned weight to pad,
 * and the context_lenth is 3, the output (y) is:
 *
 * @code
 *  x = [a1, a2;
 *       b1, b2;
 *       c1, c2;
 *       d1, d2]
 *  y = [0,  0,  a1, a2, b1, b2;
 *       a1, a2, b1, b2, c1, c2;
 *       b1, b2, c1, c2, d1, d2;
 *       c1, c2, d1, d2, 0,  0]
 * @endcode
 *
 * The config file api is context_projection.
 */
class ContextProjection : public Projection {
 public:
  /**
   * Constructor. If context_start is zero and context_lenth is one, it will
   * set trainable_padding false. trainable_padding is an optional arguments
   * and if it is set, constructor will set learned weight, which is used to
   * pad output.
   */
  ContextProjection(const ProjectionConfig& config,
                    ParameterPtr parameter,
                    bool useGpu);
  virtual void forward();
  virtual void backward(const UpdateCallback& callback);

  virtual void resetState();

  virtual void setState(LayerStatePtr state);

  virtual LayerStatePtr getState();

  virtual bool init();

 protected:
  std::unique_ptr<Weight> weight_;
  /// number of extra timesteps added at the beginning
  size_t beginPad_;
  /// number of extra timesteps added at the end
  size_t endPad_;
  /// state_ and state2_ are used in sequence generating and saved
  /// previous inputs.
  MatrixPtr state_;
  MatrixPtr state2_;
};

}  // namespace paddle
