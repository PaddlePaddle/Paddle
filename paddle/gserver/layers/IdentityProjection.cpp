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
#include "paddle/utils/Stat.h"

namespace paddle {

/**
 * IdentityProjection performs addition:
 * \f[
 *   out.row[i] += in.row[i]
 * \f]
 *
 * The config file api is identity_projection.
 */
class IdentityProjection : public Projection {
 public:
  IdentityProjection(const ProjectionConfig& config,
                     const ParameterPtr& parameter,
                     bool useGpu);
  virtual void forward();
  virtual void backward(const UpdateCallback& callback);
};

REGISTER_PROJECTION(identity, IdentityProjection);

/**
 * Constructed function.
 * @note IdentityProjection should not have any parameter.
 */
IdentityProjection::IdentityProjection(const ProjectionConfig& config,
                                       const ParameterPtr& parameter,
                                       bool useGpu)
    : Projection(config, parameter, useGpu) {
  CHECK(!parameter) << "'identity' projection should not have any parameter";
}

void IdentityProjection::forward() { out_->value->add(*in_->value); }

void IdentityProjection::backward(const UpdateCallback& callback) {
  if (in_->grad) {
    in_->grad->add(*out_->grad);
  }
}

/**
 * IdentityOffsetProjection likes IdentityProjection, but layer size may be
 * smaller
 * than input size. It selects dimensions [offset, offset+layer_size) from input
 * to
 * perform addition:
 * \f[
 *   out.row[i] += in.row[i + \textrm{offset}]
 * \f]
 *
 * The config file api is identity_projection.
 */
class IdentityOffsetProjection : public Projection {
 public:
  IdentityOffsetProjection(const ProjectionConfig& config,
                           const ParameterPtr& parameter,
                           bool useGpu);
  virtual void forward();
  virtual void backward(const UpdateCallback& callback);
};

REGISTER_PROJECTION(identity_offset, IdentityOffsetProjection);

/**
 * Constructed function.
 * @note IdentityOffsetProjection should not have any parameter.
 */
IdentityOffsetProjection::IdentityOffsetProjection(
    const ProjectionConfig& config, const ParameterPtr& parameter, bool useGpu)
    : Projection(config, parameter, useGpu) {
  CHECK(!parameter) << "'identity_offset' projection "
                       "should not have any parameter";
  CHECK_LE(config.output_size() + config.offset(), config.input_size());
}

void IdentityOffsetProjection::forward() {
  out_->value->addAtOffset(*in_->value, config_.offset());
}

void IdentityOffsetProjection::backward(const UpdateCallback& callback) {
  if (in_->grad) {
    in_->grad->addAtOffset(*out_->grad, config_.offset());
  }
}

}  // namespace paddle
