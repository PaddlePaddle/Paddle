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
 * Table projection takes index data input. It select rows from parameter
 * where row_id is in input_ids:
 * \f[
 *   out.row[i] += table.row[ids[i]]
 * \f]
 * where \f$out\f$ is out, \f$table\f$ is parameter, \f$ids\f$ is input_ids,
 * and \f$i\f$ is row_id.
 *
 * The config file api is table_projection.
 *
 * @note If \f$ids[i] = -1\f$, it will be ignored.
 */
class TableProjection : public Projection {
 public:
  TableProjection(const ProjectionConfig& config,
                  const ParameterPtr& parameter,
                  bool useGpu);
  /**
   * If use sparse row matrix as parameter, prefetch feature ids in input label.
   */
  virtual void prefetch(const Argument* in);
  virtual void forward();
  virtual void backward(const UpdateCallback& callback);

 protected:
  std::unique_ptr<Weight> table_;
};

}  // namespace paddle
