// Copyright (c) 2021 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! This file implements the class ComputeAtTransform, which help to perform the
//! isl transformation in `compute_at` optimization.
#pragma once
#include <isl/constraint.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/poly/isl_utils.h"
#include "paddle/cinn/poly/map.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace poly {

//! Help to mark the consumer parameters in the generated AST.
static const char* kConsumerParamPrefix = "_cp_";

/**
 * Generate a consumer parameter name in isl sets and maps.
 * e.g the _cp_A_0 in `[_cp_A_0] -> {...}`
 *
 * @param tuple The tuple name of the consumer set.
 * @param id The id of the parameter.
 * @return the name.
 */
std::string GenConsumerParamName(const char* tuple, int id);

/**
 * \brief The ComputeAt transform implemented in polyhedral way.
 *
 * The current implementation for `ComputeAt` schedule primitive is quite
 * complex, it contains the polyhedral transform before the AST generation, and
 * the several passes after AST generation. This class only contains the
 * polyhedral transform:
 * 1. Adjust the producer's domain by the consume accesses.
 * 2. Adjust the producer's transform by
 *   a. Insert the preceding level+1 consumer axis to the head of the original
 * producer transform's domain, to make it compute in the level of consumer
 * forloops. b. b. Adjust the range of the producer's transform by fixing the
 * preceding axis(from the previous step).
 *
 * The latter process after the execution of this class remains, including
 * 1. Get the adjusted shape of the producer after compute_at
 * 2. Update the adjusted buffer's shape
 * 3. Normalize the accesses of the consumers(by making the leftmost access
 * start from zero).
 */
class ComputeAtTransform {
 public:
  ComputeAtTransform(isl::set pdomain,
                     isl::set cdomain,
                     isl::map access,
                     isl::map ptransform,
                     isl::map ctransform,
                     int level);

  void operator()() {
    AdjustPdomain();
    AdjustPtransform();
  }

  const isl::set& adjusted_pdomain() const { return adjusted_pdomain_; }
  const isl::map& adjusted_ptransform() const { return adjusted_ptransform_; }

  //! Display C code
  void DisplayC(isl_map* __isl_give pschedule = nullptr,
                isl_map* __isl_give cschedule = nullptr);

  //! Re-calculate the producer buffer shape after compute_at transform.
  std::vector<int> GetProducerAdjustedShape() const;

  //! Get the the minimum of the preceding level+1 axis in accesses by assuming
  //! all the isl param is zero(for the consumer, the preceding level+1 axis is
  //! fixed in producer computation).
  std::vector<int> GetAccessesPrecedingIndicesMinAssumingParamsZero();

 protected:
  isl_set* __isl_give AddParamsTo(isl_set* __isl_take set);
  isl_map* __isl_give AddParamsTo(isl_map* __isl_take map);

  const char* ptuple() const { return isl_set_get_tuple_name(pdomain_.get()); }
  const char* ctuple() const { return isl_set_get_tuple_name(cdomain_.get()); }

  void AdjustPdomain();

  void AdjustPtransform();

  isl::map ctransform_with_params();
  isl::set cdomain_with_params();

 private:
  isl::set pdomain_;
  isl::set cdomain_;
  isl::map access_;
  isl::map ptransform_;
  isl::map ctransform_;

  isl::set adjusted_pdomain_;
  isl::map adjusted_ptransform_;
  isl::set adjusted_cdomain_;
  isl::map adjusted_ctransform_;

  int level_;
};

}  // namespace poly
}  // namespace cinn
