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

#include <string>
#include <vector>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {
namespace math {

/*
 * This is an implementation of beam search.
 *
 * To explain the details, lets take machine translation task for example, in
 * this task, one source sentence is translated to multiple target sentences,
 * during this period, one sentence will be translated to multiple translation
 * prefixes(target sentence that have not ended), in each time step a prefix
 * will have some candidates, input the candidate ids and their corresponding
 * scores (probabilities), it will sort and select the top beam_size candidates
 * for each source sentence, and store the selected candidates's score and their
 * corresponding ids to LoDTensors.
 *
 * A detailed example:
 *
 *  Input
 *
 *    ids:
 *      - LoD (should have 2 levels)
 *        - first level: [0, 1, 4]
 *        - second level: [0, 1, 2, 3, 4]
 *      - tensor's data:
 *          [[4, 2, 5]
 *           [2, 1, 3]
 *           [3, 5, 2]
 *           [8, 2, 1]]
 *
 *    scores:
 *      - LoD same as `ids`
 *      - tensor's data
 *          [[0.5, 0.3, 0.2]
 *           [0.6, 0.3, 0.1]
 *           [0.9, 0.5, 0.1]
 *           [0.7, 0.5, 0.1]]
 *
 * The inputs means that there are 2 source sentences to translate, and the
 * first source has 1 prefix, the second source has 2 prefix.
 *
 * Lets assume beam size is 2, and the beam search's output should be
 *      - LoD
 *        - first level: [0, 1, 2]
 *        - second level: [0, 2, 4]
 *      - id tensor's data
 *          [[4,
 *            1,
 *            3,
 *            8]]
 *      - score tensor's data
 *          [[0.5,
 *            0.3,
 *            0.9,
 *            0.7]]
 *
 * TODO all the prune operations should be in the beam search, so it is better
 * to split the beam search algorithm into a sequence of smaller operators, and
 * the prune operators can be inserted in this sequence.
 */
template <typename DeviceContext, typename T>
class BeamSearchFunctor {
 public:
  /*
   * The main function of beam search.
   *
   * @selected_ids: a [None, 1]-shaped tensor with LoD.
   *   In a machine translation model, it might be the candidate term id sets,
   *   each set stored as a varience-length sequence.
   *   The format might be described with a two-level LoD
   *   - [[0 1],
   *      [0 1 2]]
   *   - [[]
   *      [0 1]]
   *   the first level of LoD tells that there are two source sentences. The
   *   second level describes the details of the candidate id set's offsets in
   * the source sentences.
   *
   *  @selected_scores: a LoD tensor with the same shape and LoD with
   * selected_ids.
   *   It stores the corresponding scores of candidate ids in selected_ids.
   *
   * Return false if all the input tensor is empty, in machine translation task
   * that means no candidates is provided, and the task will stop running.
   */
  void operator()(const DeviceContext& context,
                  const framework::LoDTensor* pre_ids,
                  const framework::LoDTensor* pre_scores,
                  const framework::LoDTensor* ids,
                  const framework::LoDTensor* scores,
                  framework::LoDTensor* selected_ids,
                  framework::LoDTensor* selected_scores,
                  phi::DenseTensor* parent_idx,
                  size_t level,
                  size_t beam_size,
                  int end_id,
                  bool is_accumulated);
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
