/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */


#ifndef HL_SEQUENCE_H_
#define HL_SEQUENCE_H_

#include "hl_base.h"

/**
 * @brief   Maximum sequence forward.
 *
 * @param[in]   input           each sequence contains some instances.
 * @param[in]   sequence        sequence index..
 * @param[out]  output          max instance in this sequence.
 * @param[out]  index           index of max instance.
 * @param[in]   numSequences    size of sequence[in].
 * @param[in]   dim             input dimension.
 *
 */
extern void hl_max_sequence_forward(real* input,
                                    const int* sequence,
                                    real* output,
                                    int *index,
                                    int numSequences,
                                    int dim);

/**
 * @brief   Maximum sequence backward.
 *
 * @param[in]   outputGrad      output gradient.
 * @param[in]   index           index of max instance.
 * @param[out]  inputGrad       input gradient.
 * @param[in]   numSequences    size of sequence[in].
 * @param[in]   dim             input dimension.
 *
 */
extern void hl_max_sequence_backward(real* outputGrad,
                                     int *index,
                                     real* inputGrad,
                                     int numSequences,
                                     int dim);

/**
 * @brief   Context projection forward.
 *
 * @param[in]   input           input sequence.
 * @param[in]   sequence        sequence index.
 * @param[in]   weightData      padding data.
 * @param[out]  output          output sequence.
 * @param[in]   numSequences    number of sequences.
 * @param[in]   inputDim        input sequence dimension.
 * @param[in]   contextLength   context length.
 * @param[in]   contextStart    context start.
 * @param[in]   beginPad        number of extra timesteps added at the beginning.
 * @param[in]   isPadding       trainable padding.
 *
 */
extern void hl_context_projection_forward(real* input,
                                          const int* sequence,
                                          real* weightData,
                                          real* output,
                                          int numSequences,
                                          int inputDim,
                                          int contextLength,
                                          int contextStart,
                                          int beginPad,
                                          bool isPadding);

/**
 * @brief   Context projection backward data.
 *
 * @param[in]   outputGrad      output gradient.
 * @param[in]   sequence        sequence index.
 * @param[out]  inputGrad       input gradient.
 * @param[in]   numSequences    number of sequences.
 * @param[in]   inputDim        input sequence dimension.
 * @param[in]   contextLength   context length.
 * @param[in]   contextStart    context start.
 *
 */
extern void hl_context_projection_backward_data(real* outputGrad,
                                                const int* sequence,
                                                real* inputGrad,
                                                int numSequences,
                                                int inputDim,
                                                int contextLength,
                                                int contextStart);

/**
 * @brief   Context projection backward weight.
 *
 * @param[in]   outputGrad      output gradient.
 * @param[in]   sequence        sequence index.
 * @param[out]  weightGrad      weight gradient.
 * @param[in]   numSequences    number of sequences.
 * @param[in]   weightDim       input sequence dimension.
 * @param[in]   totalPad        number of extra timesteps.
 * @param[in]   contextLength   context length.
 * @param[in]   contextStart    context start.
 * @param[in]   beginPad        number of extra timesteps added at the beginning.
 *
 */
extern void hl_context_projection_backward_weight(real* outputGrad,
                                                  const int* sequence,
                                                  real* weightGrad,
                                                  int numSequences,
                                                  int weightDim,
                                                  int totalPad,
                                                  int contextLength,
                                                  int contextStart,
                                                  int beginPad);

/**
 * @brief   Memory copy from sequence to batch.
 *
 * if seq2batch == true
 *
 *    copy from sequence to batch: batch[i] = sequence[batchIndex[i]].
 *
 * if seq2batch == false
 *
 *    copy from batch to sequence: sequence[batchIndex[i]] = batch[i].
 *
 * @param[in,out]   batch       batch matrix.
 * @param[in,out]   sequence    equence matrix.
 * @param[in]       batchIndex  index vector.
 * @param[in]       seqWidth    width of sequence.
 * @param[in]       batchCount  number of batchIndex.
 * @param[in]       seq2batch   copy direction.
 *
 */
extern void hl_sequence2batch_copy(real *batch,
                                   real *sequence,
                                   const int *batchIndex,
                                   int seqWidth,
                                   int batchCount,
                                   bool seq2batch);

/**
 * @brief   Add sequence to batch.
 *
 * if seq2batch == true
 *
 *    add sequence to batch: batch[i] = sequence[batchIndex[i]].
 *
 * if seq2batch == false
 *
 *    add batch to sequence: sequence[batchIndex[i]] = batch[i].
 *
 * @param[in,out]   batch       batch matrix.
 * @param[in,out]   sequence    equence matrix.
 * @param[in]       batchIndex  index vector.
 * @param[in]       seqWidth    width of sequence.
 * @param[in]       batchCount  number of batchIndex.
 * @param[in]       seq2batch   copy direction.
 *
 */
extern void hl_sequence2batch_add(real *batch,
                                  real *sequence,
                                  int *batchIndex,
                                  int seqWidth,
                                  int batchCount,
                                  bool seq2batch);

/**
 * @brief  dst = Op(src), src is sequence.
 *
 * mode = 0, Op is average.
 *
 * mode = 1, Op is sum.
 *
 * mode = 2, Op is sum(src)/sqrt(N), N is sequence length.
 *
 * @param[in,out]   dst       destination data.
 * @param[in]       src       source data.
 * @param[in]       starts    sequence start positions.
 * @param[in]       height    height of dst data.
 * @param[in]       width     width of dst data.
 * @param[in]       mode      0: avreage,
 *                            1: sum,
 *                            2: divide by square root
 *                            of sequenceLength
 */
extern void hl_sequence_avg_forward(real* dst,
                                    real* src,
                                    const int* starts,
                                    int height,
                                    int width,
                                    const int mode);

#endif /* HL_SEQUENCE_H_ */
