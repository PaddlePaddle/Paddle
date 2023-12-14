// Copyright (c) 2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

/**
 * Tensor specially designed for KV Cache
 * Naturaly, it could be represented in the shape of [seq_length][batch_size][head_num][head_size]
 * 
 *        |________ bs[0] ________|_______|_______ bs[N-1] _______|
 * seq=0  |       |       |       |  ...  |       |       |       |
 * seq=1  |       |       |       |  ...  |       |       |       |
 * seq=2  | head0 | head1 | head2 |  ...  | head0 | head1 | head2 |
 *  ...   |       |       |       |  ...  |       |       |       |
 *        |       |       |       |  ...  |       |       |       |
 *        `````````````````````````````````````````````````````````
 * For better performance, it can be represented as [batch_size][head_num][seq_length][head_size]
 *        __________________
 *        |       |       ^
 *        |       |       |
 *        | head0 |       |
 *        |       |       |
 *        |       |       |
 *        |_______|       |
 *        |       |       |
 *        |       |       |
 *        | head1 |     bs[0]
 *        |       |       |
 *        |       |       |
 *        |_______|       |
 *        |       |       |
 *        |       |       |
 *        | head2 |       |
 *        |       |       |
 *        |       |       v
 *        ``````````````````
 *              ....
 * Note: The batch size in KVCache can be larger than the batch size in model inference (when beam size > 1)
 * The batch size of model inference is smaller to save the computing
 * The batch size of KV Cache is larger to make the KV cache expanding easier
*/
template <typename T>
class KVCacheTensor {
public:
    KVCacheTensor() : maxSeqLen(0), batchSize(0), headNum(0), headSize(0), data(nullptr), allocSize(0) {}

    ~KVCacheTensor() {
        if (this->data) { free(this->data); }
    }

    void resize(int maxSeqLen, int batchSize, int headNum, int headSize) {
        this->maxSeqLen = maxSeqLen;
        this->batchSize = batchSize;
        this->headNum = headNum;
        this->headSize = headSize;

        uint64_t requiredSize = (uint64_t)maxSeqLen * batchSize * headNum * headSize;
        if (requiredSize > allocSize) {
            this->data = (T *)aligned_alloc(64, requiredSize * sizeof(T));
            if (!this->data) {
                printf("Failed to alloc mem for KV Cache [%d][%d][%d][%d].\n", maxSeqLen, batchSize, headNum, headSize);
                exit(-1);
            }

            allocSize = requiredSize;
        }
    }

    int getBatchSize() const { return batchSize; }
    int getHeadNum() const { return headNum; }
    int getHeadSize() const { return headSize; }

    T *getData() { return data; }

    // Get a vector for a specified sequence
    T *getSequence(int seqIdx, int batchIdx, int headIdx) {
        return data + (seqIdx * batchSize + batchIdx) * (headNum * headSize) + headIdx * headSize;
    }

    // Get a head matrix, return the start address and the stride
    std::pair<T *, int> getHead(int batchIdx, int headIdx) {
        T *addr = data + batchIdx * headNum * headSize + headIdx * headSize;
        return std::make_pair(addr, batchSize * headNum * headSize);
    }

    /**
     * Expand the tensor by broadcasting each sample to multiple beams.
     * It is needed when beam_size > 1 by just passing the unique user side samples to do the inference.
     * For example, when user_side_bs=2, beam_size=3, it will expand:
     *  _______________________________________________
     * |  bs0  |  bs1  |       |       |       |       |
     *  ```````````````````````````````````````````````
     * to
     *  _______________________________________________
     * |  bs0  |  bs0  |  bs0  |  bs1  |  bs1  |  bs1  |
     *  ```````````````````````````````````````````````
    */
    void expandAllSequence(int userSideBS, int beamSize, int seqLen) {
        if (userSideBS * beamSize != batchSize) {
            printf("Cannot expand the KV Cache as userSideBS(%d) * beamSize(%d) != batchSize(%d)\n", userSideBS,
                    beamSize, batchSize);
            return;
        }

#pragma omp parallel for
        for (int seq = 0; seq < seqLen; ++seq) {
            for (int b = batchSize - 1; b > 0; --b) {
                T *dst = getSequence(seq, b, 0);
                T *src = getSequence(seq, b / beamSize, 0);
                memcpy(dst, src, headNum * headSize * sizeof(T));
            }
        }
    }

    void expandOneSequence(int userSideBS, int beamSize, int seq) {
        for (int b = batchSize - 1; b > 0; --b) {
            T *dst = getSequence(seq, b, 0);
            T *src = getSequence(seq, b / beamSize, 0);
            memcpy(dst, src, headNum * headSize * sizeof(T));
        }
    }

    // Below implementation could be a little faster (100.6 vs. 100.9), but also need to modify expand and reorder function

    // // Get a vector for a specified sequence
    // T *getSequence(int seqIdx, int batchIdx, int headIdx) {
    //     return data + batchIdx * headNum * maxSeqLen * headSize + headIdx * maxSeqLen * headSize + seqIdx * headSize;
    // }

    // // Get a head matrix, return the start address and the stride
    // std::pair<T *, int> getHead(int batchIdx, int headIdx) {
    //     T *addr = data + batchIdx * headNum * maxSeqLen * headSize + headIdx * maxSeqLen * headSize;
    //     return std::make_pair(addr, headSize);
    // }

private:
    int maxSeqLen;
    int batchSize;
    int headNum;
    int headSize;

    T *data;
    uint64_t allocSize;
};