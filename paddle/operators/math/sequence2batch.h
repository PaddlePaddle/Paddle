/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/tensor.h"
#include "paddle/platform/device_context.h"

namespace paddle {
namespace operators {
namespace math {

template <typename Place, typename T>
class CopyMatrixRowsFunctor {
 public:
  // If is_src_index is true,
  // copy the indexed rows of input src to the output dst.
  // If is_src_index is false,
  // copy the input src to the indexed rows of output dst.
  // The indexed rows are based on the input index.
  void operator()(const platform::DeviceContext& context,
                  const framework::LoDTensor& src, const size_t* index,
                  framework::LoDTensor& dst, bool is_src_index);
};

template <typename Place, typename T>
class LoDTensor2BatchFunctor {
  // Calculate the length of each sequence and
  // sort sequence index by the length.
  // example:  sequences = {s0, s1, s2}
  //           s0: 0 0 0 0, s1: 1 1 1 1 1, s2: 2 2 2
  //           seq_info[3] = {(4, 5, 1), (0, 4, 0), (9, 3, 2)}
  //
  struct SeqInfo {
    SeqInfo(int start, int length, int seq_idx)
        : start(start), length(length), seq_idx(seq_idx) {}
    int start;
    int length;
    int seq_idx;
  };

 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::LoDTensor& lod_tensor,
                  framework::LoDTensor& batch, bool is_cal_batch_lod,
                  bool is_reverse = false) const {
    if (!is_cal_batch_lod) {
      auto lods = batch.lod();
      PADDLE_ENFORCE_EQ(lods.size(), 2UL);
      PADDLE_ENFORCE_EQ(lods[1].size(),
                        static_cast<size_t>(lod_tensor.dims()[0]));
      CopyMatrixRowsFunctor<Place, T> to_batch;
      to_batch(context, lod_tensor, lods[1].data(), batch, true);
      return;
    }

    auto lods = lod_tensor.lod();
    PADDLE_ENFORCE_EQ(lods.size(), 1UL, "Only support one level sequence now.");
    auto lod = lods[0];

    std::vector<SeqInfo> seq_info;
    for (size_t seq_id = 0; seq_id < lod.size() - 1; ++seq_id) {
      int length = lod[seq_id + 1] - lod[seq_id];
      seq_info.emplace_back(lod[seq_id], length, seq_id);
    }

    std::sort(seq_info.begin(), seq_info.end(),
              [](SeqInfo a, SeqInfo b) { return a.length > b.length; });

    // calculate the start position of each batch
    // (numBatch equal the maxLength of sequences)
    // example:  sequences = {s0, s1, s2}
    //           s0: 0 0 0 0, s1: 1 1 1 1 1, s2: 2 2 2
    //           num_batch = 5,
    //           batchIndex = {b0, b1, b2, b3, b4}
    //           b0: 1 0 2, b1: 1 0 2, b2: 1 0 2, b3: 1 0, b4: 1
    //           batch_start_positions[6] = {0, 3, 6, 9, 11, 12}
    //              batch_start_positions[0] = len(b0)
    //              batch_start_positions[1] = len(b0) + len(b1)
    //              batch_start_positions[2] = len(b0) + len(b1) + len(b2)
    //              ...
    //           seq2batch_idx[12] = {4, 0, 9,
    //                                5, 1, 10,
    //                                6, 2, 11,
    //                                7, 3,
    //                                8}
    // The batch number represents batch size after rearranging the
    // input LodTensor. It is also the maximum length of input sequence.

    paddle::framework::LoD batch_lods;
    batch_lods.emplace_back(std::vector<size_t>{0});
    batch_lods.emplace_back(std::vector<size_t>{0});

    // batch_lods[0] is the start positions for batch LoDTensor
    int num_batch = seq_info[0].length;
    batch_lods[0].resize(static_cast<size_t>(num_batch + 1));
    // batch_lods[1] is the raw index in the input LoDTensor
    auto dims = lod_tensor.dims();
    batch_lods[1].resize(static_cast<size_t>(dims[0]));

    size_t* batch_starts = batch_lods[0].data();
    size_t* seq2batch_idx = batch_lods[1].data();
    batch_starts[0] = 0;
    for (int n = 0; n < num_batch; n++) {
      auto batch_id = static_cast<int>(batch_starts[n]);
      for (size_t i = 0; i < seq_info.size(); ++i) {
        int seq_len = seq_info[i].length;
        int start = seq_info[i].start;
        if (n < seq_len) {
          seq2batch_idx[batch_id] =
              is_reverse ? start + seq_len - 1 - n : start + n;
          batch_id++;
        } else {
          break;
        }
      }
      batch_starts[n + 1] = static_cast<size_t>(batch_id);
    }
    batch.set_lod(batch_lods);

    CopyMatrixRowsFunctor<Place, T> to_batch;
    to_batch(context, lod_tensor, seq2batch_idx, batch, true);
  }
};

template <typename Place, typename T>
class Batch2LoDTensorFunctor {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::LoDTensor& batch,
                  framework::LoDTensor& lod_tensor) const {
    auto in_lod = batch.lod();
    PADDLE_ENFORCE_EQ(in_lod.size(), 2UL,
                      "The LoD size of input `batch` should be 2.");
    PADDLE_ENFORCE_EQ(in_lod[1].size(),
                      static_cast<size_t>(lod_tensor.dims()[0]));
    CopyMatrixRowsFunctor<Place, T> to_seq;
    size_t* index = in_lod[1].data();
    to_seq(context, batch, index, lod_tensor, false);
  }
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
