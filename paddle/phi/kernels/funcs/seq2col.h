// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

namespace phi {
namespace funcs {

template <typename T>
struct Seq2ColFunctor {
  Seq2ColFunctor(const T* seq,
                 T* col,
                 size_t seq_length,
                 size_t frame_length,
                 size_t n_frames,
                 size_t hop_length)
      : seq_(seq),
        col_(col),
        seq_length_(seq_length),
        frame_length_(frame_length),
        n_frames_(n_frames),
        hop_length_(hop_length) {}

  /*
    Convert sequences to frames.

    1. Dimension information:

       Sequences                   Frames
    (N, seq_length)  ->  (N, frame_length, n_frames)

    2. Mapping from `i` to  `src_idx` and `trg_idx` can be derived from:

      a. Notion
        - `i` stands for the flattened index of a bunch of frames.
        - `src_idx` and `trg_idx` are the 1D indices of seqs and frames
          respectively.

      b. Sample idx
        ```cpp
        sample_idx = i / (n_frames_ * frame_length_);
        ```

      c. Maps `i` to `f` and `n`.
        ```cpp
        f = i % (n_frames_ * frame_length_) / n_frames_;
        n = i % (n_frames_ * frame_length_) % n_frames_;
        ```

      d. Replace `sample_idx`, `f` and `n` in the following equations:
        ```cpp
        src_idx = sample_idx * seq_length_ + n * hop_length_ + f;
        trg_idx = sample_idx * n_frames_ * frame_length_ + f * n_frames_ + n;
        col_[trg_idx] = seq_[src_idx];
        ```

      e. Result can be deduced shown in the function body below.
  */
  HOSTDEVICE void operator()(size_t i) const {
    size_t src_idx;
    size_t trg_idx;
    src_idx = i / (n_frames_ * frame_length_) * seq_length_ +
              i % (n_frames_ * frame_length_) % n_frames_ * hop_length_ +
              i % (n_frames_ * frame_length_) / n_frames_;
    trg_idx = i / (n_frames_ * frame_length_) * n_frames_ * frame_length_ +
              i % (n_frames_ * frame_length_) / n_frames_ * n_frames_ +
              i % (n_frames_ * frame_length_) % n_frames_;
    col_[trg_idx] = seq_[src_idx];
  }

  const T* seq_;
  T* col_;
  size_t seq_length_;
  size_t frame_length_;
  size_t n_frames_;
  size_t hop_length_;
};

template <typename T>
struct Col2SeqFunctor {
  Col2SeqFunctor(const T* col,
                 T* seq,
                 size_t seq_length,
                 size_t frame_length,
                 size_t n_frames,
                 size_t hop_length)
      : col_(col),
        seq_(seq),
        seq_length_(seq_length),
        frame_length_(frame_length),
        n_frames_(n_frames),
        hop_length_(hop_length) {}

  /*
    Accumulate output gradient d_out to d_x.

    1. Dimension information:

              d_out                        d_x
    (N, frame_length, n_frames)  ->  (N, seq_length)

    2. Using a sliding window to find source indices from `d_out` according to
       `i`:

      a. Notion
        - `i` stands for the flattened index of `d_x`.
        - `seq_i` stands for a relative index of a `d_x` sample.
        - `left`: Starting index of a frame window.
        - `right`: Ending index of a frame window.

      b. Sample idx
        ```cpp
        sample_idx = i / seq_length_;
        ```

      c. Slides a window with length of `frame_length` to find `f` and `n`.
        - `n`: The idx of num_frames_, increases in each hop.
        - `f`: The idx of frame_lengths_, relative idx from left of a sliding
               window.

      d. Accumulate all grads from d_out.
        ```cpp
        seq_[i] +=
            col_[sample_idx * frame_length_ * n_frames_ + f * n_frames_ + n];
        ```
  */
  HOSTDEVICE void operator()(size_t i) const {
    size_t sample_idx = i / seq_length_;
    size_t seq_i = i % seq_length_;

    // Sliding window
    seq_[i] = 0;  // Init seq_[i] to 0, and sums up all
                  // grads from col_ in the while loop.

    size_t n = get_start_frame_idx(seq_i);
    size_t f;
    size_t left = n * hop_length_;
    size_t right = left + frame_length_ - 1;

    while (left <= seq_i && right < seq_length_) {
      f = seq_i - left;
      seq_[i] +=
          col_[sample_idx * frame_length_ * n_frames_ + f * n_frames_ + n];
      // Next frame.
      left += hop_length_;
      right += hop_length_;
      n += 1;
    }
  }

  /*
    Calculate minimum value of frame index `n` to satisfy the inequality:

      seq_i <= right
      ==> seq_i <= left + frame_length - 1
      ==> seq_i <= hop_length_ * n + frame_length_ - 1
  */
  HOSTDEVICE size_t get_start_frame_idx(size_t seq_i) const {
    int64_t tmp = seq_i + 1 - frame_length_;
    if (tmp > 0) {
      size_t n = tmp / hop_length_;
      if (tmp % hop_length_ == 0) {
        return n;
      } else {
        return n + 1;
      }
    } else {
      return 0;
    }
  }

  const T* col_;
  T* seq_;
  size_t seq_length_;
  size_t frame_length_;
  size_t n_frames_;
  size_t hop_length_;
};

}  // namespace funcs
}  // namespace phi
