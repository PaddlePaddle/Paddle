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

#include "MultinomialSampler.h"

namespace paddle {

MultinomialSampler::MultinomialSampler(const real* prob, int size)
    : rand_(0.0, size) {
  intervals_.resize(size + 1);
  double sum = 0;
  for (int i = 0; i < size; ++i) {
    sum += prob[i];
  }

  double intervalLength = sum / size;
  double s = 1 / intervalLength;
  for (int i = 0; i < size; ++i) {
    intervals_[i] = {i, (real)(prob[i] * s)};
  }

  auto nextSmallPos = [&](int pos) {
    while (pos < size &&
           (pos != intervals_[pos].otherId || intervals_[pos].thresh >= 1)) {
      ++pos;
    }
    return pos;
  };

  auto nextBigPos = [&](int pos) {
    while (pos < size && intervals_[pos].thresh < 1) {
      ++pos;
    }
    return pos;
  };

  int smallPos = nextSmallPos(0);
  int bigPos = nextBigPos(0);

  auto fillIntervals = [&]() {
    while (bigPos < size) {
      while (intervals_[bigPos].thresh > 1 && smallPos < size) {
        intervals_[smallPos].otherId = bigPos;
        intervals_[bigPos].thresh -= 1 - intervals_[smallPos].thresh;
        smallPos = nextSmallPos(smallPos + 1);
      }
      if (smallPos >= size) break;
      bigPos = nextBigPos(bigPos + 1);
      // If intervals_[bigPos].thresh < 1, it becomes a small interval
    }
  };

  fillIntervals();

  smallPos = nextSmallPos(0);

  // At this point there is no small intervals after bigPos. And this condition
  // will remain true during the next fillIntervals()

  fillIntervals();

  // Handle the inaccuracy caused by finite-precision arithmetic which
  // may results in some unprocessed small or big intervals at this point.
  for (int i = 0; i < size; ++i) {
    if (intervals_[i].otherId == i) {
      intervals_[i].thresh = 1;
    }
  }

  // The last one is to safeguard the case that the random number is equal
  // to size
  intervals_[size] = {size - 1, 1};
}

}  // namespace paddle
