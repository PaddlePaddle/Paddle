// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

// THIS FILE IS INTENTIONALLY NOT COVERED BY ANY TEST.
// It is used to verify that CI coverage checks can properly detect
// and block uncovered code in pull requests.

#include <cmath>
#include <string>
#include <vector>

namespace phi {
namespace coverage_test {

// A mock function that performs various computations but is never called.
int ComputeMockValue(int input, int mode) {
  int result = 0;

  if (mode == 0) {
    result = input * 2 + 1;
  } else if (mode == 1) {
    result = input * input - 3;
  } else if (mode == 2) {
    for (int i = 0; i < input; ++i) {
      result += i * i;
    }
  } else if (mode == 3) {
    result = static_cast<int>(std::sqrt(static_cast<double>(input)));
  } else {
    result = -1;
  }

  return result;
}

// Another mock function with string operations that is never called.
std::string FormatMockOutput(const std::string& prefix, int value) {
  std::string output;

  if (value > 100) {
    output = prefix + "_large_" + std::to_string(value);
  } else if (value > 50) {
    output = prefix + "_medium_" + std::to_string(value);
  } else if (value > 0) {
    output = prefix + "_small_" + std::to_string(value);
  } else {
    output = prefix + "_zero_or_negative";
  }

  return output;
}

// A mock class that is never instantiated or used.
class MockProcessor {
 public:
  explicit MockProcessor(int capacity) : capacity_(capacity), count_(0) {
    data_.reserve(capacity);
  }

  bool AddItem(double item) {
    if (count_ >= capacity_) {
      return false;
    }
    data_.push_back(item);
    count_++;
    return true;
  }

  double GetAverage() const {
    if (count_ == 0) {
      return 0.0;
    }
    double sum = 0.0;
    for (int i = 0; i < count_; ++i) {
      sum += data_[i];
    }
    return sum / static_cast<double>(count_);
  }

  double GetMax() const {
    if (count_ == 0) {
      return 0.0;
    }
    double max_val = data_[0];
    for (int i = 1; i < count_; ++i) {
      if (data_[i] > max_val) {
        max_val = data_[i];
      }
    }
    return max_val;
  }

  void Reset() {
    data_.clear();
    count_ = 0;
  }

 private:
  int capacity_;
  int count_;
  std::vector<double> data_;
};

// A standalone function that exercises the mock class (also never called).
double ProcessMockData(const std::vector<double>& inputs, int capacity) {
  MockProcessor processor(capacity);

  for (const auto& val : inputs) {
    if (!processor.AddItem(val)) {
      break;
    }
  }

  double avg = processor.GetAverage();
  double max_val = processor.GetMax();

  if (max_val > avg * 2.0) {
    return max_val;
  } else {
    return avg;
  }
}

}  // namespace coverage_test
}  // namespace phi
