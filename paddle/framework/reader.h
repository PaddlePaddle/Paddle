//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/framework/ddim.h"
#include "paddle/framework/lod_tensor.h"

namespace paddle {
namespace framework {

class Reader {
 public:
  virtual int ReadNext(std::vector<LoDTensor>* outs) = 0;
  DDim shape(int idx) const;

 private:
  std::vector<DDim> shapes_;
};

// file readers

class RandomReader : public Reader {
 public:
  RandomReader(const std::vector<DDim>& shapes, float min, float max)
      : shapes_(shapes), min_(min), max_(max) {}
  int ReadNext(std::vector<LoDTensor>* outs) override;

 private:
  float min_;
  float max_;
};

// decorators

class BatchReader : public Reader {
 public:
  BatchReader(const Reader* reader) : reader_(reader) {}
  int ReadNext(std::vector<LoDTensor>* outs) override;

 private:
  const Reader* reader_;
};

class ShuffleReader : public Reader {
 public:
  ShuffleReader(const Reader* reader) : reader_(reader) {}
  int ReadNext(std::vector<LoDTensor>* outs) override;

 private:
  const Reader* reader_;
};
}  // namespace framework
}  // namespace paddle
