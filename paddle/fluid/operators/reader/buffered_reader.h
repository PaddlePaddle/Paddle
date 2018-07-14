// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <list>
#include <vector>
#include "ThreadPool.h"
#include "paddle/fluid/framework/reader.h"

namespace paddle {
namespace operators {
namespace reader {

class BufferedReader : public framework::DecoratedReader {
  using TensorVec = std::vector<framework::LoDTensor>;
  using VecFuture = std::future<TensorVec>;

 public:
  BufferedReader(const std::shared_ptr<framework::ReaderBase>& reader,
                 const platform::Place& place, size_t buffer_size);

  ~BufferedReader() override;

 private:
  void AppendFutureToBatchSize();

  void AppendFuture();

 protected:
  void ShutdownImpl() override;
  void StartImpl() override;
  void ReadNextImpl(std::vector<framework::LoDTensor>* out) override;

 private:
  ThreadPool thread_pool_;
  platform::Place place_;
  const size_t buffer_size_;
  std::list<VecFuture> buffer_;
};

}  // namespace reader
}  // namespace operators
}  // namespace paddle
