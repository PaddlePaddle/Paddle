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

#include "gtest/gtest.h"

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/operators/reader/blocking_queue.h"
#include "paddle/fluid/operators/reader/ctr_reader.h"

using paddle::operators::reader::LoDTensorBlockingQueue;
using paddle::operators::reader::LoDTensorBlockingQueueHolder;
using paddle::operators::reader::CTRReader;

TEST(CTR_READER, read_data) {
  LoDTensorBlockingQueueHolder queue_holder;
  int capacity = 64;
  queue_holder.InitOnce(capacity, {}, false);

  std::shared_ptr<LoDTensorBlockingQueue> queue = queue_holder.GetQueue();

  int batch_size = 10;
  int thread_num = 1;
  std::vector<std::string> slots = {"6003", "6004"};
  std::vector<std::string> file_list = {
      "/Users/qiaolongfei/project/gzip_test/part-00000-A.gz",
      "/Users/qiaolongfei/project/gzip_test/part-00000-A.gz"};

  CTRReader reader(queue, batch_size, thread_num, slots, file_list);

  reader.Start();
  //
  //  std::vector<LoDTensor> out;
  //  reader.ReadNext(&out);
}
