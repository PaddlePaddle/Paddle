// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/operators/reader/py_reader.h"

TEST(PyReader, EmptyBatchIsNotEOF) {
  auto queue =
      std::make_shared<paddle::operators::reader::DenseTensorBlockingQueue>(2);
  paddle::operators::reader::PyReader reader(queue, {}, {}, {});

  EXPECT_TRUE(queue->Push(phi::TensorArray{}));
  queue->Close();

  phi::TensorArray batch;
  reader.ReadNext(&batch);
  EXPECT_TRUE(batch.empty());
  EXPECT_FALSE(reader.HasReachedEnd());

  reader.ReadNext(&batch);
  EXPECT_TRUE(batch.empty());
  EXPECT_TRUE(reader.HasReachedEnd());
}

TEST(PyReader, StartClearsEOFState) {
  auto queue =
      std::make_shared<paddle::operators::reader::DenseTensorBlockingQueue>(2);
  paddle::operators::reader::PyReader reader(queue, {}, {}, {});

  queue->Close();

  phi::TensorArray batch;
  reader.ReadNext(&batch);
  EXPECT_TRUE(reader.HasReachedEnd());

  reader.Start();
  EXPECT_FALSE(reader.HasReachedEnd());

  EXPECT_TRUE(queue->Push(phi::TensorArray{}));
  queue->Close();

  reader.ReadNext(&batch);
  EXPECT_TRUE(batch.empty());
  EXPECT_FALSE(reader.HasReachedEnd());
}
