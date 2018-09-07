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

#ifndef PADDLE_NO_PYTHON
#include <gtest/gtest.h>
#include <fstream>
#include "paddle/gserver/dataproviders/DataProvider.h"
#include "paddle/utils/PythonUtil.h"
#include "paddle/utils/Util.h"

DEFINE_string(train_list, "unittest.list", "file list for unittest");

namespace paddle {
namespace unittest {
namespace pydp2 {
extern void setOnPoolFilledHook(const std::function<void(size_t)> &func);
extern void clearOnPoolFilledHook();

}  // namespace pydp2
}  // namespace unittest
}  // namespace paddle

const paddle::real epsilon = 1e-5;

static inline int64_t readDataBatch(paddle::DataBatch *batch,
                                    const std::string &funcName,
                                    int64_t batchSize = 65535) {
  paddle::DataConfig config;
  config.set_type("py2");
  config.set_files(FLAGS_train_list.c_str());
  config.set_load_data_module("test_PyDataProvider2");
  config.set_load_data_object(funcName);
  std::unique_ptr<paddle::DataProvider> provider(
      paddle::DataProvider::create(config, false));
  provider->setSkipShuffle();
  provider->reset();
  return provider->getNextBatchInternal(batchSize, batch);
}

TEST(PyDataProvider2, dense_no_seq) {
  paddle::DataConfig config;
  config.set_type("py2");
  config.set_files(FLAGS_train_list.c_str());
  config.set_load_data_module("test_PyDataProvider2");
  config.set_load_data_object("test_dense_no_seq");

  std::unique_ptr<paddle::DataProvider> provider(
      paddle::DataProvider::create(config, false));

  provider->setSkipShuffle();  // skip shuffle for unittest.

  paddle::DataBatch batch;
  for (size_t pass = 0; pass < 2; ++pass) {  // read 2 passes
    provider->reset();
    int64_t num = provider->getNextBatchInternal(100, &batch);
    ASSERT_NE(num, 0);
    ASSERT_EQ((size_t)batch.getStreams().size(), (size_t)1);
    ASSERT_EQ((size_t)batch.getSize(), (size_t)100);
    // Check batch data.
    for (size_t i = 0; i < 100; ++i) {
      for (size_t j = 0; j < 200; ++j) {
        paddle::real tmp = (paddle::real)((j - 100.0) * (i + 1) / 200.0);
        ASSERT_NEAR(
            batch.getStreams()[0].value->getData()[i * 200 + j], tmp, epsilon);
      }
    }

    num = provider->getNextBatchInternal(100, &batch);
    ASSERT_NE(num, 0);
    ASSERT_EQ(batch.getStreams().size(), (size_t)1);
    ASSERT_EQ((size_t)batch.getSize(), (size_t)100);
    // Check batch data.
    for (size_t i = 0; i < 100; ++i) {
      size_t ii = i + 100;
      for (size_t j = 0; j < 200; ++j) {
        paddle::real tmp = (paddle::real)((j - 100.0) * (ii + 1) / 200.0);
        ASSERT_NEAR(
            batch.getStreams()[0].value->getData()[i * 200 + j], tmp, epsilon);
      }
    }
    num = provider->getNextBatchInternal(100, &batch);
    ASSERT_EQ(num, 0);
  }
}

TEST(PyDataProvider2, index_no_seq) {
  paddle::DataConfig config;
  config.set_type("py2");
  config.set_files(FLAGS_train_list.c_str());
  config.set_load_data_module("test_PyDataProvider2");
  config.set_load_data_object("test_index_no_seq");
  std::unique_ptr<paddle::DataProvider> provider(
      paddle::DataProvider::create(config, false));

  provider->setSkipShuffle();  // skip shuffle for unittest.
  paddle::DataBatch batch;
  for (size_t pass = 0; pass < 2; ++pass) {
    provider->reset();
    int64_t num = provider->getNextBatchInternal(10000, &batch);
    CHECK_EQ(num, 200);
    for (int i = 0; i < 200; ++i) {
      CHECK_EQ(i, batch.getStreams()[0].ids->getData()[i]);
    }
  }
}

TEST(PyDataProvider2, init_hook) {
  paddle::PyObjectPtr pickle = paddle::py::import("pickle");
  paddle::PyObjectPtr globals(PyModule_GetDict(PyImport_AddModule("__main__")));
  PyDict_SetItemString(globals.get(), "pickle", pickle.get());
  paddle::PyObjectPtr locals(PyDict_New());
  paddle::PyObjectPtr mdl(PyRun_String(
      "dumps = pickle.dumps({'value':[float(x) for x in xrange(20)]})",
      Py_file_input,
      globals.get(),
      locals.get()));
  CHECK_PY(mdl) << "Error!";
  paddle::PyObjectPtr dps(PyDict_GetItemString(locals.get(), "dumps"));
  CHECK_PY(dps) << "Error!";

  paddle::DataConfig config;
  config.set_type("py2");
  config.set_files(FLAGS_train_list.c_str());
  config.set_load_data_module("test_PyDataProvider2");
  config.set_load_data_object("test_init_hook");
  config.set_load_data_args(PyString_AsString(dps.get()));

  std::unique_ptr<paddle::DataProvider> provider(
      paddle::DataProvider::create(config, false));
  provider->setSkipShuffle();  // skip shuffle for unittest.
  provider->reset();
  paddle::DataBatch batch;
  int64_t num = provider->getNextBatchInternal(100000, &batch);
  ASSERT_EQ(num, 200);
  auto &mat = batch.getStreams()[0].value;
  ASSERT_EQ((size_t)mat->getWidth(), (size_t)20);
  for (size_t i = 0; i < 200; ++i) {
    for (size_t j = 0; j < 20; ++j) {
      ASSERT_NEAR((paddle::real)j, mat->getData()[i * 20 + j], epsilon);
    }
  }
}

TEST(PyDataProvider2, sparse_no_value_no_seq) {
  paddle::DataConfig config;
  config.set_type("py2");
  config.set_files(FLAGS_train_list.c_str());
  config.set_load_data_module("test_PyDataProvider2");
  config.set_load_data_object("test_sparse_non_value_no_seq");
  std::unique_ptr<paddle::DataProvider> provider(
      paddle::DataProvider::create(config, false));
  provider->setSkipShuffle();
  provider->reset();
  paddle::DataBatch batch;
  int64_t num = provider->getNextBatchInternal(10000, &batch);
  CHECK_EQ(num, 200);
  auto csm = std::dynamic_pointer_cast<paddle::CpuSparseMatrix>(
      batch.getStreams()[0].value);
  CHECK(csm != nullptr);
  for (int i = 0; i < 200; ++i) {
    CHECK_EQ(csm->getColNum(i), (size_t)10);
    int *cols = csm->getRowCols(i);
    for (int j = 0; j < 10; ++j) {
      CHECK_EQ(cols[j], (i + 1) * (j + 1));
    }
  }
}

TEST(PyDataProvider2, sparse_value_no_seq) {
  paddle::DataBatch batch;
  CHECK_EQ(readDataBatch(&batch, "test_sparse_value_no_seq"), 200);
  auto csm = std::dynamic_pointer_cast<paddle::CpuSparseMatrix>(
      batch.getStreams()[0].value);
  CHECK(csm != nullptr);
  for (int i = 0; i < 200; ++i) {
    CHECK_EQ(csm->getColNum(i), (size_t)10);
    int *cols = csm->getRowCols(i);
    real *dat = csm->getRowValues(i);
    for (int j = 0; j < 10; ++j) {
      EXPECT_EQ(cols[j], (i + 1) * (j + 1));
      EXPECT_EQ(dat[j], real(j) / real(i + 1));
    }
  }
}

TEST(PyDataProvider2, index_seq) {
  paddle::DataBatch batch;
  CHECK_EQ(readDataBatch(&batch, "test_index_seq"), 200);
  auto &arg = batch.getStreams()[0];
  CHECK_EQ((int)arg.ids->getSize(), (200 + 1) * 200 / 2);
  size_t tmp = 0;
  for (size_t i = 0; i < 200; ++i) {  // CHECK DATA CORRECT
    for (size_t j = 0; j < i + 1; ++j) {
      ASSERT_EQ((size_t)arg.ids->getData()[tmp], j);
      ++tmp;
    }
  }
  ASSERT_EQ(arg.sequenceStartPositions->getSize(), (size_t)201);
  tmp = 0;
  for (size_t i = 0; i < 200; ++i) {
    tmp += i;
    ASSERT_EQ((size_t)arg.sequenceStartPositions->getData(false)[i], tmp);
  }
  tmp += 200;
  ASSERT_EQ((size_t)arg.sequenceStartPositions->getData(false)[200], tmp);
}

TEST(PyDataProvider2, index_sub_seq) {
  paddle::DataBatch batch;
  ASSERT_EQ(readDataBatch(&batch, "test_index_sub_seq"), 200);
  auto &arg = batch.getStreams()[0];
  size_t tmp = 0;
  for (size_t i = 0; i < 200; ++i) {
    for (size_t j = 0; j < i + 1; ++j) {
      for (size_t k = 0; k < j + 1; ++k) {
        CHECK_EQ((size_t)arg.ids->getData()[tmp++], k);
      }
    }
  }

  CHECK_EQ(tmp, arg.ids->getSize());

  ASSERT_EQ((size_t)arg.sequenceStartPositions->getSize(), (size_t)201);
  ASSERT_EQ(arg.subSequenceStartPositions->getData(false)[0], 0);
  ASSERT_EQ(arg.sequenceStartPositions->getData(false)[0], 0);
  size_t idx = 1;
  tmp = 0;
  for (size_t i = 0; i < 200; ++i) {
    for (size_t j = 0; j < i + 1; ++j) {
      tmp += j + 1;
      ASSERT_EQ((size_t)arg.subSequenceStartPositions->getData(false)[idx],
                (size_t)tmp);
      ++idx;
    }
    ASSERT_EQ((size_t)arg.sequenceStartPositions->getData(false)[i + 1], tmp);
  }
}

TEST(PyDataProvider2, min_pool_size) {
  paddle::DataConfig config;
  config.set_type("py2");
  config.set_files(FLAGS_train_list.c_str());
  config.set_load_data_module("test_PyDataProvider2");
  config.set_load_data_object("test_min_pool_size");
  config.set_load_data_args("");
  size_t totalData = 1 << 14;
  constexpr size_t batchSize = 100;
  constexpr size_t minPoolSize = 1000;
  paddle::DataBatch batch;
  std::unique_ptr<paddle::DataProvider> provider(
      paddle::DataProvider::create(config, false));
  provider->reset();

  paddle::unittest::pydp2::setOnPoolFilledHook([&](size_t poolSize) {
    if (totalData > batchSize) {
      CHECK_GE(poolSize, std::min(totalData - batchSize, minPoolSize));
    }
  });
  while (true) {
    int64_t realBatchSize = provider->getNextBatchInternal(batchSize, &batch);
    if (realBatchSize) {
      totalData -= realBatchSize;
    } else {
      break;
    }
  }
  paddle::unittest::pydp2::clearOnPoolFilledHook();
}

TEST(PyDataProvider2, can_over_batch_size) {
  paddle::DataConfig config;
  config.set_type("py2");
  config.set_files(FLAGS_train_list.c_str());
  config.set_load_data_module("test_PyDataProvider2");
  config.set_load_data_object("test_can_over_batch_size");
  config.set_load_data_args("");
  paddle::DataBatch batch;
  std::unique_ptr<paddle::DataProvider> provider(
      paddle::DataProvider::create(config, false));
  provider->reset();
  constexpr size_t batchSize = 100;
  while (true) {
    int64_t realBatchSize = provider->getNextBatchInternal(batchSize, &batch);
    if (realBatchSize) {
      CHECK_LE(static_cast<size_t>(realBatchSize), batchSize);
    } else {
      break;
    }
  }
}

TEST(PyDataProvider2, input_order) {
  paddle::DataConfig config;
  config.set_type("py2");
  config.set_files(FLAGS_train_list.c_str());
  config.set_load_data_module("test_PyDataProvider2");
  config.set_load_data_object("test_input_order");
  config.set_load_data_args("");

  paddle::ModelConfig modelConfig;
  *modelConfig.add_input_layer_names() = "input1";
  *modelConfig.add_input_layer_names() = "input2";
  paddle::DataBatch batch;
  std::unique_ptr<paddle::DataProvider> provider(
      paddle::DataProvider::create(config, modelConfig, false));
  provider->reset();
  constexpr size_t batchSize = 100;
  while (true) {
    int64_t realBatchSize = provider->getNextBatchInternal(batchSize, &batch);
    if (!realBatchSize) {
      break;
    }
    ASSERT_EQ(batch.getStreams().size(), static_cast<size_t>(2));
    for (int64_t i = 0; i < realBatchSize; ++i) {
      ASSERT_EQ(batch.getStream(0).ids->getData()[i], 0);
      ASSERT_EQ(batch.getStream(1).ids->getData()[i], 1);
    }
  }
}

TEST(PyDataProvider2, test_check) {
  paddle::DataConfig config;
  config.set_type("py2");
  config.set_files(FLAGS_train_list.c_str());
  config.set_load_data_module("test_PyDataProvider2");
  config.set_load_data_object("test_check");
  config.set_load_data_args("");
  paddle::DataBatch batch;
  std::unique_ptr<paddle::DataProvider> provider(
      paddle::DataProvider::create(config, false));
  provider->reset();
  while (true) {
    int64_t realBatchSize = provider->getNextBatchInternal(100, &batch);
    if (!realBatchSize) {
      break;
    } else {
      auto &ivec = batch.getStream(0).ids;
      for (size_t i = 0; i < ivec->getSize(); ++i) {
        CHECK_LT(ivec->getData()[i], 10);
      }
    }
  }
}

TEST(PyDataProvider2, multiThread) {
  paddle::DataConfig config;
  config.set_type("py2");
  config.set_files(FLAGS_train_list.c_str());
  config.set_load_data_module("test_PyDataProvider2");
  config.set_load_data_object("test_dense_no_seq");
  config.set_async_load_data(true);

  std::unique_ptr<paddle::DataProvider> provider(
      paddle::DataProvider::create(config, false));
  provider->reset();
  paddle::DataBatch batch;
  provider->getNextBatch(100, &batch);
  provider->reset();
  provider.reset();
}

TEST(PyDataProvider2, minPoolSizeWithCache) {
  paddle::DataConfig config;
  config.set_type("py2");
  config.set_files(FLAGS_train_list.c_str());
  config.set_load_data_module("test_PyDataProvider2");
  config.set_load_data_object("test_min_pool_size_with_cache");
  config.set_async_load_data(true);

  std::unique_ptr<paddle::DataProvider> provider(
      paddle::DataProvider::create(config, false));

  paddle::DataBatch batch;

  for (int i = 0; i < 10; ++i) {
    provider->reset();
    int64_t sum = 0;
    while (int64_t actualNum = provider->getNextBatch(100, &batch)) {
      sum += actualNum;
    }
    ASSERT_EQ(1 << 20, sum);
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  paddle::initMain(argc, argv);
  paddle::initPython(argc, argv);

  std::ofstream fout(FLAGS_train_list);
  CHECK(fout.is_open());
  fout << "stub file name" << std::endl;  // in unittest, filename is not used.
  fout.close();

  return RUN_ALL_TESTS();
}

#endif
