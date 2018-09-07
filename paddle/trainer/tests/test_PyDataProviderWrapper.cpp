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

#ifndef PADDLE_NO_PYTHON
#include <DataConfig.pb.h>
#include <gtest/gtest.h>
#include <paddle/gserver/dataproviders/DataProvider.h>
#include <paddle/math/Matrix.h>
#include <paddle/parameter/Argument.h>
#include <paddle/utils/PythonUtil.h>
#include <fstream>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include "picojson.h"

void checkValue(std::vector<paddle::Argument>& arguments, picojson::array& arr);
const std::string kDir = "./trainer/tests/pydata_provider_wrapper_dir/";

TEST(PyDataProviderWrapper, SequenceData) {
  paddle::DataConfig conf;
  conf.set_type("py");
  conf.set_load_data_module("testPyDataWrapper");
  conf.set_load_data_object("processSeqAndGenerateData");
  conf.set_load_data_args(kDir + "test_pydata_provider_wrapper.json");
  conf.clear_files();
  conf.set_files(kDir + "test_pydata_provider_wrapper.list");
  paddle::DataProviderPtr provider(paddle::DataProvider::create(conf, false));
  provider->setSkipShuffle();
  provider->reset();
  paddle::DataBatch batchFromPy;
  provider->getNextBatch(100, &batchFromPy);

  picojson::value val;
  std::fstream fin;
  fin.open(kDir + "test_pydata_provider_wrapper.json", std::ios_base::in);
  EXPECT_TRUE(fin.is_open());
  if (fin.is_open()) {
    std::string err = picojson::parse(val, fin);
    EXPECT_TRUE(err.empty());
    EXPECT_TRUE(val.is<picojson::array>());
    picojson::array& arr = val.get<picojson::array>();
    std::vector<paddle::Argument>& arguments = batchFromPy.getStreams();
    // CHECK Value
    checkValue(arguments, arr);
    // CHECK sequenceStartPositions
    for (size_t i = 0; i < arr.size(); i++) {
      int row_id = arr[i].get<picojson::array>().size();
      EXPECT_EQ(0, arguments[i].sequenceStartPositions->getData(false)[0]);
      EXPECT_EQ((int)row_id,
                arguments[i].sequenceStartPositions->getData(false)[1]);
    }
    fin.close();
  }
}

TEST(PyDataProviderWrapper, HasSubSequenceData) {
  paddle::DataConfig conf;
  conf.set_type("py");
  conf.set_load_data_module("testPyDataWrapper");
  conf.set_load_data_object("processSubSeqAndGenerateData");
  conf.set_load_data_args(kDir + "test_pydata_provider_wrapper.json");
  conf.clear_files();
  conf.set_files(kDir + "test_pydata_provider_wrapper.list");
  paddle::DataProviderPtr provider(paddle::DataProvider::create(conf, false));
  provider->setSkipShuffle();
  provider->reset();
  paddle::DataBatch batchFromPy;
  provider->getNextBatch(1, &batchFromPy);

  picojson::value val;
  std::fstream fin;
  fin.open(kDir + "test_pydata_provider_wrapper.json", std::ios_base::in);
  EXPECT_TRUE(fin.is_open());
  if (fin.is_open()) {
    std::string err = picojson::parse(val, fin);
    EXPECT_TRUE(err.empty());
    EXPECT_TRUE(val.is<picojson::array>());
    picojson::array& arr = val.get<picojson::array>();
    std::vector<paddle::Argument>& arguments = batchFromPy.getStreams();
    // CHECK Value
    checkValue(arguments, arr);
    // CHECK sequenceStartPositions and subSequenceStartPositions
    for (size_t i = 0; i < arr.size(); i++) {
      int row_id = arr[i].get<picojson::array>().size();
      EXPECT_EQ(0, arguments[i].sequenceStartPositions->getData(false)[0]);
      EXPECT_EQ((int)row_id,
                arguments[i].sequenceStartPositions->getData(false)[1]);
      EXPECT_EQ(0, arguments[i].subSequenceStartPositions->getData(false)[0]);
      EXPECT_EQ((int)row_id,
                arguments[i].subSequenceStartPositions->getData(false)[1]);
    }
    fin.close();
  }
}

int main(int argc, char** argv) {
  paddle::initMain(argc, argv);
  paddle::initPython(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

void checkValue(std::vector<paddle::Argument>& arguments,
                picojson::array& arr) {
  // CHECK SLOT 0, Sparse Value.
  paddle::Argument& sparse_values_seq = arguments[0];
  paddle::MatrixPtr& sparse_values_seq_rawmatrix = sparse_values_seq.value;
  EXPECT_TRUE(sparse_values_seq_rawmatrix != nullptr);
  paddle::CpuSparseMatrix* sparse_val_seq_sparse_mat =
      dynamic_cast<paddle::CpuSparseMatrix*>(sparse_values_seq_rawmatrix.get());
  EXPECT_TRUE(sparse_val_seq_sparse_mat != nullptr);
  EXPECT_EQ(arr.size(), arguments.size());
  EXPECT_TRUE(arr[0].is<picojson::array>());
  size_t row_id = 0;
  for (picojson::value& sparse_val_seq : arr[0].get<picojson::array>()) {
    std::unordered_map<int, real> cols;
    for (picojson::value& kv : sparse_val_seq.get<picojson::array>()) {
      EXPECT_TRUE(kv.get(0).is<double>());
      EXPECT_TRUE(kv.get(1).is<double>());
      int col = (int)(kv.get(0).get<double>());
      real val = (real)(kv.get(1).get<double>());
      cols.insert({col, val});
    }
    size_t colNum = sparse_val_seq_sparse_mat->getColNum(row_id);
    EXPECT_EQ(cols.size(), colNum);
    int* rowIds = sparse_val_seq_sparse_mat->getRowCols(row_id);
    real* rowBuf = sparse_val_seq_sparse_mat->getRowValues(row_id);
    for (size_t i = 0; i < colNum; ++i) {
      int id = rowIds[i];
      auto it = cols.find(id);
      EXPECT_NE(cols.end(), it);
      real expect = it->second;
      EXPECT_NEAR(expect, *rowBuf, 1e-5);
      ++rowBuf;
    }
    ++row_id;
  }

  // CHECK SLOT 1, Dense Value.
  paddle::Argument& dense_arg = arguments[1];
  paddle::MatrixPtr& dense_mat = dense_arg.value;
  EXPECT_NE(nullptr, dense_mat);
  EXPECT_TRUE(arr[1].is<picojson::array>());
  row_id = 0;
  for (picojson::value& dense_seq : arr[1].get<picojson::array>()) {
    EXPECT_TRUE(dense_seq.is<picojson::array>());
    picojson::array& row = dense_seq.get<picojson::array>();
    EXPECT_EQ(row.size(), dense_mat->getWidth());
    real* rowBuf = dense_mat->getRowBuf(row_id++);

    for (picojson::value& val : row) {
      EXPECT_TRUE(val.is<double>());
      real expect = val.get<double>();
      EXPECT_NEAR(expect, *rowBuf++, 1e-5);
    }
  }

  // CHECK SLOT 2, Sparse Non Value.
  paddle::Argument& sparse_non_val_arg = arguments[2];
  paddle::MatrixPtr& sparse_non_val_rawm = sparse_non_val_arg.value;
  EXPECT_NE(nullptr, sparse_non_val_rawm);
  paddle::CpuSparseMatrix* sparse_non_val_m =
      dynamic_cast<paddle::CpuSparseMatrix*>(sparse_non_val_rawm.get());
  EXPECT_NE(nullptr, sparse_non_val_m);
  row_id = 0;
  for (picojson::value& row : arr[2].get<picojson::array>()) {
    EXPECT_TRUE(row.is<picojson::array>());
    std::unordered_set<int> ids;
    for (picojson::value& id : row.get<picojson::array>()) {
      EXPECT_TRUE(id.is<double>());
      ids.insert((int)(id.get<double>()));
    }
    size_t colNum = sparse_non_val_m->getColNum(row_id);
    EXPECT_EQ(ids.size(), colNum);
    for (size_t i = 0; i < colNum; ++i) {
      int col = sparse_non_val_m->getRowCols(row_id)[i];
      EXPECT_TRUE(ids.find(col) != ids.end());
    }
    ++row_id;
  }

  // CHECK SLOT 3, Index.
  paddle::Argument& index_arg = arguments[3];
  paddle::IVectorPtr indices = index_arg.ids;
  EXPECT_NE(nullptr, indices);
  int* idPtr = indices->getData();
  for (picojson::value& id : arr[3].get<picojson::array>()) {
    EXPECT_TRUE(id.is<double>());
    int _id = (int)(id.get<double>());
    EXPECT_EQ(_id, *idPtr++);
  }

  // CHECK SLOT 4, String.
  paddle::Argument& strArg = arguments[4];
  std::vector<std::string>* strPtr = strArg.strs.get();
  EXPECT_NE(nullptr, strPtr);
  size_t vecIndex = 0;
  for (picojson::value& str : arr[4].get<picojson::array>()) {
    EXPECT_TRUE(str.is<std::string>());
    std::string _str = str.get<std::string>();
    EXPECT_EQ(_str, (*strPtr)[vecIndex++]);
  }
}

#else
int main() { return 0; }

#endif
