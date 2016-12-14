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

/*
 * Basically from tensorflow/core/platform/default/logging.cc
 * Used in embedded system where there is no glogs.
 */

#include <dirent.h>
#include <gtest/gtest.h>
#include <stdlib.h>
#include <fstream>
#include "paddle/utils/Logging.h"
#include "paddle/utils/Util.h"
#ifndef PADDLE_USE_GLOG
TEST(Logging, BasicalLog) {
  auto pinfo = [] {
    P_LOG(INFO) << "INFO";
    exit(1);
  };
  ASSERT_DEATH(pinfo(), "I .*test_Logging.cpp:[0-9]+] INFO");

  auto pwarn = [] {
    P_LOG(WARNING) << "WARN";
    exit(1);
  };
  ASSERT_DEATH(pwarn(), "W .*test_Logging.cpp:[0-9]+] WARN");

  auto perr = [] {
    P_LOG(ERROR) << "ERROR";
    exit(1);
  };
  ASSERT_DEATH(perr(), "E .*test_Logging.cpp:[0-9]+] ERROR");

  auto pfatal = [] { P_LOG(FATAL) << "FATAL"; };
  ASSERT_DEATH(pfatal(), "F .*test_Logging.cpp:[0-9]+] FATAL");
}

TEST(Logging, Check) {
  int a = 1;
  int b = 2;
  P_CHECK(a != b);

  auto pcheckDown = [&] { P_CHECK(a == b); };
  ASSERT_DEATH(pcheckDown(),
               "F .*test_Logging.cpp:[0-9]+] Check failed: a == b ");

  P_CHECK_LE(a, b);
  P_CHECK_LT(a, b);
  double t = 1.2;
  P_CHECK_LE(a, t);
  double* ptr = nullptr;

  auto pcheckDown2 = [&] { P_CHECK_NOTNULL(ptr); };
  ASSERT_DEATH(pcheckDown2(), "F");
}

#define cc(x) const_cast<char*>(x)

TEST(Logging, LogToStderr) {
  auto logToStderrCallback = [] {
    setenv("PLOG_LOGTOSTDERR", "0", true);
    char* argv[] = {cc("test")};
    paddle::initializeLogging(1, argv);
    P_LOG(INFO) << "This output will not print to std error";
    exit(1);
  };

  ASSERT_DEATH(logToStderrCallback(), "");
}

constexpr char kLogDirName[] = "./test_log_dir";
const std::vector<std::string> kLevels = {"INFO", "WARNING", "ERROR", "FATAL"};

TEST(Logging, LogToDir) {
  ASSERT_EQ(0, mkdir(kLogDirName, 0777));
  auto logToDirCallback = [] {
    setenv("PLOG_LOGTOSTDERR", "0", true);
    setenv("PLOG_LOGDIR", kLogDirName, true);
    char* argv[] = {cc("test")};
    paddle::initializeLogging(1, argv);

    P_LOG(INFO) << "INFO";
    P_LOG(WARNING) << "WARNING";
    P_LOG(ERROR) << "ERROR";
    P_LOG(FATAL) << "FATAL";
  };
  ASSERT_DEATH(logToDirCallback(), "");

  // There 4 file in logdir
  auto dir = opendir(kLogDirName);
  size_t fileCount = 0;
  std::vector<std::string> filenames;
  for (auto dirContent = readdir(dir); dirContent != nullptr;
       dirContent = readdir(dir)) {
    std::string filename(dirContent->d_name);
    if (filename == "." || filename == "..") {
      continue;
    } else {
      ++fileCount;
      for (size_t i = 0; i < kLevels.size(); ++i) {
        const std::string& curLevel = kLevels[i];
        if (filename.size() > curLevel.length()) {
          size_t diff = filename.size() - curLevel.length();
          size_t j = 0;
          for (; j < curLevel.length(); ++j) {
            if (filename[j + diff] != curLevel[j]) {
              // File Suffix Not Same, then break.
              break;
            }
          }
          if (j == curLevel.length()) {  // Same suffix.
            std::ifstream fin;
            auto fn = paddle::path::join(kLogDirName, filename);
            fin.open(fn);
            filenames.push_back(fn);
            ASSERT_TRUE(fin.is_open());
            size_t lineCounter = 0;
            for (std::string line; std::getline(fin, line); ++lineCounter) {
              // Do Nothing, Just calc lineCounter.
            }

            // For example.
            // The info channel will have all log which level >= INFO
            // So the info file's lineCounter should == 4.
            ASSERT_EQ(kLevels.size() - i, lineCounter);
            fin.close();
          }
        }
      }
    }
  }
  closedir(dir);
  ASSERT_EQ(4UL, fileCount);  // 4 levels.
  // Clean Unittest.
  for (std::string& fn : filenames) {
    ASSERT_EQ(remove(fn.c_str()), 0);
  }
  ASSERT_EQ(rmdir(kLogDirName), 0);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#else

int main(int, char**) { return 0; }

#endif
