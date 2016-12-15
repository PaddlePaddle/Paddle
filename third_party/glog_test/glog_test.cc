#include <iostream>
#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"

TEST(GlogTest, Logging) { LOG(INFO) << "Hello world"; }
