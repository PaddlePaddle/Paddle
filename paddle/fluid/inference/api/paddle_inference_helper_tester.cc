#include <gtest/gtest.h>
#include <chrono>
#include <thread>

#include "paddle/fluid/inference/api/paddle_inference_helper.h"

namespace paddle {
namespace helper {

TEST(Timer, Basic) {
  Timer timer;
  timer.tic();
  std::this_thread::sleep_for(std::chrono::seconds(1));
  ASSERT_GE(timer.toc(), 1000);  // ms
}

TEST(to_string, Basic) {
  std::vector<int> x({0, 1, 2, 3});
  auto str = to_string(x, ',');
  ASSERT_EQ(str, "0,1,2,3");
}

TEST(TensorSniffer, Basic) {
  PaddleTensor x;
  x.dtype = PaddleDType::FLOAT32;
  x.lod.emplace_back(std::vector<size_t>({0, 1, 2}));
  x.name = "x";
  x.shape.assign({10, 3});

  TensorSniffer sniffer(x);

  EXPECT_EQ(sniffer.dtype(), "float32");
  EXPECT_EQ(sniffer.shape(), "[10 3]");
  EXPECT_EQ(sniffer.lod(), "[[0 1 2],]");
}

}  // namespace helper
}  // namespace paddle
