#include "paddle/platform/place.h"
#include <sstream>
#include "gtest/gtest.h"

TEST(Place, Equality) {
  paddle::platform::CPUPlace cpu;
  paddle::platform::GPUPlace g0(0), g1(1), gg0(0);

  EXPECT_EQ(g0, gg0);

  EXPECT_NE(g0, g1);

  EXPECT_TRUE(paddle::platform::places_are_same_class(g0, gg0));
  EXPECT_FALSE(paddle::platform::places_are_same_class(g0, cpu));

  paddle::platform::NCCLPlace nccl_place(0);
  paddle::platform::MKLPlace mkl_place;
  paddle::platform::CPUPlace cpu_place;
  EXPECT_EQ(mkl_place, cpu_place);
}

TEST(Place, Default) {
  // EXPECT_TRUE(paddle::platform::is_gpu_place(paddle::platform::get_place()));
  EXPECT_TRUE(paddle::platform::is_gpu_place(paddle::platform::default_gpu()));
  EXPECT_TRUE(paddle::platform::is_cpu_place(paddle::platform::default_cpu()));

  paddle::platform::set_place(paddle::platform::CPUPlace());
  EXPECT_TRUE(paddle::platform::is_cpu_place(paddle::platform::get_place()));
}

TEST(Place, Print) {
  {
    std::stringstream ss;
    ss << paddle::platform::GPUPlace(1);
    EXPECT_EQ("GPUPlace(1)", ss.str());
  }
  {
    std::stringstream ss;
    ss << paddle::platform::CPUPlace();
    EXPECT_EQ("CPUPlace", ss.str());
  }
}
