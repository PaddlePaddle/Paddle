#include "paddle/platform/place.h"
#include <sstream>
#include "gtest/gtest.h"

TEST(Place, Equality) {
  paddle::platform::CPUPlace cpu;
  paddle::platform::CUDAPlace g0(0), g1(1), gg0(0);
  paddle::platform::CUDNNPlace d0(0), d1(1), dd0(0);

  EXPECT_EQ(cpu, cpu);
  EXPECT_EQ(g0, g0);
  EXPECT_EQ(g1, g1);
  EXPECT_EQ(g0, gg0);
  EXPECT_EQ(d0, dd0);

  EXPECT_NE(g0, g1);
  EXPECT_NE(d0, d1);

  EXPECT_TRUE(paddle::platform::places_are_same_class(g0, gg0));
  EXPECT_FALSE(paddle::platform::places_are_same_class(g0, cpu));

  EXPECT_TRUE(paddle::platform::is_gpu_place(d0));
  EXPECT_FALSE(paddle::platform::places_are_same_class(g0, d0));
}

TEST(Place, Default) {
  EXPECT_TRUE(paddle::platform::is_gpu_place(paddle::platform::get_place()));
  EXPECT_TRUE(paddle::platform::is_gpu_place(paddle::platform::default_gpu()));
  EXPECT_TRUE(paddle::platform::is_cpu_place(paddle::platform::default_cpu()));
  EXPECT_TRUE(
      paddle::platform::is_mkldnn_place(paddle::platform::default_mkldnn()));

  paddle::platform::set_place(paddle::platform::CPUPlace());
  EXPECT_TRUE(paddle::platform::is_cpu_place(paddle::platform::get_place()));

  paddle::platform::set_place(paddle::platform::MKLDNNPlace());
  EXPECT_FALSE(paddle::platform::is_cpu_place(paddle::platform::get_place()));
  EXPECT_TRUE(paddle::platform::is_mkldnn_place(paddle::platform::get_place()));
}

TEST(Place, Print) {
  {
    std::stringstream ss;
    ss << paddle::platform::CUDAPlace(1);
    EXPECT_EQ("CUDAPlace(1)", ss.str());
  }
  {
    std::stringstream ss;
    ss << paddle::platform::CPUPlace();
    EXPECT_EQ("CPUPlace", ss.str());
  }
}
