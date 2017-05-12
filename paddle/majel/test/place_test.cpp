#include "gtest/gtest.h"
#include "majel/place.h"
#include <sstream>

TEST(Place, Equality) {
  majel::CpuPlace cpu;
  majel::GpuPlace g0(0), g1(1), gg0(0);

  EXPECT_EQ(cpu, cpu);
  EXPECT_EQ(g0, g0);
  EXPECT_EQ(g1, g1);
  EXPECT_EQ(g0, gg0);

  EXPECT_NE(g0, g1);

  EXPECT_TRUE(majel::places_are_same_class(g0, gg0));
  EXPECT_FALSE(majel::places_are_same_class(g0, cpu));
}

TEST(Place, Default) {
  EXPECT_TRUE(majel::is_gpu_place( majel::get_place()));
  EXPECT_TRUE(majel::is_gpu_place( majel::default_gpu()));
  EXPECT_TRUE(majel::is_cpu_place( majel::default_cpu()));

  majel::set_place(majel::CpuPlace());
  EXPECT_TRUE(majel::is_cpu_place( majel::get_place()));
}

TEST(Place, Print) {
  {
    std::stringstream ss;
    ss << majel::GpuPlace(1);
    EXPECT_EQ("GpuPlace(1)", ss.str());
  }
  {
     std::stringstream ss;
     ss << majel::CpuPlace();
     EXPECT_EQ("CpuPlace", ss.str());
  }
}
