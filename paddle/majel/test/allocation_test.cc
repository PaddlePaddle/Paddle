#include "paddle/majel/allocation.h"
#include "gtest/gtest.h"

TEST(Allocation, malloc) {
  majel::CpuPlace cpu;
  majel::Allocation cpuBlock(100, cpu);
  EXPECT_EQ(static_cast<size_t>(100), cpuBlock.size());
  EXPECT_TRUE(majel::places_are_same_class(cpu, cpuBlock.place()));
  EXPECT_EQ(100, (uint8_t*)cpuBlock.end() - (uint8_t*)cpuBlock.ptr());

#ifndef PADDLE_ONLY_CPU
  majel::GpuPlace gpu;
  majel::Allocation gpuBlock(200, gpu);
  EXPECT_EQ(static_cast<size_t>(200), gpuBlock.size());
  EXPECT_TRUE(majel::places_are_same_class(gpu, gpuBlock.place()));
  EXPECT_EQ(200, (uint8_t*)gpuBlock.end() - (uint8_t*)gpuBlock.ptr());
#endif
}
