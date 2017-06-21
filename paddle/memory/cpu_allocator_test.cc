#include "paddle/memory/cpu_allocator.h"
#include <sstream>
#include "gtest/gtest.h"
#include "gflags/gflags.h"

DECLARE_bool(uses_pinned_allocator);

namespace cpu = ::paddle::memory::cpu;

TEST(CpuAllocator, Pinned) {
	size_t index;
	size_t size = 256;
	EXPECT_FALSE(cpu::SystemAllocator::uses_gpu());

	// use cpu pinned allocator
	FLAGS_uses_pinned_allocator = true;

	cpu::SystemAllocator::init();

	EXPECT_EQ(cpu::SystemAllocator::index_count(), 2U);

	void* pin_alloc_ptr1 = cpu::SystemAllocator::malloc(index, size);
	void* pin_alloc_ptr2 = cpu::SystemAllocator::malloc(index, size * 2);

	cpu::SystemAllocator::free(pin_alloc_ptr1, size, index);
	cpu::SystemAllocator::free(pin_alloc_ptr2, size * 2, index);

	cpu::SystemAllocator::shutdown();
}


TEST(CpuAllocator, Standard) {
	size_t index;
	size_t size = 256;
	EXPECT_FALSE(cpu::SystemAllocator::uses_gpu());

	// use cpu default allocator
	FLAGS_uses_pinned_allocator = false;

	cpu::SystemAllocator::init();

	EXPECT_EQ(cpu::SystemAllocator::index_count(), 1U);

	void* def_alloc_ptr1 = cpu::SystemAllocator::malloc(index, size);
	void* def_alloc_ptr2 = cpu::SystemAllocator::malloc(index, size * 2);

	cpu::SystemAllocator::free(def_alloc_ptr1, size, index);
	cpu::SystemAllocator::free(def_alloc_ptr2, size * 2, index);

	cpu::SystemAllocator::shutdown();
}
