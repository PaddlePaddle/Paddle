#include "malloc.h"
#include "default_allocator.h"

#ifndef PADDLE_ONLY_CPU
#include <cuda.h>
#endif

namespace majel {
namespace malloc {

void* malloc(majel::Place place, size_t size) {
  return detail::DefaultAllocator::malloc(place, size);
}

void free(majel::Place place, void* ptr) {
  detail::DefaultAllocator::free(place, ptr);
}

}  // namespace malloc
}  // namespace majel