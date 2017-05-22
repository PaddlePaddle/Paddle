#include "place.h"

namespace majel {
namespace malloc {
namespace detail {

class DefaultAllocator {
public:
  static void* malloc(majel::Place place, size_t size);
  static void free(majel::Place, void* ptr);
};
}  // namespace detail
}  // namespace malloc
}  // namespace majel
