#pragma once
#include "paddle/majel/place.h"

namespace majel {
namespace malloc {

void* malloc(majel::Place place, size_t size);
void free(majel::Place place, void* ptr);

}  // namespace malloc
}  // namespace majel
