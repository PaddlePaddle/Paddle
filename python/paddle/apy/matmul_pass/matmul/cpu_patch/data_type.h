#pragma once

#include <immintrin.h>
#include <type_traits>

namespace gops {

// vector type

template <typename T, int ElementsPerAccess>
struct AVXVector;

template <>
struct AVXVector<float, 8> {
   using type = __m256;
};

template <typename T, int ElementsPerAccess>
using VectorType = typename AVXVector<T, ElementsPerAccess>::type;


// compile time number
template <std::size_t Is>
using Number = std::integral_constant<std::size_t, Is>;

}