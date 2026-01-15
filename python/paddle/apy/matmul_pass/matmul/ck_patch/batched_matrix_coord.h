
#pragma once

#include "ck/ck.hpp"

namespace ck {

struct BatchedMatrixCoord {
  int batch;
  int row;
  int column;
  bool is_valid;

  __host__ __device__
  BatchedMatrixCoord() : batch(0), row(0), column(0), is_valid(false) {}

  __host__ __device__
  BatchedMatrixCoord(int b, int r, int c) : batch(b), row(r), column(c), is_valid(true) {}

  __host__ __device__
  BatchedMatrixCoord(int b, int r, int c, bool valid) : batch(b), row(r), column(c), is_valid(valid) {}
};

};  // namespace ck
