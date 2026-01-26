
#pragma once

namespace gops {

struct BatchedMatrixCoord {
  int batch;
  int row;
  int column;
  bool is_valid;

  BatchedMatrixCoord() : batch(0), row(0), column(0), is_valid(false) {}

  BatchedMatrixCoord(int b, int r, int c) : batch(b), row(r), column(c), is_valid(true) {}

  BatchedMatrixCoord(int b, int r, int c, bool valid) : batch(b), row(r), column(c), is_valid(valid) {}
};

};  // namespace ck
