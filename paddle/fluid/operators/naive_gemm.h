#pragma once
#include <iostream>
#define __NAIVE_GEMM__
namespace naive {

template <typename T>
class Matrix {
 public:
  Matrix(const T* data, const int rows, const int cols, const int ldx, const bool trans)
      : _data((T*)data), _rows(rows), _cols(cols), _ldx(ldx), _trans(trans) {}
  ~Matrix() {}

  T& get(int i, int j, int t1=0, int t2=0, int t3=0, int t4=0) {
    if (_trans) {
      if (i < rows() && j < cols())
      {
        return _data[j * _ldx + i];
      }
      std::cerr << "wrong index: i = " << i << ", j = " << j
              << ", _rows = " << _rows << ", _cols = " << _cols 
              << "trans: " << _trans 
              << "t1: " << t1 << ", t2: " << t2 << ", t3: " << t3 << ", t4: " << t4
              << std::endl;
      exit(1);
      return _get(j, i, t1, t2, t3, t4);
    } else {
      if (i < rows() && j < cols())
      {
        return _data[i * _ldx + j];
      }
      std::cerr << "wrong index: i = " << i << ", j = " << j
              << ", _rows = " << _rows << ", _cols = " << _cols
              << "trans: " << _trans
              << "t1: " << t1 << ", t2: " << t2 << ", t3: " << t3 << ", t4: " << t4
              << std::endl;
      exit(1);
      return _get(i, j, t1, t2, t3, t4);
    }
  }

  inline int rows() { return _rows; }
  inline int cols() { return _cols; }

 private:
  T& _get(const int i, const int j, int t1=0, int t2=0, int t3=0, int t4=0) {
    if (i < rows() && j < cols()) {
      return _data[i * _ldx + j];
    }
    std::cerr << "wrong index: i = " << i << ", j = " << j
              << ", _rows = " << _rows << ", _cols = " << _cols 
              << "trans: " << _trans 
              << "t1: " << t1 << ", t2: " << t2 << ", t3: " << t3 << ", t4: " << t4
              << std::endl;
    exit(1);
  }
  T* _data;
  int _rows, _cols, _ldx;
  bool _trans;
};

template <typename T>
void gemm(const bool row_major,
          const bool trans_a,
          const bool trans_b,
          const int M,
          const int N,
          const int K,
          const T alpha,
          const T* A,
          const int lda,
          const T* B,
          const int ldb,
          const T beta,
          T* C,
          const int ldc) {
  Matrix<T> ma(A, M, K, lda, trans_a);
  Matrix<T> mb(B, K, N, ldb, trans_b);
  Matrix<T> mc(C, M, N, ldc, false);
  /*
  std::cerr << "M: " << M << ", N: " << N << ", K: " << K 
            << ", lda: " << lda << ", ldb: " << ldb << ", ldc: "<< ldc
            << ", trans_a:" << trans_a << ", trans_b: " << trans_b
            << std::endl;
  */
  if (row_major) {
    for (int row_c = 0; row_c < M; row_c++) {
      for (int col_c = 0; col_c < ldc; col_c++) {
        float a_mul_b = 0;
        if (ma.cols() != mb.rows())
        {
          std::cerr << "ma.cols() != mb.rows(): " << ma.cols() << " != " << mb.rows() << std::endl;
          exit(1);
        }
        for (int pos = 0; pos < ma.cols(); pos++)
        {
          a_mul_b += ma.get(row_c, pos, row_c, col_c, pos) * mb.get(pos, col_c, row_c, col_c, pos);
        }
        mc.get(row_c, col_c) = alpha * a_mul_b + beta * mc.get(row_c, col_c, row_c, col_c);
      }
    }

  } else {
    std::cerr << "error: support RowMajor only!" << std::endl;
    exit(1);
  }
}

template <typename T>
void gemm(const bool trans_a,
          const bool trans_b,
          int M,
          int N,
          int K,
          T alpha,
          const T* A,
          const T* B,
          T beta,
          T* C) {
  int lda = (trans_a == false) ? K : M;
  int ldb = (trans_b == false) ? N : K;
  int ldc = N;
  gemm(true, trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

}  // namespace naive
