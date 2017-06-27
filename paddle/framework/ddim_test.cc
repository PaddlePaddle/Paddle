#include <sstream>
#include <vector>

#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"
#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "gtest/gtest.h"
#include "paddle/framework/ddim.h"

TEST(DDim, Equality) {
  // construct a DDim from an initialization list
  paddle::framework::DDim ddim = paddle::framework::make_ddim({9, 1, 5});
  EXPECT_EQ(ddim[0], 9);
  EXPECT_EQ(ddim[1], 1);
  EXPECT_EQ(ddim[2], 5);

  // construct a DDim from a vector
  std::vector<int> vec({9, 1, 5});
  paddle::framework::DDim vddim = paddle::framework::make_ddim(vec);
  EXPECT_EQ(ddim[0], 9);
  EXPECT_EQ(ddim[1], 1);
  EXPECT_EQ(ddim[2], 5);

  // mutate a DDim
  ddim[1] = 2;
  EXPECT_EQ(ddim[1], 2);
  paddle::framework::set(ddim, 0, 6);
  EXPECT_EQ(paddle::framework::get(ddim, 0), 6);

  // vectorize a DDim
  std::vector<int> res_vec = paddle::framework::vectorize(vddim);
  EXPECT_EQ(res_vec[0], 9);
  EXPECT_EQ(res_vec[1], 1);
  EXPECT_EQ(res_vec[2], 5);
  paddle::framework::Dim<3> d(3, 2, 1);
  res_vec = paddle::framework::vectorize(paddle::framework::DDim(d));
  EXPECT_EQ(res_vec[0], 3);
  EXPECT_EQ(res_vec[1], 2);
  EXPECT_EQ(res_vec[2], 1);

  // add two DDims
  paddle::framework::DDim ddim_sum = ddim + vddim;
  EXPECT_EQ(ddim_sum[0], 15);
  EXPECT_EQ(ddim_sum[1], 3);
  EXPECT_EQ(ddim_sum[2], 10);

  // multiply two DDims
  paddle::framework::DDim ddim_mul = ddim * vddim;
  EXPECT_EQ(ddim_mul[0], 54);
  EXPECT_EQ(ddim_mul[1], 2);
  EXPECT_EQ(ddim_mul[2], 25);

  // arity of a DDim
  EXPECT_EQ(paddle::framework::arity(ddim), 3);

  // product of a DDim
  EXPECT_EQ(paddle::framework::product(vddim), 45);
}

TEST(DDim, Print) {
  // print a DDim
  std::stringstream ss;
  paddle::framework::DDim ddim = paddle::framework::make_ddim({2, 3, 4});
  ss << ddim;
  EXPECT_EQ("2, 3, 4", ss.str());
}

template <typename T>
using Vec =
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>,
                     Eigen::Aligned>;

template <typename T>
using Matrix =
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex>,
                     Eigen::Aligned>;

template <typename T>
void print(T* input, int size) {
  for (int i = 0; i < size; i++) {
    std::cout << input[i] << " ";
  }
  std::cout << std::endl;
}

TEST(Eigen, start) {
  int size = 4;

  float* t_a = (float*)malloc(size * sizeof(float));
  float* t_b = (float*)malloc(size * sizeof(float));
  float* t_c = (float*)malloc(size * sizeof(float));
  for (int i = 0; i < size; i++) {
    t_a[i] = i;
    t_b[i] = i;
  }
  Vec<float> a(t_a, size);
  Vec<float> b(t_b, size);
  Vec<float> c(t_c, size);

  Eigen::DefaultDevice dd;
  c.device(dd) = a + b;
  print<float>(t_c, size);

  free(t_a);
  free(t_b);
  free(t_c);
}
