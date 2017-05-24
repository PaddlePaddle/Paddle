//#include <stdexcept>
//#include <unittest/unittest.h>
#include <sstream>
#include <vector>

#include "gtest/gtest.h"
#include "paddle/majel/ddim/ddim.h"

TEST(DDim, Equality) {
  // construct a DDim from an initialization list
  majel::DDim ddim = majel::make_ddim({9, 1, 5});
  EXPECT_EQ(ddim[0], 9);
  EXPECT_EQ(ddim[1], 1);
  EXPECT_EQ(ddim[2], 5);

  // construct a DDim from a vector
  std::vector<int> vec({9, 1, 5});
  majel::DDim vddim = majel::make_ddim(vec);
  EXPECT_EQ(ddim[0], 9);
  EXPECT_EQ(ddim[1], 1);
  EXPECT_EQ(ddim[2], 5);

  // mutate a DDim
  ddim[1] = 2;
  EXPECT_EQ(ddim[1], 2);
  majel::set(ddim, 0, 6);
  EXPECT_EQ(majel::get(ddim, 0), 6);

  // vectorize a DDim
  std::vector<int> res_vec = majel::vectorize(vddim);
  EXPECT_EQ(res_vec[0], 9);
  EXPECT_EQ(res_vec[1], 1);
  EXPECT_EQ(res_vec[2], 5);
  majel::Dim<3> d(3, 2, 1);
  res_vec = majel::vectorize(majel::DDim(d));
  EXPECT_EQ(res_vec[0], 3);
  EXPECT_EQ(res_vec[1], 2);
  EXPECT_EQ(res_vec[2], 1);

  // add two DDims
  majel::DDim ddim_sum = ddim + vddim;
  EXPECT_EQ(ddim_sum[0], 15);
  EXPECT_EQ(ddim_sum[1], 3);
  EXPECT_EQ(ddim_sum[2], 10);

  // multiply two DDims
  majel::DDim ddim_mul = ddim * vddim;
  EXPECT_EQ(ddim_mul[0], 54);
  EXPECT_EQ(ddim_mul[1], 2);
  EXPECT_EQ(ddim_mul[2], 25);

  // arity of a DDim
  EXPECT_EQ(majel::arity(ddim), 3);

  // product of a DDim
  EXPECT_EQ(majel::product(vddim), 45);
}

TEST(DDim, Print) {
  // print a DDim
  std::stringstream ss;
  majel::DDim ddim = majel::make_ddim({2, 3, 4});
  ss << ddim;
  EXPECT_EQ("2, 3, 4", ss.str());
}
