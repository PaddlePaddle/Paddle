#include <thrust/device_vector.h>
#include <sstream>

#include "paddle/majel/ddim/dim.h"
#include "gtest/gtest.h"

__global__ void test(majel::Dim<2>* o) {
    o[0] = majel::make_dim(5, 6);
}

__global__ void dyn_idx_gpu(int* o) {
    auto d = majel::make_dim(5, 6);
    o[0] = d[1];
}

TEST(Dim, Equality) {
    // construct a Dim on the CPU
    auto a = majel::make_dim(3, 4);
    EXPECT_EQ(majel::get<0>(a), 3);
    EXPECT_EQ(majel::get<1>(a), 4);

    // construct a Dim on the GPU
    thrust::device_vector<majel::Dim<2>> t(2);
    test<<<1,1>>>(thrust::raw_pointer_cast(t.data()));
    a = t[0];
    EXPECT_EQ(majel::get<0>(a), 5);
    EXPECT_EQ(majel::get<1>(a), 6);

    // linearization
    auto b = majel::make_dim(7, 8);
    EXPECT_EQ(majel::linearize(a, b), 83);

    // product
    EXPECT_EQ(majel::product(a), 30);

    // mutate a Dim
    majel::get<1>(b) = 10;
    EXPECT_EQ(majel::get<0>(b), 7);
    EXPECT_EQ(majel::get<1>(b), 10);

    // dynamic access
    majel::get(b, 0) = 8;
    b[1] = 11;
    EXPECT_EQ(majel::get<0>(b), 8);
    EXPECT_EQ(majel::get<1>(b), 11);
    EXPECT_EQ(majel::get(b, 0), 8);
    EXPECT_EQ(b[1], 11);

    // dynamic access on GPU
    thrust::device_vector<int> r(1);
    dyn_idx_gpu<<<1,1>>>(thrust::raw_pointer_cast(r.data()));
    int res = r[0];
    EXPECT_EQ(res, 6);

    // ex_prefix_mul
    majel::Dim<3> c = majel::ex_prefix_mul(majel::Dim<3>(3, 4, 5));
    EXPECT_EQ(majel::get<0>(c), 1);
    EXPECT_EQ(majel::get<1>(c), 3);
    EXPECT_EQ(majel::get<2>(c), 12);

    // contiguous_strides
    c = majel::contiguous_strides(majel::Dim<3>(10, 1, 10));
    EXPECT_EQ(majel::get<0>(c), 1);
    EXPECT_EQ(majel::get<1>(c), 0);
    EXPECT_EQ(majel::get<2>(c), 10);
    c = majel::contiguous_strides(majel::Dim<3>(10, 10, 1));
    EXPECT_EQ(majel::get<0>(c), 1);
    EXPECT_EQ(majel::get<1>(c), 10);
    EXPECT_EQ(majel::get<2>(c), 0);
    c = majel::contiguous_strides(majel::Dim<3>(1, 10, 10));
    EXPECT_EQ(majel::get<0>(c), 0);
    EXPECT_EQ(majel::get<1>(c), 1);
    EXPECT_EQ(majel::get<2>(c), 10);
    c = majel::contiguous_strides(majel::Dim<3>(2, 3, 4));
    EXPECT_EQ(majel::get<0>(c), 1);
    EXPECT_EQ(majel::get<1>(c), 2);
    EXPECT_EQ(majel::get<2>(c), 6);

    // generate from an index
    auto size = majel::make_dim(4, 5, 2);
    c = majel::Dim<3>(14, size);
    EXPECT_EQ(majel::get<0>(c), 2);
    EXPECT_EQ(majel::get<1>(c), 3);
    EXPECT_EQ(majel::get<2>(c), 0);
    c = majel::Dim<3>(25, size);
    EXPECT_EQ(majel::get<0>(c), 1);
    EXPECT_EQ(majel::get<1>(c), 1);
    EXPECT_EQ(majel::get<2>(c), 1);
}

TEST(Dim, Bool) {
    auto a = majel::make_dim(3, 4);
    auto b = majel::make_dim(5, 6);
    auto c = majel::make_dim(3, 4);

    // in_bounds check
    EXPECT_TRUE(majel::contained(a, b));
    EXPECT_FALSE(majel::contained(b, a));

    // comparison
    EXPECT_TRUE(a == a);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a == c);

    // contiguous check
    int x = 4, y = 5, z = 2;
    majel::Dim<3> sizef(x, y, z);
    majel::Dim<3> stridea(1, x, x*y);
    majel::Dim<3> strideb(2, 2*x, 2*x*y);
    majel::Dim<3> stridec(1, x, 2*x*y);
    EXPECT_TRUE(majel::contiguous(sizef, stridea));
    EXPECT_FALSE(majel::contiguous(sizef, strideb));
    EXPECT_FALSE(majel::contiguous(sizef, stridec));
}

TEST(Dim, Print) {
    {
        std::stringstream ss;
        auto a = majel::make_dim(2, 3);
        ss << a;
        EXPECT_EQ(ss.str(), "2, 3");
    }
    {
        std::stringstream ss;
        ss << majel::make_dim(8);
        EXPECT_EQ(ss.str(), "8");
    }
}
