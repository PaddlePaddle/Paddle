#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest

import paddle.profiler.statistic_helper as statistic_helper


class TestStatisticHelper(unittest.TestCase):
    def test_sum_ranges_case1(self):
        src = [(1, 3), (4, 10), (11, 15)]
        self.assertEqual(statistic_helper.sum_ranges(src), 12)

    def test_sum_ranges_case2(self):
        src = [(3, 3), (5, 5), (7, 7)]
        self.assertEqual(statistic_helper.sum_ranges(src), 0)

    def test_merge_self_ranges_case1(self):
        src = [(1, 5), (2, 7), (4, 9), (14, 19)]
        dst = statistic_helper.merge_self_ranges(src)
        self.assertEqual(dst, [(1, 9), (14, 19)])
        src = [(4, 9), (14, 19), (1, 5), (2, 7)]
        dst = statistic_helper.merge_self_ranges(src)
        self.assertEqual(dst, [(1, 9), (14, 19)])

    def test_merge_self_ranges_case2(self):
        src = [(1, 1), (2, 3), (4, 7), (5, 12)]
        dst = statistic_helper.merge_self_ranges(src)
        self.assertEqual(dst, [(1, 1), (2, 3), (4, 12)])
        src = [(5, 12), (1, 1), (2, 3), (4, 7)]
        dst = statistic_helper.merge_self_ranges(src)
        self.assertEqual(dst, [(1, 1), (2, 3), (4, 12)])

    def test_merge_ranges_case1(self):
        src1 = [(1, 2), (5, 7), (9, 14)]
        src2 = [(1, 2), (4, 9), (13, 15)]
        dst = statistic_helper.merge_ranges(src1, src2)
        self.assertEqual(dst, [(1, 2), (4, 15)])
        dst = statistic_helper.merge_ranges(src1, src2, True)
        self.assertEqual(dst, [(1, 2), (4, 15)])
        src1 = []
        src2 = []
        dst = statistic_helper.merge_ranges(src1, src2, True)
        self.assertEqual(dst, [])
        src1 = [(1, 2), (3, 5)]
        src2 = []
        dst = statistic_helper.merge_ranges(src1, src2, True)
        self.assertEqual(dst, src1)
        src1 = []
        src2 = [(1, 2), (3, 5)]
        dst = statistic_helper.merge_ranges(src1, src2, True)
        self.assertEqual(dst, src2)
        src1 = [(3, 4), (1, 2), (17, 19)]
        src2 = [(6, 9), (13, 15)]
        dst = statistic_helper.merge_ranges(src1, src2)
        self.assertEqual(dst, [(1, 2), (3, 4), (6, 9), (13, 15), (17, 19)])
        dst = statistic_helper.merge_ranges(src2, src1)
        self.assertEqual(dst, [(1, 2), (3, 4), (6, 9), (13, 15), (17, 19)])
        src1 = [(1, 2), (5, 9), (12, 13)]
        src2 = [(6, 8), (9, 15)]
        dst = statistic_helper.merge_ranges(src1, src2)
        self.assertEqual(dst, [(1, 2), (5, 15)])
        dst = statistic_helper.merge_ranges(src2, src1)
        self.assertEqual(dst, [(1, 2), (5, 15)])

    def test_merge_ranges_case2(self):
        src1 = [(3, 4), (1, 2), (9, 14)]
        src2 = [(6, 9), (13, 15)]
        dst = statistic_helper.merge_ranges(src1, src2)
        self.assertEqual(dst, [(1, 2), (3, 4), (6, 15)])
        src2 = [(9, 14), (1, 2), (5, 7)]
        src1 = [(4, 9), (1, 2), (13, 15)]
        dst = statistic_helper.merge_ranges(src1, src2)
        self.assertEqual(dst, [(1, 2), (4, 15)])

    def test_intersection_ranges_case1(self):
        src1 = [(1, 7), (9, 12), (14, 18)]
        src2 = [(3, 8), (10, 13), (15, 19)]
        dst = statistic_helper.intersection_ranges(src1, src2)
        self.assertEqual(dst, [(3, 7), (10, 12), (15, 18)])
        dst = statistic_helper.intersection_ranges(src1, src2, True)
        self.assertEqual(dst, [(3, 7), (10, 12), (15, 18)])
        src1 = []
        src2 = []
        dst = statistic_helper.intersection_ranges(src1, src2, True)
        self.assertEqual(dst, [])
        src1 = [(3, 7), (10, 12)]
        src2 = [(2, 9), (11, 13), (15, 19)]
        dst = statistic_helper.intersection_ranges(src1, src2)
        self.assertEqual(dst, [(3, 7), (11, 12)])
        dst = statistic_helper.intersection_ranges(src2, src1)
        self.assertEqual(dst, [(3, 7), (11, 12)])

    def test_intersection_ranges_case2(self):
        src2 = [(9, 12), (1, 7), (14, 18)]
        src1 = [(10, 13), (3, 8), (15, 19), (20, 22)]
        dst = statistic_helper.intersection_ranges(src1, src2)
        self.assertEqual(dst, [(3, 7), (10, 12), (15, 18)])
        src2 = [(1, 7), (14, 18), (21, 23)]
        src1 = [(6, 9), (10, 13)]
        dst = statistic_helper.intersection_ranges(src1, src2, True)
        self.assertEqual(dst, [(6, 7)])

    def test_subtract_ranges_case1(self):
        src1 = [(1, 10), (12, 15)]
        src2 = [(3, 7), (9, 11)]
        dst = statistic_helper.subtract_ranges(src1, src2, True)
        self.assertEqual(dst, [(1, 3), (7, 9), (12, 15)])
        src1 = [(1, 10), (12, 15)]
        src2 = []
        dst = statistic_helper.subtract_ranges(src1, src2, True)
        self.assertEqual(dst, src1)
        dst = statistic_helper.subtract_ranges(src2, src1, True)
        self.assertEqual(dst, src2)

    def test_subtract_ranges_case2(self):
        src2 = [(12, 15), (1, 10)]
        src1 = [(9, 11), (3, 7)]
        dst = statistic_helper.subtract_ranges(src1, src2)
        self.assertEqual(dst, [(10, 11)])


if __name__ == '__main__':
    unittest.main()
