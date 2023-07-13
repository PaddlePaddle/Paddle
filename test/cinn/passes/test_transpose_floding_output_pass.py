#!/usr/bin/env python3

# Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

import unittest

from pass_test import PassTest


class TestTransposeFoldingOutputPass(PassTest):
    def init_input_data(self):
        """Do not set the shape like [B, N, N].
        You should set the shape like [B, M, N], where M != N.
        """
        self.feed_data = {
            'x': self.random([4, 3, 5], "float32"),
            'y': self.random([4, 5, 6], "float32"),
        }

    def expect_folding_number(self):
        return 1

    def trans_out_func(self, builder, out):
        return builder.transpose(out, [0, 2, 1])

    def build_program(self, builder, target):
        x = builder.create_input(
            str(self.feed_data['x'].dtype), self.feed_data['x'].shape, "x"
        )
        y = builder.create_input(
            str(self.feed_data['y'].dtype), self.feed_data['y'].shape, "y"
        )
        res = builder.matmul(x, y)
        out = self.trans_out_func(builder, res)
        return [x, y], [out]

    def test_check_results(self):
        self.check_pass_outputs(
            pass_diff=self.expect_folding_number(),
            test_passes=[
                "TransposeFoldingInput",
                "GemmRewriter",
                "TransposeFoldingOutput",
                "GemmRewriter",
            ],
        )


class TestTransposeFoldingOutputPassWithScale(TestTransposeFoldingOutputPass):
    def expect_folding_number(self):
        return 2

    def trans_out_func(self, builder, out):
        out_s = builder.scale(out, scale=2.0)
        return builder.transpose(out_s, [0, 2, 1])


class TestTransposeFoldingOutputPassWithIdentity(
    TestTransposeFoldingOutputPass
):
    def expect_folding_number(self):
        return 2

    def trans_out_func(self, builder, out):
        out_i = builder.identity(out)
        out_s = builder.scale(out_i, scale=2.0)
        return builder.transpose(out_s, [0, 2, 1])


class TestTransposeFoldingOutputPassInvlidTrans(TestTransposeFoldingOutputPass):
    def expect_folding_number(self):
        return 1

    def trans_out_func(self, builder, out):
        out_t = builder.transpose(out, [1, 0, 2])
        return builder.scale(out_t, scale=2.0)


class TestTransposeFoldingOutputPassInvlidScale(TestTransposeFoldingOutputPass):
    def expect_folding_number(self):
        return 1

    def trans_out_func(self, builder, out):
        out_s = builder.scale(out, scale=2.0, bias=1.0)
        return builder.transpose(out_s, [0, 2, 1])


class TestTransposeFoldingOutputPassNoFold(TestTransposeFoldingOutputPass):
    def expect_folding_number(self):
        return 0

    def trans_out_func(self, builder, out):
        out_r = builder.reshape(out, [4, 6, 3])
        out_s = builder.scale(out_r, scale=2.0)
        return builder.transpose(out_s, [0, 2, 1])


if __name__ == "__main__":
    unittest.main()
