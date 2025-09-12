# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


import random
import string
import unittest

from test_case_base import TestCaseBase

from paddle.jit.sot.utils.exceptions import (
    BuiltinFunctionBreak,
    DataDependencyControlFlowBreak,
    DataDependencyDynamicShapeBreak,
    DataDependencyOperationBreak,
    DygraphInconsistentWithStaticBreak,
    FallbackInlineCallBreak,
    InferMetaBreak,
    InlineCallBreak,
    OtherInlineCallBreak,
    PsdbBreakReason,
    SideEffectBreak,
    UnsupportedIteratorBreak,
    UnsupportedPaddleAPIBreak,
)
from paddle.jit.sot.utils.info_collector import (
    BreakGraphReasonInfo,
    InfoBase,
    SubGraphInfo,
)

generate_random_string = lambda N: ''.join(
    random.choices(string.ascii_uppercase + string.digits, k=N)
)


class TestSerialize(TestCaseBase):
    def test_case(self):
        x_dict = {
            'a': 'b',
            'c': 'd',
        }

        x_str = InfoBase.serialize(x_dict)
        y_dict = InfoBase.deserialize(x_str)

        self.assertEqual(x_dict, y_dict)


class TestBreakGraphReasonInfo(TestCaseBase):
    def test_case(self):
        history = [
            BreakGraphReasonInfo(
                BreakReasonClass(
                    reason_str=generate_random_string(random.randint(1, 5))
                )
            )
            for BreakReasonClass in [
                FallbackInlineCallBreak,
                DataDependencyControlFlowBreak,
                DataDependencyDynamicShapeBreak,
                DataDependencyOperationBreak,
                UnsupportedPaddleAPIBreak,
                BuiltinFunctionBreak,
                SideEffectBreak,
                UnsupportedIteratorBreak,
                InlineCallBreak,
                OtherInlineCallBreak,
                DygraphInconsistentWithStaticBreak,
                PsdbBreakReason,
                InferMetaBreak,
            ]
        ]

        serialized = BreakGraphReasonInfo.json_report(history)
        deserialized = BreakGraphReasonInfo.restore_from_string(
            serialized[5:-6]  # remove `<sot>` and `</sot>`
        )  # `removeprefix` & `removesuffix` are only available from python3.9

        origin_reasons_dict, _ = BreakGraphReasonInfo.classify(history)
        origin_reasons2count = {
            k: len(v) for k, v in origin_reasons_dict.items()
        }

        new_reasons_dict, _ = BreakGraphReasonInfo.classify(deserialized)
        new_reasons2count = {k: len(v) for k, v in new_reasons_dict.items()}

        self.assertEqual(origin_reasons2count, new_reasons2count)


class TestSubGraphInfo(TestCaseBase):
    def test_case(self):
        history = [
            SubGraphInfo(
                generate_random_string(random.randint(1, 5)),
                random.randint(0, 20),
                generate_random_string(random.randint(1, 5)),
            ),
        ] * 10

        serialized = SubGraphInfo.json_report(history)
        deserialized = SubGraphInfo.restore_from_string(
            serialized[5:-6]  # remove `<sot>` and `</sot>`
        )  # `removeprefix` & `removesuffix` are only available from python3.9

        self.assertEqual(history, deserialized)


if __name__ == "__main__":
    unittest.main()
