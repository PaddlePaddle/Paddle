# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np

from paddle.distributed.auto_parallel.tuner import recorder as rd


class TestRecorder(unittest.TestCase):
    def test_register(self):
        recorder = rd.MetricsRecorder()
        recorder.register("metric")
        self.assertEqual(set(recorder.records.keys()), {"metric"})
        self.assertEqual(recorder.records["metric"].direction, "min")

    def test_exists(self):
        recorder = rd.MetricsRecorder()
        recorder.register("metric", direction="max")
        self.assertTrue(recorder.exists("metric"))

    def test_update(self):
        recorder = rd.MetricsRecorder()
        recorder.update("metric", 4, 1000)
        self.assertEqual(recorder.records["metric"].direction, "min")
        self.assertEqual(
            recorder.get_records("metric"), [rd.MetricRecord(4, 1000)])

    def test_get_records(self):
        recorder = rd.MetricsRecorder()
        recorder.update("metric", 1, step=0)
        recorder.update("metric", 2, step=1)
        recorder.update("metric", 3, step=2)
        recorder.update("metric", 4, step=3)
        self.assertEqual(
            recorder.get_records("metric"), [
                rd.MetricRecord(1, 0),
                rd.MetricRecord(2, 1),
                rd.MetricRecord(3, 2),
                rd.MetricRecord(4, 3),
            ])

    def test_set_records(self):
        recorder = rd.MetricsRecorder()
        recorder.set_records(
            "metric",
            [
                rd.MetricRecord(1, 0),
                rd.MetricRecord(2, 1),
                rd.MetricRecord(3, 2),
                rd.MetricRecord(4, 3),
            ], )
        self.assertEqual(
            recorder.get_records("metric"), [
                rd.MetricRecord(1, 0),
                rd.MetricRecord(2, 1),
                rd.MetricRecord(3, 2),
                rd.MetricRecord(4, 3),
            ])

    def test_get_best_value(self):
        recorder = rd.MetricsRecorder()
        recorder.register("metric_min", "min")
        recorder.register("metric_max", "max")

        recorder.set_records(
            "metric_min",
            [
                rd.MetricRecord(1, 0),
                rd.MetricRecord(2, 1),
                rd.MetricRecord(3, 2),
                rd.MetricRecord(4, 3),
            ], )
        self.assertEqual(recorder.get_best_value("metric_min"), 1)

        recorder.set_records(
            "metric_max",
            [
                rd.MetricRecord(1, 0),
                rd.MetricRecord(2, 1),
                rd.MetricRecord(3, 2),
                rd.MetricRecord(4, 3),
            ], )
        self.assertEqual(recorder.get_best_value("metric_max"), 4)

    def test_get_best_step(self):
        recorder = rd.MetricsRecorder()

        recorder.register("metric_min", "min")
        recorder.set_records(
            "metric_min",
            [
                rd.MetricRecord(1, 0),
                rd.MetricRecord(2, 1),
                rd.MetricRecord(3, 2),
                rd.MetricRecord(4, 3),
            ], )
        self.assertEqual(recorder.get_best_step("metric_min"), 0)

        recorder.register("metric_max", "max")
        recorder.set_records(
            "metric_max",
            [
                rd.MetricRecord(1, 0),
                rd.MetricRecord(2, 1),
                rd.MetricRecord(3, 2),
                rd.MetricRecord(4, 3),
            ], )
        self.assertEqual(recorder.get_best_step("metric_max"), 3)

    def test_get_statistics(self):
        recorder = rd.MetricsRecorder()
        records = [rd.MetricRecord(np.random.random(), i) for i in range(14)]
        recorder.set_records("metric", records)
        stats = recorder.get_statistics("metric")
        records = [r.value for r in records]
        self.assertEqual(stats["min"], np.min(records))
        self.assertEqual(stats["max"], np.max(records))
        self.assertEqual(stats["mean"], np.mean(records))
        self.assertEqual(stats["median"], np.median(records))
        self.assertEqual(stats["var"], np.var(records))
        self.assertEqual(stats["std"], np.std(records))

    def test_serialization(self):
        recorder = rd.MetricsRecorder()
        recorder.register("metric")
        recorder.set_records(
            "metric",
            [
                rd.MetricRecord(1, 0),
                rd.MetricRecord(2, 1),
                rd.MetricRecord(3, 2),
                rd.MetricRecord(4, 3),
            ], )
        print(recorder.get_state())
        new_recorder = rd.MetricsRecorder.from_state(recorder.get_state())
        self.assertEqual(new_recorder.records.keys(), recorder.records.keys())


if __name__ == "__main__":
    unittest.main()
