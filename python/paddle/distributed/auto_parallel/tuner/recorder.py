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

# Notice that the following codes are modified from KerasTuner for a different purpose.
# Please refer to https://github.com/keras-team/keras-tuner/blob/master/keras_tuner/engine/metrics_tracking.py.

import numpy as np


class MetricRecord:
    """
    One record for a single metric at a given execution step.
    """

    def __init__(self, value, step):
        self._value = value
        self._step = step

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, step):
        self._step = step

    def mean(self):
        return np.mean(self.value)

    def get_state(self):
        return {"value": self.value, "step": self.step}

    @classmethod
    def from_state(cls, state):
        return cls(**state)

    def __eq__(self, other):
        if not isinstance(other, MetricRecord):
            return False
        return other.value == self.value and other.step == self.step

    def __repr__(self):
        return f"MetricRecord(value={self.value}, step={self.step})"


class MetricRecords:
    """
    Records of a single metric across different executions.
    """

    def __init__(self, direction="min"):
        if direction not in {"min", "max"}:
            raise ValueError(
                "direction should be one of {{min, max}}, but got: {}.".format(
                    direction
                )
            )
        self._direction = direction
        self._records = {}

    @property
    def records(self):
        return sorted(self._records.values(), key=lambda r: r.step)

    @records.setter
    def records(self, records):
        for r in records:
            self.update(r.value, step=r.step)

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, direction):
        self._direction = direction

    def update(self, value, step=0):
        if step in self._records:
            self._records[step].set_value(value)
        else:
            self._records[step] = MetricRecord(value, step=step)

    def get_best_value(self):
        values = [r.mean() for r in self._records.values()]
        if not values:
            return None
        if self._direction == "min":
            return np.nanmin(values)
        return np.nanmax(values)

    def get_best_step(self):
        best_value = self.get_best_value()
        if best_value is None:
            return None
        for r in self._records.values():
            if r.mean() == best_value:
                return r.step

    def get_statistics(self):
        records = self.records
        records_values = [r.mean() for r in records]
        if not len(records_values):
            return {}
        return {
            "min": float(np.nanmin(records_values)),
            "max": float(np.nanmax(records_values)),
            "mean": float(np.nanmean(records_values)),
            "median": float(np.nanmedian(records_values)),
            "var": float(np.nanvar(records_values)),
            "std": float(np.nanstd(records_values)),
        }

    def get_state(self):
        state = {}
        state["direction"] = self._direction
        state["records"] = [r.get_state() for r in self.records]
        return state

    @classmethod
    def from_state(cls, state):
        records = cls(state["direction"])
        records.records = [MetricRecord.from_state(r) for r in state["records"]]
        return records


class MetricsRecorder:
    """
    Record the values for all metrics.
    """

    def __init__(self, metrics=None):
        self._records = {}
        self.register_metrics(metrics)

    @property
    def records(self):
        return self._records

    def exists(self, name):
        return name in self._records

    def register_metrics(self, metrics=None):
        metrics = metrics or []
        for metric in metrics:
            self.register(metric.name)

    def register(self, name, direction=None):
        if self.exists(name):
            raise ValueError(f"Metric {name} have been registered.")
        if direction is None:
            direction = "min"
        self._records[name] = MetricRecords(direction)

    def update(self, name, value, step=0):
        value = float(value)
        if not self.exists(name):
            self.register(name)

        prev_best = self._records[name].get_best_value()
        self._records[name].update(value, step=step)
        new_best = self._records[name].get_best_value()

        improved = new_best != prev_best
        return improved

    def get_records(self, name):
        return self._records[name].records

    def set_records(self, name, records):
        if not self.exists(name):
            self.register(name)
        self._records[name].records = records

    def get_best_value(self, name):
        return self._records[name].get_best_value()

    def get_best_step(self, name):
        return self._records[name].get_best_step()

    def get_statistics(self, name):
        return self._records[name].get_statistics()

    def get_state(self):
        return {
            "metrics": {
                name: metric_records.get_state()
                for name, metric_records in self._records.items()
            }
        }

    @classmethod
    def from_state(cls, state):
        recorder = cls()
        recorder._records = {
            name: MetricRecords.from_state(metric_records)
            for name, metric_records in state["metrics"].items()
        }
        return recorder
