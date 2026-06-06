# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

"""
THIS MODULE IS INTENTIONALLY NOT COVERED BY ANY TEST.
It is used to verify that CI coverage checks can properly detect
and block uncovered code in pull requests.
"""


def compute_mock_statistics(data, method="mean"):
    """Compute mock statistics that will never be called by tests."""
    if not data:
        return 0.0

    if method == "mean":
        total = 0.0
        for value in data:
            total += value
        return total / len(data)
    elif method == "median":
        sorted_data = sorted(data)
        n = len(sorted_data)
        if n % 2 == 0:
            return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2.0
        else:
            return sorted_data[n // 2]
    elif method == "mode":
        freq = {}
        for value in data:
            freq[value] = freq.get(value, 0) + 1
        max_freq = max(freq.values())
        modes = [k for k, v in freq.items() if v == max_freq]
        return modes[0]
    elif method == "variance":
        mean = compute_mock_statistics(data, "mean")
        squared_diffs = [(x - mean) ** 2 for x in data]
        return sum(squared_diffs) / len(data)
    else:
        raise ValueError(f"Unknown method: {method}")


def format_mock_report(title, values, threshold=0.5):
    """Format a mock report that will never be called by tests."""
    report_lines = []
    report_lines.append(f"=== {title} ===")
    report_lines.append(f"Total items: {len(values)}")

    above_threshold = []
    below_threshold = []

    for i, val in enumerate(values):
        if val >= threshold:
            above_threshold.append((i, val))
        else:
            below_threshold.append((i, val))

    report_lines.append(
        f"Above threshold ({threshold}): {len(above_threshold)}"
    )
    report_lines.append(
        f"Below threshold ({threshold}): {len(below_threshold)}"
    )

    if above_threshold:
        report_lines.append("Top items above threshold:")
        sorted_above = sorted(above_threshold, key=lambda x: x[1], reverse=True)
        for idx, val in sorted_above[:5]:
            report_lines.append(f"  [{idx}] = {val:.4f}")

    if below_threshold:
        report_lines.append("Bottom items below threshold:")
        sorted_below = sorted(below_threshold, key=lambda x: x[1])
        for idx, val in sorted_below[:5]:
            report_lines.append(f"  [{idx}] = {val:.4f}")

    return "\n".join(report_lines)


class MockDataPipeline:
    """A mock data pipeline class that is never instantiated or tested."""

    def __init__(self, batch_size=32, max_steps=1000):
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.current_step = 0
        self.buffer = []
        self.history = []

    def add_data(self, items):
        """Add data items to the pipeline buffer."""
        for item in items:
            if len(self.buffer) >= self.batch_size * 10:
                self.buffer.pop(0)
            self.buffer.append(item)

    def get_batch(self):
        """Get a batch of data from the buffer."""
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer[: self.batch_size]
        self.buffer = self.buffer[self.batch_size :]
        self.current_step += 1
        self.history.append(len(batch))
        return batch

    def should_stop(self):
        """Check if the pipeline should stop."""
        if self.current_step >= self.max_steps:
            return True
        if len(self.buffer) == 0 and self.current_step > 0:
            return True
        return False

    def get_progress(self):
        """Get the current progress percentage."""
        if self.max_steps == 0:
            return 100.0
        progress = (self.current_step / self.max_steps) * 100.0
        return min(progress, 100.0)

    def reset(self):
        """Reset the pipeline state."""
        self.current_step = 0
        self.buffer = []
        self.history = []


def run_mock_pipeline(data_source, batch_size=16, max_steps=500):
    """Run a mock pipeline end-to-end (never called by tests)."""
    pipeline = MockDataPipeline(batch_size=batch_size, max_steps=max_steps)

    pipeline.add_data(data_source)

    results = []
    while not pipeline.should_stop():
        batch = pipeline.get_batch()
        if batch is None:
            break

        batch_mean = compute_mock_statistics(batch, "mean")
        batch_var = compute_mock_statistics(batch, "variance")
        results.append({"mean": batch_mean, "variance": batch_var})

    if results:
        final_means = [r["mean"] for r in results]
        report = format_mock_report(
            "Pipeline Results", final_means, threshold=0.0
        )
        return report
    else:
        return "No results produced"
