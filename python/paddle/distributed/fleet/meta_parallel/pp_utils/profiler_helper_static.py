# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import json
import os
import re
from argparse import ArgumentParser

import paddle


def parse_args():
    parser = ArgumentParser()
    device_count = paddle.device.cuda.device_count()
    all_devices = ",".join([str(i) for i in range(device_count)])
    parser.add_argument("--devices", type=str, default=all_devices)
    parser.add_argument(
        "--log_dir", type=str, default="build/Testing/Temporary/"
    )
    args = parser.parse_args()
    return args


def process_log_data(log_data, device_id):
    log_pattern = r'.*?Profiler Info: Job \((\d+)\), type = (\w+), micro_batch_id = (\d+), job_start_time = (\d+.\d+), job_end_time = (\d+.\d+)'
    matches = re.findall(log_pattern, log_data)
    events = []
    for match in matches:
        job_id, job_type, micro_batch_id, job_start_time, job_end_time = match
        if job_type in ["lr"]:
            continue

        event_start = {
            "name": job_type[0].upper() + "_" + str(job_id),
            "cat": job_type,
            "ph": "B",
            "ts": float(job_start_time.strip()) * 1000,
            "pid": "Main",
            "tid": "GPU: " + str(device_id),
        }
        event_end = {
            "name": job_type[0].upper() + "_" + str(job_id),
            "cat": job_type,
            "ph": "E",
            "pid": "Main",
            "ts": float(job_end_time.strip()) * 1000,
            "tid": "GPU: " + str(device_id),
        }
        events.append(event_start)
        events.append(event_end)

    return events


def main():
    args = parse_args()
    all_events = []
    for device_id in args.devices.split(","):
        print(f"Process device {device_id}")
        device_id = int(device_id)
        log_file = os.path.join(args.log_dir, "workerlog." + str(device_id))
        with open(log_file, "r") as f:
            log_data = f.read()
        events = process_log_data(log_data, device_id)
        all_events.extend(events)

    save_path = os.path.join(args.log_dir, "pipeline_profile.json")
    with open(save_path, "w") as f:
        f.write(json.dumps({"traceEvents": all_events}))
    print(f"Save pipeline profile to {save_path}")


if __name__ == "__main__":
    main()
