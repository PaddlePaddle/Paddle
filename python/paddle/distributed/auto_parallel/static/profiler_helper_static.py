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
    color_map = {
        "forward": "thread_state_running",  # RGB: 126, 200, 148
        "backward": "rail_idle",  # RGB: 238, 142, 0
        "optimizer": "rail_response",  # RGB: 238, 142, 0
        "default": "thread_state_unknown",  # RGB: 199, 155, 125
    }
    step = 0
    for match in matches:
        job_id, job_type, micro_batch_id, job_start_time, job_end_time = match
        if job_type in ["lr"]:
            continue

        if job_type == "forward" and int(micro_batch_id) == 0:
            if step > 0:
                event_step_stop = {
                    "name": "Step " + str(step - 1),
                    "cat": "Step",
                    "ph": "E",
                    "ts": float(job_start_time.strip()) * 1000,
                    "pid": "Main",
                    "tid": "Step" + str(device_id),
                    "cname": color_map["default"],
                }
                events.append(event_step_stop)

            event_step_start = {
                "name": "Step " + str(step),
                "cat": "Step",
                "ph": "B",
                "ts": float(job_start_time.strip()) * 1000,
                "pid": "Main",
                "tid": "Step" + str(device_id),
                "cname": color_map["default"],
            }
            events.append(event_step_start)

            step += 1

        event_start = {
            "name": job_type[0].upper() + "_" + str(job_id),
            "cat": job_type,
            "ph": "B",
            "ts": float(job_start_time.strip()) * 1000,
            "pid": "Main",
            "tid": "GPU" + str(device_id),
            "cname": color_map[job_type],
        }
        event_end = {
            "name": job_type[0].upper() + "_" + str(job_id),
            "cat": job_type,
            "ph": "E",
            "pid": "Main",
            "ts": float(job_end_time.strip()) * 1000,
            "tid": "GPU" + str(device_id),
            "cname": color_map[job_type],
        }
        events.append(event_start)
        events.append(event_end)

    event_step_end = {
        "name": "Step " + str(step),
        "cat": "Step",
        "ph": "E",
        "ts": events[-1]["ts"],
        "pid": "Main",
        "tid": "Step" + str(device_id),
        "cname": color_map["default"],
    }
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

    # support Perfetto format
    save_path = os.path.join(args.log_dir, "pipeline_profile_perfetto.json")
    for i in range(len(args.devices.split(","))):
        all_events.extend(
            [
                {
                    "args": {"name": "Step"},
                    "cat": "__metadata",
                    "name": "thread_name",
                    "ph": "M",
                    "pid": "Main",
                    "tid": i * 2,
                    "ts": 0,
                },
                {
                    "args": {"name": f"GPU:{i}"},
                    "cat": "__metadata",
                    "name": "thread_name",
                    "ph": "M",
                    "pid": "Main",
                    "tid": i * 2 + 1,
                    "ts": 0,
                },
            ]
        )
    json_str = json.dumps({"traceEvents": all_events})
    for i in range(len(args.devices.split(","))):
        json_str = json_str.replace(f'"Step{i}"', f'{i * 2}')
        json_str = json_str.replace(f'"GPU{i}"', f'{i * 2 + 1}')

    with open(save_path, "w") as f:
        f.write(json_str)
    print(f"Save pipeline profile to {save_path}")


if __name__ == "__main__":
    main()
