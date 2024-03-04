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

import logging
import time

import etcd3


class ETCDClient:
    def __init__(self, host, port, retry_times=20):
        self.retry_times = retry_times
        times = 0
        while times < self.retry_times:
            try:
                self.client = etcd3.client(host=host, port=port)
                break
            except Exception as e:
                times += 1
                logging.info(
                    f"Initialize etcd client failed with exception {e}, retry after 1 second."
                )
                time.sleep(1)

        if times >= self.retry_times:
            raise ValueError(
                f"Initialize etcd client failed failed after {self.retry_times} times."
            )

    def put(self, key, value, lease=None, prev_kv=False):
        times = 0
        while times < self.retry_times:
            try:
                return self.client.put(key, value, lease, prev_kv)
            except Exception as e:
                times += 1
                logging.info(
                    f"Put failed with exception {e}, retry after 1 second."
                )
                time.sleep(1)

        if times >= self.retry_times:
            raise ValueError(f"Put failed after {self.retry_times} times.")

    def get(self, key):
        times = 0
        while times < self.retry_times:
            try:
                return self.client.get(key)
                break
            except Exception as e:
                times += 1
                logging.info(
                    f"Get {key} failed with exception {e}, retry after 1 second."
                )
                time.sleep(1)

        if times >= self.retry_times:
            raise ValueError(
                f"Get {key} failed after {self.retry_times} times."
            )

    def delete(self, key, prev_kv=False, return_response=False):
        times = 0
        while times < self.retry_times:
            try:
                return self.client.delete(key, prev_kv, return_response)
                break
            except Exception as e:
                times += 1
                logging.info(
                    f"Delete {key} failed with exception {e}, retry after 1 second."
                )
                time.sleep(1)

        if times >= self.retry_times:
            raise ValueError(
                f"Delete {key} failed after {self.retry_times} times."
            )

    def get_prefix(self, key_prefix, sort_order=None, sort_target='key'):
        times = 0
        while times < self.retry_times:
            try:
                return self.client.get_prefix(key_prefix)
                break
            except Exception as e:
                times += 1
                logging.info(
                    f"Get prefix {key_prefix} failed with exception {e}, retry after 1 second."
                )
                time.sleep(1)

        if times >= self.retry_times:
            raise ValueError(
                f"Get prefix {key_prefix} failed after {self.retry_times} times."
            )

    def delete_prefix(self, prefix):
        times = 0
        while times < self.retry_times:
            try:
                return self.client.delete_prefix(prefix)
                break
            except Exception as e:
                times += 1
                logging.info(
                    f"Delete prefix {prefix} failed with exception {e}, retry after 1 second."
                )
                time.sleep(1)

        if times >= self.retry_times:
            raise ValueError(
                f"Delete prefix {prefix} failed after {self.retry_times} times."
            )

    def lease(self, ttl, lease_id=None):
        times = 0
        while times < self.retry_times:
            try:
                return self.client.lease(ttl, lease_id)
                break
            except Exception as e:
                times += 1
                logging.info(
                    f"Lease failed with exception {e}, retry after 1 second."
                )
                time.sleep(1)

        if times >= self.retry_times:
            raise ValueError(f"Lease failed after {self.retry_times} times.")

    def add_watch_prefix_callback(self, key_prefix, callback, **kwargs):
        times = 0
        while times < self.retry_times:
            try:
                return self.client.add_watch_prefix_callback(
                    key_prefix, callback, **kwargs
                )
                break
            except Exception as e:
                times += 1
                logging.info(
                    f"Add watch prefix callback failed with exception {e}, retry after 1 second."
                )
                time.sleep(1)

        if times >= self.retry_times:
            raise ValueError(
                f"Add watch prefix callback failed after {self.retry_times} times."
            )

    def cancel_watch(self, watch_id):
        times = 0
        while times < self.retry_times:
            try:
                return self.client.cancel_watch(watch_id)
                break
            except Exception as e:
                times += 1
                logging.info(
                    f"Cancel watch failed with exception {e}, retry after 1 second."
                )
                time.sleep(1)

        if times >= self.retry_times:
            raise ValueError(
                f"Cancel watch failed after {self.retry_times} times."
            )
