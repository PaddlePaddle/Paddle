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

import time

import httpx


class KVClient:
    def __init__(self, endpoint='localhost:2379'):
        self.endpoint = (
            endpoint if endpoint.startswith("http://") else f"http://{endpoint}"
        )

    def put(self, key, value):
        key = key if key.startswith('/') else f"/{key}"
        u = f"{self.endpoint}{key}"
        try:
            r = httpx.post(u, data=value, timeout=None, follow_redirects=True)
            if r.status_code == 200:
                return True
            else:
                return False
        except:
            return False

    def get(self, key):
        key = key if key.startswith('/') else f"/{key}"
        u = f"{self.endpoint}{key}"
        try:
            r = httpx.get(u, timeout=None, follow_redirects=True)
            if r.status_code == 200:
                ret = r.json()
                return ret.get(key, '')
            else:
                return "error"
        except:
            return ""

    def get_prefix(self, key):
        key = key if key.startswith('/') else f"/{key}"
        u = f"{self.endpoint}{key}"
        try:
            r = httpx.get(u, timeout=None, follow_redirects=True)
            if r.status_code == 200:
                return r.json()
        except:
            return ""

    def delete(self, key):
        key = key if key.startswith('/') else f"/{key}"
        u = f"{self.endpoint}{key}"
        try:
            r = httpx.delete(u, timeout=None, follow_redirects=True)
            if r.status_code == 200:
                return True
            else:
                return False
        except:
            return False

    def wait_server_ready(self, timeout=3):
        end = time.time() + timeout
        while time.time() < end:
            if self.get("/healthy") == "ok":
                return True


if __name__ == '__main__':
    cli = KVClient("http://localhost:8090")
    data = {"/workers/1": "rank1", "/workers/2": "rank2"}
    for k, v in data.items():
        cli.put(k, v)
    x = cli.get_prefix("/workers")
    print(x)
    for k, v in data.items():
        assert x[k] == v

    cli.put("key", "value")
    print(cli.get("key"))
    assert cli.get("key") == "value"
    cli.delete("key")
    print(cli.get("/key"))
    print(cli.get("/healthy"))
    assert cli.get("/healthy") == "ok"
