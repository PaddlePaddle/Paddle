# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import os
import socket
import threading
import time
import unittest
from http.server import ThreadingHTTPServer

from paddle.distributed.launch.utils.kv_client import KVClient
from paddle.distributed.launch.utils.kv_server import KVHandler, KVServer


class KVServerTestBase(unittest.TestCase):
    def setUp(self):
        # KVClient talks to the server over httpx, which honors *_PROXY env vars.
        # On CI a global HTTP proxy is configured, so without a bypass the
        # loopback request gets hijacked by the proxy (which answers 500 for
        # 127.0.0.1) instead of reaching our server. Drop the proxy vars and
        # force a bypass for the whole test, restoring the env afterwards.
        self._proxy_env_backup = {}
        for var in (
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "no_proxy",
            "NO_PROXY",
        ):
            self._proxy_env_backup[var] = os.environ.pop(var, None)
        os.environ["no_proxy"] = "*"
        os.environ["NO_PROXY"] = "*"
        self.addCleanup(self._restore_proxy_env)

    def _restore_proxy_env(self):
        for var, val in self._proxy_env_backup.items():
            if val is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = val

    def _start_server(self):
        # port 0 -> let the OS pick a free ephemeral port to avoid CI collisions.
        server = KVServer(0)
        # KVServer binds to ('', port), so server_address[0] is 0.0.0.0. That is
        # a valid *bind* address but not a valid *connect* target on Windows
        # (WinError 10049), so always reach the server via the loopback address.
        self.host = "127.0.0.1"
        self.port = server.server_address[1]
        server.start()
        self.addCleanup(self._safe_stop, server)
        client = KVClient(f"127.0.0.1:{self.port}")
        self.assertTrue(
            client.wait_server_ready(timeout=10), "KV server never became ready"
        )
        return server, client

    @staticmethod
    def _safe_stop(server):
        if not server.stopped:
            server.stop()


class TestKVServerConcurrent(KVServerTestBase):
    def test_concurrent_put_get_prefix(self):
        # HTTPMaster.sync_peers relies on many launchers concurrently doing
        # put()/get_prefix() against a single KV store. Now that KVServer is a
        # ThreadingHTTPServer, each connection is handled on its own thread; the
        # shared kv dict is guarded by kv_lock. Hammer it from many threads and
        # assert every write lands exactly once with no lost/corrupted values.
        server, _ = self._start_server()
        num_clients = 32

        # Guard the concurrent thread model itself: a regression back to a
        # single-threaded HTTPServer would reintroduce the head-of-line blocking
        # this fix removed.
        self.assertIsInstance(server, ThreadingHTTPServer)

        errors = []
        barrier = threading.Barrier(num_clients)

        def worker(idx):
            try:
                # All threads block here first so the puts really overlap.
                barrier.wait(timeout=10)
                client = KVClient(f"127.0.0.1:{self.port}")
                self.assertTrue(client.put(f"/workers/{idx}", f"rank{idx}"))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(num_clients)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(errors, [], f"concurrent workers raised: {errors}")

        reader = KVClient(f"127.0.0.1:{self.port}")
        result = reader.get_prefix("/workers")
        self.assertIsInstance(result, dict)
        expected = {f"/workers/{i}": f"rank{i}" for i in range(num_clients)}
        self.assertEqual(result, expected)


class TestKVServerRequestTimeout(KVServerTestBase):
    def setUp(self):
        super().setUp()
        # The production timeout is 30s; override it to keep the test fast while
        # still exercising the exact same StreamRequestHandler.setup() ->
        # handle_one_request() timeout-to-close path.
        self._orig_timeout = KVHandler.timeout
        KVHandler.timeout = 0.5
        self.addCleanup(self._restore_timeout)

    def _restore_timeout(self):
        KVHandler.timeout = self._orig_timeout

    def test_half_open_connection_is_released_after_timeout(self):
        # Guard the production configuration: a None/unset handler timeout is
        # exactly the bug (rfile.readline() would block forever), so the value
        # must stay a positive number even though we shrink it here for speed.
        self.assertIsNotNone(
            self._orig_timeout, "KVHandler.timeout must be configured"
        )
        self.assertGreater(self._orig_timeout, 0)

        # A peer that connects but never finishes sending its request line used
        # to wedge the handler forever in rfile.readline(). With the socket
        # timeout the handler must give up, close the connection, and free its
        # worker thread. Observable contract: the client sees EOF shortly after
        # the timeout, and the server keeps serving other requests.
        server, client = self._start_server()

        sock = socket.create_connection((self.host, self.port), timeout=10)
        self.addCleanup(sock.close)
        try:
            start = time.time()
            sock.settimeout(10)
            # Send nothing: leave the request line unfinished on purpose.
            data = sock.recv(64)
            elapsed = time.time() - start
        finally:
            sock.close()

        # Server-side timeout closed the connection -> clean EOF, not a hang.
        self.assertEqual(
            data, b"", "server did not close the stalled connection"
        )
        self.assertLess(
            elapsed,
            5,
            "stalled connection was not released promptly after timeout",
        )

        # The stalled connection must not have wedged the store: a normal
        # request still succeeds (proves the worker thread was freed and the
        # server is still accepting).
        self.assertTrue(client.put("/after_timeout", "ok"))
        self.assertEqual(client.get("/after_timeout"), "ok")


class TestKVServerStop(KVServerTestBase):
    def test_stop_is_clean_and_idempotent_state(self):
        server, client = self._start_server()
        self.assertTrue(server.started)
        self.assertFalse(server.stopped)
        # Sanity: it is actually serving before we stop it.
        self.assertTrue(client.put("/k", "v"))
        self.assertEqual(client.get("/k"), "v")

        # stop() must shut down serve_forever, join the listener thread and
        # close the socket without hanging.
        stop_done = threading.Event()

        def _stop():
            server.stop()
            stop_done.set()

        stopper = threading.Thread(target=_stop)
        stopper.start()
        stopper.join(timeout=15)
        self.assertTrue(stop_done.is_set(), "server.stop() hung")

        self.assertTrue(server.stopped)
        self.assertFalse(server.listen_thread.is_alive())

        # The port must be released so it can be bound again.
        rebind = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        rebind.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            rebind.bind((self.host, self.port))
        finally:
            rebind.close()


if __name__ == '__main__':
    unittest.main()
