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

from http.server import HTTPServer
import http.server as SimpleHTTPServer

from multiprocessing import Process

import threading
import json


class KVHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    def do_GET(self):
        with self.server.kv_lock:
            ret = {}
            for k, v in self.server.kv.items():
                if k.startswith(self.path):
                    ret[k] = v.decode(encoding="utf-8")
            if ret:
                self.output(200, json.dumps(ret).encode("utf-8"))
            else:
                self.output(404)

    def do_PUT(self):
        self.do_POST()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'] or 0)
        try:
            value = self.rfile.read(content_length)
            with self.server.kv_lock:
                self.server.kv[self.path] = value
                self.output(200)
                return
        except:
            self.output(500)

    def do_DELETE(self):
        with self.server.kv_lock:
            if self.path in self.server.kv:
                del self.server.kv[self.path]
                self.output(200)
            else:
                self.output(404)

    def output(self, code, value=''):
        self.send_response(code)
        self.send_header("Content-Length", len(value))
        self.send_header("Content-Type", "application/json; charset=utf8")
        self.end_headers()
        if value:
            self.wfile.write(value)

    def log_message(self, format, *args):
        return


class KVServer(HTTPServer, object):
    def __init__(self, port):
        super(KVServer, self).__init__(('', port), KVHandler)
        self.kv_lock = threading.Lock()
        self.kv = {'/healthy': b'ok'}
        self.port = port
        self.stopped = False
        self.started = False

    def start(self):
        self.listen_thread = threading.Thread(target=self.serve_forever)
        self.listen_thread.start()
        self.started = True

    def stop(self):
        self.shutdown()
        self.listen_thread.join()
        self.server_close()
        self.stopped = True


class PKVServer():
    def __init__(self, port):
        self._server = KVServer(port)

    def start(self):
        self.proc = Process(target=self._server.start)
        self.proc.daemon = True
        self.proc.start()

    def stop(self):
        self._server.stop()
        self.proc.join()

    @property
    def started(self):
        return self._server.started

    @property
    def stopped(self):
        return self._server.stopped


if __name__ == '__main__':
    #kv = PKVServer(8090)
    kv = KVServer(8090)
    kv.start()
    import time

    #print("serve at 8090 for 600 s")

    time.sleep(600)
