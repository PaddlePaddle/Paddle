import logging
import BaseHTTPServer
import SimpleHTTPServer
import time
import threading
import socket

def get_logger(name, level, fmt):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.FileHandler('http.log', mode='w')
    formatter = logging.Formatter(fmt=fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


_http_server_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class KVHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    def do_GET(self):
        log_str = "GET " + self.address_string() + self.path
        paths = self.path.split('/')
        if len(paths) < 3:
            print('len of request path must be 3: ' + self.path)
            self.send_status_code(400)
            return
        _, scope, key = paths
        with self.server.kv_lock:
            value = self.server.kv.get(scope, {}).get(key)
        if value is None:
            log_str += ' , key not found: ' + key
            self.send_status_code(404)
        else:
            log_str += ' , key found: ' + key
            self.send_response(200)
            self.send_header("Content-Length", str(len(value)))
            self.end_headers()
            self.wfile.write(value)
        _http_server_logger.info(log_str)
    
    def do_PUT(self):
        log_str = "PUT " + self.address_string() + self.path
        paths = self.path.split('/')
        if len(paths) < 3:
            print('len of request path must be 3: ' + self.path)
            self.send_status_code(400)
            return
        _, scope, key = paths
        content_length = int(self.headers['Content-Length'])
        try:
            value = self.rfile.read(content_length)
        except:
            print("receive error invalid request")
            self.send_status_code(404)
            return
        with self.server.kv_lock:
            if self.server.kv.get(scope) is None:
                self.server.kv[scope] = {}
            self.server.kv[scope][key] = value
        self.send_status_code(200)
        _http_server_logger.info(log_str)

    def do_DELETE(self):
        log_str = "DELETE " + self.address_string() + self.path
        paths = self.path.split('/')
        if len(paths) < 3:
            print('len of request path must be 3: ' + self.path)
            self.send_status_code(400)
            return
        _, scope, key = paths
        with self.server.delete_kv_lock:
            if self.server.delete_kv.get(scope) is None:
                self.server.delete_kv[scope] = []
            self.server.delete_kv[scope].append(key)
        self.send_status_code(200)
        _http_server_logger.info(log_str)

    def log_message(self, format, *args):
        pass

    def send_status_code(self, code):
        self.send_response(code)
        self.send_header("Content-Length", 0)
        self.end_headers()


class KVHTTPServer(BaseHTTPServer.HTTPServer, object):
    def __init__(self, port, handler):
        super(KVHTTPServer, self).__init__(('', port), handler)
        self.delete_kv_lock = threading.Lock()
        self.delete_kv = {}
        self.kv_lock = threading.Lock()
        self.kv = {}


class KVServer:
    def __init__(self, port):
        self.http_server = KVHTTPServer(port, KVHandler)
        self.listen_thread = None

    def start(self):
        self.listen_thread = threading.Thread(
            target=lambda: self.http_server.serve_forever())
        self.listen_thread.start()

    def stop(self):
        self.http_server.shutdown()
        self.listen_thread.join()
        self.http_server.server_close()
