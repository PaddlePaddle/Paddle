#!flask/bin/python
from flask import Flask, jsonify, abort, request, make_response, url_for
from contextlib import closing
import socket
from flask import request
import random
import threading
import time
import argparse
import copy
import functools
import multiprocessing
import os
import sys
import utils
import subprocess
from utils import logger

app = Flask(__name__, static_url_path="")

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(utils.add_arguments, argparser=parser)

add_arg('host', str, None, "host ip")
add_arg('port', int, None, "host port")


class ScopeKV(object):
    def __init__(self):
        self._lock = threading.Lock()
        self._scope = {}

    def get_value(self, scope, key):
        with self._lock:
            if scope in self._scope and key in self._scope[scope]:
                return self._scope[scope][key]
        return None

    def post(self, scope, key, data):
        with self._lock:
            if scope not in self._scope[scope]:
                self._scope[scope] = {}
            self.job[scope][key] = data

    def get_scope(self, scope):
        with self._lock:
            if scope in self._scope[scope]:
                return self._scope[scope]
        return None


kv_store = ScopeKV()


@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({'error': 'Bad request'}), 400)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.route('/rest/1.0/get/pod', methods=['GET'])
def get_value():
    try:
        scope = request.args.get('scope')
        key = request.args.get('scope')
        assert scope is not None and key is not None
        value = kv_store.get_value(scope, key)
    except:
        return jsonify({})
    return jsonify({'scope': scope, 'key': key, "value": value})


@app.route('/rest/1.0/get/scope', methods=['GET'])
def get_scope():
    try:
        scope = request.args.get('scope')
        value = kv_store.get_scope(scope)
        assert scope is not None and value is not None
    except:
        return jsonify({})
    return jsonify({'scope': scope, "value": data})


@app.route('/rest/1.0/post/pod', methods=['POST'])
def update_data():
    print("get_request:", request)
    try:
        scope = request.json["scope"]
        key = request.json["key"]
        value = request.json["value"]
        assert scope is not None and value is not None and key is not None
    except:
        return jsonify({'invalid arguments'})
    kv_store.post(scope, key, value)


class HttpServer(object):
    def __init__(self):
        self._proc = None
        self._cmd = None
        self._fn = None

    def start(self, host, port):
        current_env = copy.copy(os.environ.copy())
        current_env.pop("http_proxy", None)
        current_env.pop("https_proxy", None)
        current_env["PYTHONUNBUFFERED"] = "1"

        #self._cmd = [sys.executable, "-m", "paddle.distributed.http_store", "--host={}".format(host), "--port={}".format(port)]
        self._cmd = [
            "stdbuf", "-oL", sys.executable, "-m",
            "paddle.distributed.http_store", "--host={}".format(
                "yq01-sys-hic-v100-box-a223-0155.yq01.baidu.com"),
            "--port={}".format(port)
        ]
        logger.info("start http store:{}".format(self._cmd))
        #self._fn = open("paddle_edl_launch_http_store.log" , "a")
        self._proc = subprocess.Popen(
            self._cmd, env=current_env, stdout=sys.stdout, stderr=sys.stderr)
        #self._proc = subprocess.Popen(self._cmd, stdout=self._fn, stderr=self._fn)

    def stop(self):
        if self._proc and self._proc.is_alive():
            self._proc.terminate()

    def is_alive(self):
        if self._proc is None:
            return False

        if self._proc.poll() is not None:
            return False

        return True

    def join(self):
        while self._proc.poll() is None:  # not teminate
            time.sleep(1)


kv_server = HttpServer()

if __name__ == '__main__':
    args = parser.parse_args()
    assert args.host and args.port, "input host and port"
    logger = utils.get_logger(10)
    app.run(host=args.host, port=args.port, threaded=True, processes=1)
