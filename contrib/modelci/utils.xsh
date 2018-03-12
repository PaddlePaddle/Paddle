#!/usr/bin/env xonsh
import os
import logging

class log:
    @staticmethod
    def logger():
        return logging.getLogger(__name__)

    @staticmethod
    def info(*args):
        log.logger().info(' '.join([str(s) for s in args]))

    @staticmethod
    def warn(*args):
        log.logger().warning(' '.join([str(s) for s in args]))

    def debug(*args):
        log.logger().debug(' '.join([str(s) for s in args]))


def download(url, dst=None):
    log.warn('download', url, 'to %s' if dst else '')
    curl -o @(dst) @(url)

def pjoin(root, path):
    return os.path.join(root, path)

SUC = True, ""

def CHECK_EQ(a, b, msg=""):
    return CHECK(a, b, "==", msg)
def CHECK_GT(a, b, msg=""):
    return CHECK(a, b, ">", msg)
def CHECK_GE(a, b, msg=""):
    return CHECK(a, b, ">=", msg)
def CHECK_LT(a, b, msg=""):
    return CHECK(a, b, "<", msg)
def CHECK_LE(a, b, msg=""):
    return CHECK(a, b, "<=", msg)

conds = {
    '>' : lambda a,b: a > b,
    '>=': lambda a,b: a >= b,
    '==' : lambda a,b: a == b,
    '<': lambda a,b: a < b,
    '<=': lambda a,b: a <= b,
}
def CHECK(a, b, cond, msg):
    if not conds[cond](a,b):
        return False, "CHECK {} {} {} failed.\n{}".format(a, cond, b, msg)
    return SUC

