#!/bin/env python
from kubernetes import client, config


class PaddleCloudConfiguration(object):
    def __init__(self):
        self._namespace = ""

    @property
    def namespace(self):
        return self._namespace

    @namespace.setter
    def namespace(self, namespace):
        self._namespace = namespace


conf = PaddleCloudConfiguration()


def Configuration():
    global conf
    return conf
