#!/bin/env python
#-*-coding:utf-8-*-
from kubernetes import client, config
from configuration import Configuration

import job


def init(server, namespace):
    #Configuration().namespace(namespace)
    Configuration().namespace = namespace
    client.configuration.host = server
    #config.load_kube_config()
