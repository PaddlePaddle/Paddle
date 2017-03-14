#!/bin/env python
import os
from kubernetes import client
from kubernetes.client.rest import ApiException
from base import KubeJob
from ..configuration import Configuration

DOCKER_IMAGE = "yancey1989/paddle-bash-handler:0.2"
CONFIG_MAP_MOUNT_PATH = "/opt/data"


class BashJob(KubeJob):
    def __init__(self, name, persistent_volume_claim_name=None):
        self.conf = Configuration()
        self._namespace = self.conf.namespace
        self._name = name
        self._persistent_volume_claim_name = persistent_volume_claim_name

        super(BashJob, self).__init__(self._namespace, name)
        if not persistent_volume_claim_name:
            self._update_pvc_volume(persistent_volume_claim_name)

    def _prepare_config_map(self, filename):
        config_map = client.V1ConfigMap()
        config_map.metadata = client.V1ObjectMeta()
        config_map.metadata.namespace = self._namespace
        config_map.metadata.name = self._name

        with open(filename, "r") as f:
            data = f.read()
            config_map.data = {os.path.basename(filename): data}

        return config_map

    def update_configmap_volume(self, configmap):
        volume_mount_name = "prepare-script"
        # Describe configmap volume
        volume_mount = client.V1VolumeMount()
        volume_mount.name = volume_mount_name
        volume_mount.mount_path = CONFIG_MAP_MOUNT_PATH

        volume = client.V1Volume()
        source = client.V1ConfigMapVolumeSource()
        source.name = self._name

        volume.config_map = source
        volume.name = volume_mount_name

        self.append_volume(volume_mount, volume)

    def _update_pvc_volume(self, claim_name):
        # Describe NFS volume
        volume_mount = client.V1VolumeMount()
        volume_mount.mount_path = "/opt/data"
        volume_mount.name = "pvc-mnt-" + self._name

        volume = client.V1Volume()
        volume.name = "pvc-mnt-" + self._name
        volume.persistent_volume_claim = client.V1PersistentVolumeClaimVolumeSource(
        )
        volume.persistent_volume_claim.claim_name = claim_name

        self.append_volume(volume_mount, volume)

    def run(self, filename, trainner_count, docker_image=DOCKER_IMAGE):
        # Step1: patch configmap
        k8s_filename = os.path.join(CONFIG_MAP_MOUNT_PATH,
                                    os.path.basename(filename))
        config_map = self._prepare_config_map(filename)
        api_instance = client.CoreV1Api()
        try:
            config_maps = api_instance.list_namespaced_config_map(
                self._namespace)
            updated = False
            for item in config_maps.items:
                if self._name == item.metadata.name:
                    updated = True
                    api_instance.patch_namespaced_config_map(
                        self._name, self._namespace, config_map)
                    break
            if not updated:
                api_instance.create_namespaced_config_map(self._namespace,
                                                          config_map)
        except ApiException as e:
            print("Exception when calling CoreV1Api: %s\n" % e)

        # Step2: create job to run bash script
        k8s_filename = os.path.join(CONFIG_MAP_MOUNT_PATH,
                                    os.path.basename(filename))
        self.docker_image = docker_image
        self.command = ["/bin/bash", k8s_filename]
        self.update_configmap_volume(config_map)
        self.append_env("JOB_NAME", self._name)
        self.append_env("JOB_PATH", "/opt/data")
        self.append_env("SPLIT_COUNT", str(trainner_count))
        api_instance = client.BatchV1Api()
        api_instance.create_namespaced_job(self._namespace, self._job)
