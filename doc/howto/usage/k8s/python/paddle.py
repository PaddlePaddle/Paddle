#/bin/env python
#-*-coding:utf-8-*-
from kubernetes import client,config
from pprint import pprint
from bash_script_handler import BashScriptHandler
from base import PaddlePaddleJobBase
import os
# Load local kubernetes config file located '~/.kube/config'
config.load_kube_config()


JOB_STATUS_ACTIVE = "Active"
JOB_STATUS_COMPLETE = "Complete"
JOB_STATUS_FAILED = "Failed"
JOB_STATUS_UNKNOW = "Unknow"

class PrepareTrainingDataJob(object):
    def __init__(self,namespace, name):
        self._namespace = namespace
        self._name = name
        self._bash_handler = BashScriptHandler(namespace, name)
        self._job = PaddlePaddleJobBase(namespace, name)

    def upload_local_file(self, filename, claim_name):
        self._filename = filename
        self._claim_name = claim_name
        self._bash_handler.init_with_bash_filename(filename)

        self._bash_handler.run()

    def run(self, trainner_count):
        # Run bash scritp handler

        script_mount_path = "/opt/scripts"
        script_mount_name = "prepare-script"
        k8s_filepath = os.path.join(script_mount_path, os.path.basename(self._filename))

        # Describe configmap volume
        configmap_volume_mount = client.V1VolumeMount()
        configmap_volume_mount.name = script_mount_name
        configmap_volume_mount.mount_path = script_mount_path

        configmap_volume = client.V1Volume()
        configmap_source = client.V1ConfigMapVolumeSource()
        configmap_source.name = self._name

        configmap_volume.config_map = configmap_source
        configmap_volume.name = script_mount_name

        # Describe NFS volume
        nfs_volume_mount = client.V1VolumeMount()
        nfs_volume_mount.mount_path = "/opt/data"
        nfs_volume_mount.name = "nfs"

        nfs_volume = client.V1Volume()
        nfs_volume.name = "nfs"
        nfs_volume.persistent_volume_claim = client.V1PersistentVolumeClaimVolumeSource()
        nfs_volume.persistent_volume_claim.claim_name = self._claim_name

        self._job.update_image("yancey1989/paddle-bash-handler:0.2")
        self._job.update_command(["/bin/bash", k8s_filepath])
        self._job.append_volume(configmap_volume_mount, configmap_volume)
        self._job.append_volume(nfs_volume_mount, nfs_volume)
        self._job.append_env("JOB_NAME", self._name)
        self._job.append_env("JOB_PATH", "/opt/data")
        self._job.append_env("SPLIT_COUNT", str(trainner_count))

        api_instance = client.BatchV1Api()
        response = api_instance.create_namespaced_job(self._namespace, self._job.job)

    def get_job_status(self):
        api_instance = client.BatchV1Api()
        api_response = api_instance.read_namespaced_job_status(self._name, self._namespace)
        if api_response.status.succeeded == 1:
            return JOB_STATUS_COMPLETE
        elif api_response.status.active:
            return JOB_STATUS_ACTIVE
        elif api_response.status.failed:
            return JOB_STATUS_FAILED
        else:
            return JOB_STATUS_UNKNOW

class PaddlePaddleCluster(object):
    def __init__(self, namespace, name):
        self._namespace = namespace
        self._name = name
        self._prepare_training_data = PrepareTrainingDataJob(namespace, "prepare-"+name)

    @property
    def prepare_training_data(self):
        return self._prepare_training_data

    def get_job_status(self, namespace, job_id):
        pass

    def run(self):
        """
        Deploy a job on kubernetes and run the paddle distributed train process.
        """
        pass
if __name__ == "__main__":

    api_instance = client.CoreV1Api()
    api_response = api_instance.list_namespaced_config_map("yancey")
    pprint(api_response)
