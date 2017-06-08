#-*-coding:utf-8-*-
import time
from kubernetes import client
from ..configuration import Configuration

# If any onf the pod is runing, the status is JOB_STATUS_ACTIVE
JOB_STATUS_ACTIVE = 0
# If a specified number of successful completions is reached, the status is JOB_STATUS_COMPLETE
JOB_STATUS_COMPLETE = 1

DEFAULT_CHECK_JOB_INTERVAL = 10


class KubeJob(object):
    """
    A special kubernetes job struct designed for PaddlePaddle,
    such as there is only one container in pod...
    You can describe a PaddlePaddle job in a more concise way.


    Job Basic Struct
    Job
      |-job_metadata
      `-job_spec
        `-pod_template
            |-pod_metadata
            `-pod_spec
                `-container
                    |-image
                    `-env

    """

    def __init__(self, namespace, name):
        self.conf = Configuration()
        self._namespace = self.conf.namespace
        self._name = name
        # Job will be execute command
        self._command = []

        # Docker image of the job
        self._docker_image = ""

        # When a specified number of successful completions is reached,
        # the job itself is complete, default number is 1
        self._completions = 1

        self._init_job_struct()

    @property
    def job(self):
        return self._job

    @property
    def completions(self):
        return self._completions

    @completions.setter
    def completions(self, completions):
        self.completions = completions

    def append_volume(self, volume_mount, volume):
        container = self._job.spec.template.spec.containers[0]
        if not container.volume_mounts:
            container.volume_mounts = [volume_mount]
        else:
            container.volume_mounts.append(volume_mount)
        pod_spec = self._job.spec.template.spec
        if not pod_spec.volumes:
            pod_spec.volumes = [volume]
        else:
            pod_spec.volumes.append(volume)

    def append_env(self, name, value="", value_from=None):
        env = client.V1EnvVar()
        if value_from:
            env.name = name
            env.value_from = value_from
        else:
            env.name = name
            env.value = value
        if not self._job.spec.template.spec.containers[0].env:
            self._job.spec.template.spec.containers[0].env = [env]
        else:
            self._job.spec.template.spec.containers[0].env.append(env)

    @property
    def docker_image(self):
        return self._docker_image

    @docker_image.setter
    def docker_image(self, docker_image):
        self._job.spec.template.spec.containers[0].image = docker_image

    @property
    def command(self):
        return self._command

    @command.setter
    def command(self, command):
        self._job.spec.template.spec.containers[0].command = command

    def get_job_status(self):
        api_instance = client.BatchV1Api()
        api_response = api_instance.read_namespaced_job_status(self._name,
                                                               self._namespace)
        if api_response.status.succeeded == self.completions:
            return JOB_STATUS_COMPLETE
        else:
            return JOB_STATUS_ACTIVE

    def sync_wait(self, timeout=None):
        remaind = timeout
        while True:
            state = self.get_job_status()
            # Check timeout
            if remaind and remaind < 0:
                # TODO: how to deal with timeouted job? kill it or waiting?
                print "Job: [%s] run timeout!!" % self._name
                break
            if state == JOB_STATUS_ACTIVE:
                print "Job: [%s] is running, waitting for 10 seconds..." % self._name
                remaind -= DEFAULT_CHECK_JOB_INTERVAL
                time.sleep(10)
                continue
            elif state == JOB_STATUS_COMPLETE:
                print "Job: [%s] has alread completed" % self._name
                break
            else:
                print "Job: [%s] run failed" % self._name
                break

    def _init_job_struct(self):
        self._job = client.V1Job()
        job_spec = client.V1JobSpec()
        job_metadata = client.V1ObjectMeta()
        job_metadata.namespace = self._namespace
        job_metadata.name = self._name
        pod_template_spec = client.V1PodTemplateSpec()
        pod_spec = client.V1PodSpec()
        pod_spec.restart_policy = "Never"

        container = client.V1Container()
        container.name = self._name
        pod_spec.containers = [container]
        pod_template_spec.spec = pod_spec
        job_spec.template = pod_template_spec

        self._job.spec = job_spec
        self._job.metadata = job_metadata


if __name__ == "__main__":
    job = KubeJob(namespace="yancey", name="paddle-cluster-job")
    print job.get_job_status()
