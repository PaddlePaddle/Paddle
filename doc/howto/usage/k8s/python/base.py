#-*-coding:utf-8-*-
from kubernetes import client
class PaddlePaddleJobBase(object):
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
    def __init__(self,namespace, name):
        self._namespace = namespace
        self._name = name
        self.__init_job_struct()
        self._image = None
        self._name = None

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

    def update_command(self, command):
        self._job.spec.template.spec.containers[0].command = command

    def update_image(self, image):
        self._job.spec.template.spec.containers[0].image = image

    def __init_job_struct(self):
        self._job = client.V1Job()
        job_spec = client.V1JobSpec()
        job_metadata = client.V1ObjectMeta()
        pod_template_spec = client.V1PodTemplateSpec()
        pod_spec = client.V1PodSpec()
        pod_spec.restart_policy = "Never"

        container = client.V1Container()
        container.name = self._name
        pod_spec.containers = [container]
        pod_template_spec.spec = pod_spec
        job_spec.template=pod_template_spec

        self._job.spec = job_spec
        job_metadata.name = self._name
        job_metadata.namespace = self._namespace

        self._job.metadata = job_metadata

    @property
    def job(self):
        return self._job
if __name__ == "__main__":
    job = PaddlePaddleJobBase("test-namespace", "test-name")

    volume_mount = client.V1VolumeMount()
    volume_mount.name = "paddle-k8s-test"
    volume_mount.mount_path = "/dev/null"

    volume = client.V1Volume()
    configmap_source = client.V1ConfigMapVolumeSource()
    configmap_source.name = "paddle-k8s-test"

    volume.config_map = configmap_source
    volume.name = "paddle-k8s-test"
    job.append_volume(volume_mount, volume)
    print job.job
