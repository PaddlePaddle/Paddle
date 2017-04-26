import kubernetes
from kubernetes import client, config
import os

__all__ = ["PaddleJob"]

## kubernetes default configuration
DEFAULT_GLUSTERFS_ENDPOINT = "glusterfs-cluster"
GLUSTERFS_MOUNT_PATH = "/mnt/glusterfs"


class PaddleJob(object):
    """
        PaddleJob
    """

    def __init__(self,
                 trainers,
                 pservers,
                 base_image,
                 glusterfs_volume,
                 input,
                 output,
                 job_name,
                 trainer_package_path,
                 entry_point,
                 namespace="default",
                 use_gpu=False,
                 num_gradient_servers=1,
                 port=7164,
                 ports_num=1,
                 ports_num_for_sparse=1,
                 env={}):
        self.trainers = trainers
        self.pservers = pservers
        self.base_iamge = base_image
        self.glusterfs_volume = glusterfs_volume
        self.input = input
        self.output = output
        self.job_name = job_name
        self.namespace = namespace
        self.ports_num = ports_num
        self.ports_num_for_sparse = ports_num_for_sparse
        self.port = port
        self.user_env = env
        self.use_gpu = use_gpu
        self.trainer_package_path = trainer_package_path
        self.entry_point = entry_point
        self.num_gradient_servers = num_gradient_servers

    def get_pserver_job_name(self):
        return "%s-pserver" % self.job_name

    def get_trainer_job_name(self):
        return "%s-trainer" % self.job_name

    def get_env(self):
        envs = []
        for k, v in self.user_env.items():
            env = client.V1EnvVar()
            env.name = k
            env.value = v
            envs.append(env)
        envs.append(
            client.V1EnvVar(
                name="PADDLE_JOB_NAME", value=self.job_name))
        envs.append(client.V1EnvVar(name="INPUT", value=self.input))
        envs.append(client.V1EnvVar(name="PORT", value=str(self.port)))
        envs.append(client.V1EnvVar(name="TRAINERS", value=str(self.trainers)))
        envs.append(client.V1EnvVar(name="PSERVERS", value=str(self.pservers)))
        envs.append(
            client.V1EnvVar(
                name="PORTS_NUM", value=str(self.ports_num)))
        envs.append(
            client.V1EnvVar(
                name="PORTS_NUM_FOR_SPARSE",
                value=str(self.ports_num_for_sparse)))
        envs.append(
            client.V1EnvVar(
                name="NUM_GRADIENT_SERVERS",
                value=str(self.num_gradient_servers)))
        envs.append(client.V1EnvVar(name="OUTPUT", value=self.output))
        envs.append(client.V1EnvVar(name="ENTRY_POINT", value=self.entry_point))
        envs.append(
            client.V1EnvVar(
                name="TRAINER_PACKAGE_PATH",
                value=os.path.join(GLUSTERFS_MOUNT_PATH,
                                   self.trainer_package_path.lstrip("/"))))
        envs.append(
            client.V1EnvVar(
                name="NAMESPACE",
                value_from=client.V1EnvVarSource(
                    field_ref=client.V1ObjectFieldSelector(
                        field_path="metadata.namespace"))))
        return envs

    def get_pserver_container_ports(self):
        ports = []
        port = self.port
        for i in xrange(self.ports_num + self.ports_num_for_sparse):
            ports.append(
                client.V1ContainerPort(
                    container_port=port, name="jobport-%d" % i))
            port += 1
        return ports

    def get_pserver_labels(self):
        return {"paddle-job": self.get_pserver_job_name()}

    def get_pserver_entrypoint(self):
        return ["paddle_k8s", "start_pserver"]

    def get_trainer_entrypoint(sefl):
        return ["paddle_k8s", "start_trainer"]

    def get_trainer_labels(self):
        return {"paddle-job": self.get_trainer_job_name()}

    def get_runtime_docker_image_name(self):
        #TODO: use runtime docker image
        return self.base_iamge
        #return "%s-%s:latest" % (self.namespace, self.job_name)

    def new_pserver_job(self):
        return client.V1beta1StatefulSet(
            metadata=client.V1ObjectMeta(name=self.get_pserver_job_name()),
            spec=client.V1beta1StatefulSetSpec(
                service_name=self.get_pserver_job_name(),
                replicas=self.pservers,
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels=self.get_pserver_labels()),
                    spec=client.V1PodSpec(containers=[
                        client.V1Container(
                            name=self.get_pserver_job_name(),
                            image=self.get_runtime_docker_image_name(),
                            ports=self.get_pserver_container_ports(),
                            env=self.get_env(),
                            command=self.get_pserver_entrypoint())
                    ]))))

    def new_trainer_job(self):
        return client.V1Job(
            metadata=client.V1ObjectMeta(name=self.get_trainer_job_name()),
            spec=client.V1JobSpec(
                parallelism=self.trainers,
                completions=self.trainers,
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels=self.get_trainer_labels()),
                    spec=client.V1PodSpec(
                        volumes=[
                            client.V1Volume(
                                name="glusterfsvol",
                                glusterfs=client.V1GlusterfsVolumeSource(
                                    endpoints=DEFAULT_GLUSTERFS_ENDPOINT,
                                    path=self.glusterfs_volume))
                        ],
                        containers=[
                            client.V1Container(
                                name="trainer",
                                image=self.get_runtime_docker_image_name(),
                                image_pull_policy="Always",
                                command=self.get_trainer_entrypoint(),
                                env=self.get_env(),
                                volume_mounts=[
                                    client.V1VolumeMount(
                                        mount_path=GLUSTERFS_MOUNT_PATH,
                                        name="glusterfsvol")
                                ])
                        ],
                        restart_policy="Never"))))
