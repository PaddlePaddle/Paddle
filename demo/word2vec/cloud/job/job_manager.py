import os
from paddle_job import PaddleJob
from kubernetes import client, config
from kubernetes.client.rest import ApiException

__all__ = ["JobManager"]

if os.getenv("KUBERNETES_SERVICE_HOST", None):
    config.load_incluster_config()
else:
    config.load_kube_config()

NAMESPACE = os.getenv("NAMESPACE", "yanxu")


class JobManager(object):
    def __init__(self, paddle_job):
        self.paddle_job = paddle_job

    def submit(self):
        #submit parameter server statefulset
        try:
            ret = client.AppsV1beta1Api().create_namespaced_stateful_set(
                namespace=NAMESPACE,
                body=self.paddle_job.new_pserver_job(),
                pretty=True)
        except ApiException, e:
            print "Exception when submit pserver job: %s " % e
            return False

        #submit trainer job
        try:
            ret = client.BatchV1Api().create_namespaced_job(
                namespace=NAMESPACE,
                body=self.paddle_job.new_trainer_job(),
                pretty=True)
        except ApiException, e:
            print "Exception when submit trainer job: %s" % e
            return False
        return True


if __name__ == "__main__":
    paddle_job = PaddleJob(
        trainers=3,
        pservers=3,
        base_image="yancey1989/paddle-cloud",
        glusterfs_volume="gfs_vol",
        input="/yanxu05",
        output="/yanxu05",
        job_name="paddle-cloud",
        namespace="yanxu",
        use_gpu=False,
        port=7164,
        ports_num=1,
        ports_num_for_sparse=1,
        num_gradient_servers=1,
        trainer_package_path="/yanxu05/word2vec",
        entry_point="python api_train_v2.py")
    jm = JobManager(paddle_job)
    jm.submit()
