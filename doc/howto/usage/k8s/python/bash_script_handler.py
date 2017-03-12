#-*-coding:utf-8-*-
from kubernetes import client
from pprint import pprint
import os
__all__=["BashScriptHandler"]
class BashScriptHandler(object):
    def __init__(self, namespace, name):
        self._name = name
        self._namespace = namespace
        self._config_map = client.V1ConfigMap()

        metadata = client.V1ObjectMeta()
        metadata.namespace = namespace
        metadata.name = name
        self._config_map.metadata = metadata

    def init_with_bash_filename(self, filename):
        """
        Init ConfigMap struct with filename
        """
        with open(filename, "r") as f:
            data = f.read()
            self._config_map.data = {
                os.path.basename(filename):
                data
            }

    def _upload_file(self):
        api_instance = client.CoreV1Api()
        ret = api_instance.patch_namespaced_config_map(self._name, self._namespace, self._config_map)
        pprint(ret)

    def run(self):
        self._upload_file()


    @property
    def config_map(self):
        return self._config_map

    @property
    def name(self):
        return self._name
if __name__ == "__main__":
    handler = BashScriptHandler("yancey", "yanxu-prepate-data")
    handler.init_with_filename("./get_data.sh")
    print handler.config_map
