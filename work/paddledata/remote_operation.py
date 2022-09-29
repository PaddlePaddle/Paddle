import os
import wget
import paramiko


"""
# 目标路由
url = 'https://bj.bcebos.com/ai-studio-online/74a0254c3ea5447cae952b40c8c7582d7d2823b41c3f4ac7aa3a6053d8ca71bf?authorization=bce-auth-v1%2F5cfe9a5e1454405eb2a975c43eace6ec%2F2022-09-04T15%3A25%3A41Z%2F-1%2F%2F944020cb68fdb471af746d5bce395da5fec4f7c58e5e80ad24486d2c521d2af3&responseContentDisposition=attachment%3B%20filename%3Dvalid_gt.zip'
# 保存的路径
save_path = 'data'
# 下载
wget.download(url, path)
"""


"""
# 对paramiko方法下的SSHclient进行实例化
ssh = paramiko.SSHClient()
# 保存服务器密钥
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# 输入服务器地址，账户名，密码
ssh.connect(hostname='xxx.xx.xxx.xx', port=22, username='root', password='xxxxxx')
# 创建sftp客户端
sftp = paramiko.SFTPClient.from_transport(ssh.get_transport())
# 本地路径
local_path = "local_path.txt"
# 远程路径
remote_path = "/home/remote_path.txt"
sftp.put(local_path, remote_path)
"""


"""
# 对paramiko方法下的SSHclient进行实例化
ssh = paramiko.SSHClient()
# 保存服务器密钥
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# 输入服务器地址，账户名，密码
ssh.connect(hostname='xxx.xx.xxx.xx', port=22, username='root', password='xxxxxx')
# 创建sftp客户端
sftp = paramiko.SFTPClient.from_transport(ssh.get_transport())
# 远程路径
remote_path = "/home/remote_path"
# 本地路径
local_path = "E:/web/local_path.txt"
# 下载文件
sftp.get(remote_path, local_path)
"""


class Downloader(object):
    def __init__(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_path = save_path
        self.cur_path = os.path.abspath(self.save_path)
    
    def __iter__(self):
        for f in self.files:
            yield f
    
    def __call__(self, dataset_source):
        self.dataset_source = dataset_source
        for url in dataset_source:
            wget.download(url, self.save_path)
        files = os.listdir(self.save_path)
        self.files = [os.path.join(self.cur_path, f) for f in files]
        return self

