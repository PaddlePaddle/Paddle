<p align="center">
<img align="center" src="doc/imgs/logo.png", width=1600>
<p>

--------------------------------------------------------------------------------

[Official README.md](./README_official.md)

[![Build Status](https://jenkins-aidev.mthreads.com/buildStatus/icon?job=paddle_musa%2Fmain)](https://jenkins-aidev.mthreads.com/job/paddle_musa/job/develop/)

--------------------------------------------------------------------------------

<!-- toc -->

- [Installation](#installation)
  - [From Python Package](#from-python-package)
  - [From Source](#from-source)
  - [Prerequisites](#prerequisites)
  - [Install Dependencies](#install-dependencies)
  - [Set Important Environment Variables](#set-important-environment-variables)
  - [Building With Script](#building-with-script-recommended)
  - [Docker Image](#docker-image)
    - [Docker Image for Developer](#docker-image-for-developer)
    - [Docker Image for User](#docker-image-for-user)
- [Useful Environment Variables](#useful-environment-variables)
- [Getting Started](#getting-started)
  - [Demo](#demo)

<!-- tocstop -->

## Installation

### From Python Package
- Not ready now.

### From Source

#### Prerequisites
- [MUSA ToolKit](https://github.mthreads.com/mthreads/musa_toolkit)
- [MUDNN](https://github.mthreads.com/mthreads/muDNN)
- Other Libs (including muThrust, muSparse, muAlg, muRand)
- [Docker Container Toolkits](https://mcconline.mthreads.com/software)

**NOTE:** Since some of the dependent libraries are in beta and have not yet been officially released, we recommend using the [development docker](#docker-image-for-developer) provided below to compile **paddle_musa**. If you really want to compile **paddle_musa** in your own environment, then please contact us for additional dependencies.

#### Install Dependencies

```bash
apt-get install ccache
pip install -r requirements.txt
```

#### Set Important Environment Variables
```bash
export MUSA_HOME=path/to/musa_libraries(including mudnn and musa_toolkits) # defalut value is /usr/local/musa/
export LD_LIBRARY_PATH=$MUSA_HOME/lib:$LD_LIBRARY_PATH
export WITH_MUSA=ON
```

#### Building With Script (Recommended)
```bash
python setup.py install
```

### Docker Image

**NOTE:** If you want to use **paddle_musa** in docker container, please install [mt-container-toolkit](https://mcconline.mthreads.com/software/1?id=1) first and use '--env MTHREADS_VISIBLE_DEVICES=all' when starting a container.

#### Docker Image for Developer
```bash
docker run -it --privileged --name=paddle_musa_dev --env MTHREADS_VISIBLE_DEVICES=all --network=host --shm-size=80g sh-harbor.mthreads.com/mt-ai/musa-paddle-dev:latest /bin/bash
```
<details>
<summary>Docker Image List</summary>

| Docker Tag | Description |
| ---- | --- |
| [**v0.1.4/latest**](https://sh-harbor.mthreads.com/harbor/projects/20/repositories/musa-paddle-dev/artifacts-tab) | musatoolkits-v1.4.2 (driver2.2.0 develop or newer)<br> mcc-20230823-daily <br> mudnn 20230823-daily <br> mccl_20230823-daily <br> muAlg_dev-20230823-daily <br> muRAND_dev1.0.0 <br> muSPARSE_dev0.1.0 <br> muThrust_dev-0.1.1 |
| [**v0.1.3**](https://sh-harbor.mthreads.com/harbor/projects/20/repositories/musa-paddle-dev/artifacts-tab) | musatoolkits-v1.4.0 (ddk_1.4.0 develop or newer)<br> mcc-20230814-daily <br> mudnn v1.4.0 <br> mccl_rc1.1.0 <br> muAlg_dev-20230814-daily <br> muRAND_dev1.0.0 <br> muSPARSE_dev0.1.0 <br> muThrust_dev-0.1.1 |
| [**v0.1.2**](https://sh-harbor.mthreads.com/harbor/projects/20/repositories/musa-paddle-dev/artifacts-tab) | musatoolkits-v1.4.0 (ddk_1.4.0 develop or newer)<br> mcc-20230814-daily <br> mudnn v1.4.0 <br> mccl_rc1.1.0 <br> muAlg_dev-20230814-daily <br> muRAND_dev1.0.0 <br> muSPARSE_dev0.1.0 <br> muThrust_dev-0.1.1 |
| [**v0.1.1**](https://sh-harbor.mthreads.com/harbor/projects/20/repositories/musa-paddle-dev/artifacts-tab) | musatoolkits-v1.4.0 (ddk_1.4.0 develop or newer)<br> mudnn v1.4.0 <br> mccl_rc1.1.0 <br> muAlg_dev-0.1.1 <br> muRAND_dev1.0.0 <br> muSPARSE_dev0.1.0 <br> muThrust_dev-0.1.1 |

</details>

#### Docker Image for User
- Not ready now.

## Useful Environment Variables

| Docker Tag | Module | Description |
| ---- | --- | --- |
| export GLOG_v=0~9 | Paddle | Control the log level |
| export MUSA_VISIBLE_DEVICES=0,1,2,3 | Driver | Control the visible devices |
| export MUSA_LAUNCH_BLOCKING=1 | Driver | Set the synchronization mode |

## Getting Started
### Demo

<details>
<summary>code</summary>

```python
import paddle
cpu_tensor1 = paddle.to_tensor([2.0, 3.0, 4.0], place=paddle.CPUPlace())
cpu_tensor2 = paddle.to_tensor([2.0, 3.0, 4.0], place=paddle.CPUPlace())
cpu_result = cpu_tensor1 + cpu_tensor2
print("cpu_result: ", cpu_result)

#paddle.device.set_device("gpu")
gpu_tensor1 = paddle.to_tensor([6, 2], dtype="int32", place=paddle.CUDAPlace(0))
gpu_tensor2 = paddle.to_tensor([5, 3], dtype="int32", place=paddle.CUDAPlace(0))
print(gpu_tensor1.place)
gpu_result = gpu_tensor1 + gpu_tensor2
print("gpu_result: ", gpu_result)
```
</details>
