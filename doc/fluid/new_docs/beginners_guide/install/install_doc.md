# **安装说明**
本说明将指导您在*64位台式机或笔记本电脑*上编译和安装PaddlePaddle，目前PaddlePaddle支持以下环境：

* *Ubuntu 14.04 /16.04 /18.04*
* *CentOS 7 / 6*
* *MacOS 10.12 / 10.13*
* *Windows7 / 8/ 10(专业版/企业版)*

请确保您的环境满足以上条件       
如在安装或编译过程中遇到问题请参见[FAQ](#FAQ)          


## **安装PaddlePaddle**

* Ubuntu下安装PaddlePaddle
* CentOS下安装PaddlePaddle
* MacOS下安装PaddlePaddle
* Windows下安装PaddlePaddle

***
### **Ubuntu下安装PaddlePaddle**

本说明将介绍如何在*64位台式机或笔记本电脑*以及Ubuntu系统下安装PaddlePaddle，我们支持的Ubuntu系统需满足以下要求：

请注意：在其他系统上的尝试可能会导致安装失败。

* *Ubuntu 14.04 /16.04 /18.04*

#### 确定要安装的PaddlePaddle版本

* 仅支持CPU的PaddlePaddle。如果您的计算机没有 NVIDIA® GPU，则只能安装此版本。如果您的计算机有GPU，
也推荐您先安装CPU版本的PaddlePaddle，来检测您本地的环境是否适合。

* 支持GPU的PaddlePaddle。为了使PaddlePaddle程序运行更加迅速，我们通过GPU对PaddlePaddle程序进行加速，但安装GPU版本的PaddlePaddle需要先拥有满足以下条件的NVIDIA® GPU（具体安装流程和配置请务必参见NVIDIA官方文档：[For CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[For cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)）
	* *CUDA 工具包9.0配合cuDNN v7*
	* *CUDA 工具包8.0配合cuDNN v7*
	* *GPU运算能力超过1.0的硬件设备*



#### 选择如何安装PaddlePaddle
在Ubuntu的系统下我们提供4种不同的安装方式：

* Docker安装
* pip安装
* 源码编译安装
* Docker源码编译安装


我们更加推荐**使用Docker进行安装**，因为我们在把工具和配置都安装在一个 Docker image 里，这样如果遇到问题，其他人可以复现问题以便帮助。另外，对于习惯使用Windows和MacOS的开发者来说，使用Docker就不用配置交叉编译环境了。需要强调的是：Docker 不会虚拟任何硬件，Docker container 里运行的编译工具实际上都是在本机的 CPU 和操作系统上直接运行的，性能和把编译工具安装在本机运行一样。        



**使用pip安装**，我们为您提供pip安装方法，但它更依赖您的本机环境，可能会出现和您本机环境相关的一些问题。         



从[**源码编译安装**](#ubt_source)以及[**使用Docker进行源码编译安装**](#ubt_docker)，这是一种通过将PaddlePaddle源代码编译成为二进制文件，然后在安装这个二进制文件的过程，相比使用我们为您编译过的已经通过测试的二进制文件形式的PaddlePaddle，手动编译更为复杂，我们将在说明的最后详细为您解答。
<br/><br/>
##### ***使用Docker进行安装***

<!-- TODO: uncomment it when the offical website can split it to different pages我们更加推荐**使用Docker进行安装**，因为我们在把工具和配置都安装在一个 Docker image 里，这样如果遇到问题，其他人可以复现问题以便帮助。另外，对于习惯使用Windows和MacOS的开发者来说，使用Docker就不用配置交叉编译环境了。需要强调的是：Docker 不会虚拟任何硬件，Docker container 里运行的编译工具实际上都是在本机的 CPU 和操作系统上直接运行的，性能和把编译工具安装在本机运行一样。-->

为了更好的使用Docker并避免发生问题，我们推荐使用**最高版本的Docker**，关于**安装和使用Docker**的细节请参阅Docker[官方文档](https://docs.docker.com/install/)。



> 请注意，要安装和使用支持 GPU 的PaddlePaddle版本，您必须先安装[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)



如果已经**正确安装Docker**，即可以开始**使用Docker安装PaddlePaddle**

1. 使用以下指令拉取我们为您预安装好PaddlePaddle的镜像：


	* 对于需要**CPU版本的PaddlePaddle**的用户请使用以下指令拉取我们为您预安装好*PaddlePaddle For CPU*的镜像：

		`docker pull hub.baidubce.com/paddlepaddle/paddle:0.14.0`
		

	* 对于需要**GPU版本的PaddlePaddle**的用户请使用以下指令拉取我们为您预安装好*PaddlePaddle For GPU*的镜像：

		`docker pull hub.baidubce.com/paddlepaddle/paddle:0.14.0-gpu-cuda9.0-cudnn7`
		

	* 您也可以通过以下指令拉取任意的我们提供的Docker镜像：

		`docker pull hub.baidubce.com/paddlepaddle/paddle:[tag]`
		> （请把[tag]替换为[镜像表](#dockers)中的内容）
		
2. 使用以下指令用已经拉取的镜像构建并进入Docker容器：

	`docker run --name [Name of container] -it -v $PWD:/paddle <imagename> /bin/bash`

	> 上述命令中，--name [Name of container] 设定Docker的名称；-it 参数说明容器已和本机交互式运行； -v $PWD:/paddle 指定将当前路径（Linux中$PWD变量会展开为当前路径的绝对路径）挂载到容器内部的 /paddle 目录； `<imagename>` 指定需要使用的image名称，如果您需要使用我们的镜像请使用`hub.baidubce.com/paddlepaddle/paddle:[tag]` 注：tag的意义同第二步；/bin/bash是在Docker中要执行的命令。

3. （可选：当您需要第二次进入Docker容器中）使用如下命令使用PaddlePaddle：

	`docker start [Name of container]`
	> 启动之前创建的容器。

	`docker attach [Name of container]`
	> 进入启动的容器。

4. （可选：当您镜像中的numpy版本不匹配）在Docker中 使用如下命令安装numpy 1.14.0：
	
	`pip install numpy==1.14.0`
	
至此您已经成功使用Docker安装PaddlePaddle，您只需要进入Docker容器后运行PaddlePaddle即可，更多Docker使用请参见[Docker官方文档](https://docs.docker.com)。

> 注：PaddlePaddle Docker镜像为了减小体积，默认没有安装`vim`，您可以在容器中执行 `apt-get install -y vim` 安装后，在容器中编辑代码。


<br/><br/>
##### ***使用pip安装PaddlePaddle***

您可以直接粘贴以下命令到命令行来安装PaddlePaddle(适用于ubuntu16.04及以上安装CPU-ONLY的版本)，如果出现问题，您可以参照后面的解释对命令作出适应您系统的更改：
		
	apt update && apt install python-dev python-pip && pip install numpy==1.14.0 paddlepaddle && export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

首先，我们使用以下指令来**检测本机的环境**是否适合安装PaddlePaddle：

`uname -m && cat /etc/*release`
> 上面的命令将会显示本机的操作系统和位数信息，请确保您的计算机和本教程的要求一致。


其次，您的电脑需要满足以下要求：

*	Python2.7.x (dev)
*	Pip >= 9.0.1      
	
	> 您的Ubuntu上可能已经安装pip请使用pip -V来确认我们建议使用pip 9.0.1或更高版本来安装

	更新apt的源：   `apt update`

	使用以下命令安装或升级Python和pip到需要的版本： `sudo apt install python-dev python-pip`
	> 即使您的环境中已经有Python2.7也需要安装Python dev。

现在，让我们来安装PaddlePaddle：

1. 使用pip install来安装PaddlePaddle

	* 对于需要**CPU版本PaddlePaddle**的用户：`pip install paddlepaddle`
	

	* 对于需要**GPU版本PaddlePaddle**的用户：`pip install paddlepaddle-gpu`
	> 1. 为防止出现nccl.h找不到的问题请首先按照以下命令安装nccl2（这里提供的是ubuntu 16.04，CUDA8，cuDNN v7下nccl2的安装指令），更多版本的安装信息请参考NVIDIA[官方网站](https://developer.nvidia.com/nccl/nccl-download):      
		a. `wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb`             
		b. `sudo apt-get install libnccl2=2.2.13-1+cuda8.0 libnccl-dev=2.2.13-1+cuda8.0`
	> 2. 如果您不规定pypi包版本号，我们默认为您提供支持Cuda 8/cuDNN v7的PaddlePaddle版本。


	对于出现`Cannot uninstall 'six'.`问题的用户，可是由于您的系统中已有的Python安装问题造成的，请使用`pip install paddlepaddle --ignore-installed six`（CPU）或`pip 	install paddlepaddle --ignore-installed six`（GPU）解决。      
	
	* 对于有**其他要求**的用户：`pip install paddlepaddle==[版本号]`
	> `版本号`参见[安装包列表](#whls)或者您如果需要获取并安装**最新的PaddlePaddle开发分支**，可以从[多版本whl包列表](#ciwhls)或者我们的[CI系统](https://paddleci.ngrok.io/project.html?projectId=Manylinux1&tab=projectOverview) 中下载最新的whl安装包和c-api开发包并安装。如需登录，请点击“Log in as guest”。
	
2. 使用以下指令将默认装在`/usr/local/lib`下的`libmkldnn`放在`LD_LIBRARY_PATH中`:

	`export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH`
	> 如果您的`libmkldnn`没有装在`/usr/local/lib`下，请使用`find / -name libmkldnn.so.0`从根目录开始找到`libmkldnn.so.0`之后将路径填到以下命令[dir]的的位置：`export LD_LIBRARY_PATH=[dir]:$LD_LIBRARY_PATH`。

3. 使用以下指令将numpy的版本降至1.12.0 - 1.14.0之间：
	
	> 由于numpy支持造成numpy 1.15.0 及以上版本引发`shape warning`。

	`pip install -U numpy==1.14.0`
	> 如果遇到`Python.h: No such file or directory`请设置`python.h`路径到`C_INCLUDE_PATH/CPLUS_INCLUDE_PATH`
	

现在您已经完成使用`pip install` 来安装的PaddlePaddle的过程。

<br/><br/>
##### ***验证安装***
安装完成后您可以使用：`python` 进入python解释器，然后使用`import paddle.fluid` 验证是否安装成功。

<br/><br/>
##### ***如何卸载PaddlePaddle***
请使用以下命令卸载PaddlePaddle：

* ***CPU版本的PaddlePaddle***: `pip uninstall paddlepaddle`

* ***GPU版本的PaddlePaddle***: `pip uninstall paddlepaddle-gpu`

<br/><br/>
### **CentOS下安装PaddlePaddle**

本说明将介绍如何在*64位台式机或笔记本电脑*以及CentOS系统下安装PaddlePaddle，我们支持的CentOS系统需满足以下要求：      


请注意：在其他系统上的尝试可能会导致安装失败。

* *CentOS 6 / 7*

#### 确定要安装的PaddlePaddle版本
* 仅支持CPU的PaddlePaddle。如果您的计算机没有 NVIDIA® GPU，则只能安装此版本。如果您的计算机有GPU，
推荐您先安装CPU版本的PaddlePaddle，来检测您本地的环境是否适合。

* 支持GPU的PaddlePaddle，为了使PaddlePaddle程序运行的更加迅速，我们通过GPU对PaddlePaddle程序进行加速，但安装GPU版本的PaddlePaddle需要先拥有满足以下条件的NVIDIA® GPU（具体安装流程和配置请务必参见NVIDIA官方文档：[For CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[For cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)）
	* *CUDA 工具包9.0配合cuDNN v7*
	* *CUDA 工具包8.0配合cuDNN v7*
	* *GPU运算能力超过1.0的硬件设备*



#### 选择如何安装PaddlePaddle
在CentOS的系统下我们提供4种不同的安装方式：

* Docker安装（不支持GPU版本）
* pip安装
* 源码编译安装（不支持CentOS 6的所有版本以及CentOS 7的GPU版本）
* Docker源码编译安装（不支持GPU版本）


我们更加推荐**使用Docker进行安装**，因为我们在把工具和配置都安装在一个 Docker image 里，这样如果遇到问题，其他人可以复现问题以便帮助。另外，对于习惯使用Windows和MacOS的开发者来说，使用Docker就不用配置交叉编译环境了。需要强调的是：Docker 不会虚拟任何硬件，Docker container 里运行的编译工具实际上都是在本机的 CPU 和操作系统上直接运行的，性能和把编译工具安装在本机运行一样。        




**使用pip安装**，我们为您提供pip安装方法，但它更依赖您的本机环境，可能会出现和您本机环境相关的一些问题。

从[**源码编译安装**](#ct_source)以及[**使用Docker进行源码编译安装**](#ct_docker)，这是一种通过将PaddlePaddle源代码编译成为二进制文件，然后在安装这个二进制文件的过程，相比使用我们为您编译过的已经通过测试的二进制文件形式的PaddlePaddle，手动编译更为复杂，我们将在说明的最后详细为您解答。
<br/><br/>
##### ***使用Docker进行安装***

<!-- 我们更加推荐**使用Docker进行安装**，因为我们在把工具和配置都安装在一个 Docker image 里，这样如果遇到问题，其他人可以复现问题以便帮助。另外，对于习惯使用Windows和MacOS的开发者来说，使用Docker就不用配置交叉编译环境了。需要强调的是：Docker 不会虚拟任何硬件，Docker container 里运行的编译工具实际上都是在本机的 CPU 和操作系统上直接运行的，性能和把编译工具安装在本机运行一样。-->

为了更好的使用Docker并避免发生问题，我们推荐使用**最高版本的Docker**，关于**安装和使用Docker**的细节请参阅Docker[官方文档](https://docs.docker.com/install/)


> 请注意，要安装和使用支持 GPU 的PaddlePaddle版本，您必须先安装[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)



当您已经**正确安装Docker**后你就可以开始**使用Docker安装PaddlePaddle**

1. 使用以下指令拉取我们为您预安装好PaddlePaddle的镜像：


	* 对于需要**CPU版本的PaddlePaddle**的用户请使用以下指令拉取我们为您预安装好*PaddlePaddle For CPU*的镜像：

		`docker pull hub.baidubce.com/paddlepaddle/paddle:0.14.0`       
		
		


	* 您也可以通过以下指令拉取任意的我们提供的Docker镜像：

		`docker pull hub.baidubce.com/paddlepaddle/paddle:[tag]`
		> （请把[tag]替换为[镜像表](#dockers)中的内容）
2. 使用以下指令用已经拉取的镜像构建并进入Docker容器：

	`docker run --name [Name of container] -it -v $PWD:/paddle <imagename> /bin/bash`
	
	> 上述命令中，--name [Name of container] 设定Docker的名称；-it 参数说明容器已和本机交互式运行； -v $PWD:/paddle 指定将当前路径（Linux中$PWD变量会展开为当前路径的[绝对路径](https://baike.baidu.com/item/绝对路径/481185)）挂载到容器内部的 /paddle 目录； `<imagename>` 指定需要使用的image名称，如果您需要使用我们的镜像请使用`hub.baidubce.com/paddlepaddle/paddle:[tag]` 注：tag的意义同第二步，/bin/bash是在Docker中要执行的命令。  

3. （可选：当您需要第二次进入Docker容器中）使用如下命令使用PaddlePaddle：

	`docker start [Name of container]`
	> 启动之前创建的容器。

	`docker attach [Name of container]`
	> 进入启动的容器。

4. （可选：当您镜像中的numpy版本不匹配）在Docker中 使用如下命令安装numpy 1.14.0：
	
	`pip install numpy==1.14.0`
	
至此您已经成功使用Docker安装PaddlePaddle，您只需要进入Docker容器后运行PaddlePaddle即可，更多Docker使用请参见[Docker官方文档](https://docs.docker.com)。
> 注：PaddlePaddle Docker镜像为了减小体积，默认没有安装`vim`，您可以在容器中执行 `apt-get install -y vim` 安装后，在容器中编辑代码。


<br/><br/>
##### ***使用pip安装PaddlePaddle***

您可以直接粘贴以下命令到命令行来安装PaddlePaddle(适用于CentOS7安装CPU-ONLY的版本)，如果出现问题，您可以参照后面的解释对命令作出适应您系统的更改：
		
	yum update && yum install -y epel-release gcc && yum install -y python-devel python-pip && pip install numpy==1.14.0 paddlepaddle && export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH

首先，我们使用以下指令来**检测本机的环境**是否适合安装PaddlePaddle：

`uname -m && cat /etc/*release`
> 上面的命令将会显示本机的操作系统和位数信息，请确保您的计算机和本教程的要求一致。


其次，您的计算机需要满足以下要求：

*	Python2.7.x (devel)   
	
	> CentOS6需要编译Python2.7成[共享库](#FAQ)。         


*	Pip >= 9.0.1       
	
	> 您的CentOS上可能已经安装pip请使用pip -V来确认我们建议使用pip 9.0.1或更高版本来安装。

	更新yum的源：   `yum update` 并安装拓展源以安装pip：   `yum install -y epel-release`

	使用以下命令安装或升级Python和pip到需要的版本： `sudo yum install python-devel python-pip`
	> 即使您的环境中已经有`Python2.7`也需要安装`python devel`。

下面将说明如何安装PaddlePaddle：

1. 使用pip install来安装PaddlePaddle：
	
	* 对于需要**CPU版本PaddlePaddle**的用户：`pip install paddlepaddle`


	* 对于需要**GPU版本PaddlePaddle**的用户: `pip install paddlepaddle-gpu`
	> 1. 为防止出现nccl.h找不到的问题请首先按照NVIDIA[官方网站](https://developer.nvidia.com/nccl/nccl-download)的指示正确安装nccl2
	> 2. 如果您不规定pypi包版本号，我们默认为您提供支持Cuda 8/cuDNN v7的PaddlePaddle版本。 

	对于出现`Cannot uninstall 'six'.`问题的用户，可是由于您的系统中已有的Python安装问题造	成的，请使用`pip install paddlepaddle --ignore-installed six`（CPU）或`pip 	install paddlepaddle-gpu --ignore-installed six`（GPU）解决。
	
	* 对于有**其他要求**的用户：`pip install paddlepaddle==[版本号]`
	> `版本号`参见[安装包列表](#whls)或者您如果需要获取并安装**最新的PaddlePaddle开发分支**，可以从[多版本whl包列表](#ciwhls)或者我们的[CI系统](https://paddleci.ngrok.io/project.html?projectId=Manylinux1&tab=projectOverview) 中下载最新的whl安装包和c-api开发包并安装。如需登录，请点击“Log in as guest”。
	
2. 使用以下指令将默认装在`/usr/lib`下的`libmkldnn`放在`LD_LIBRARY_PATH中`:

	`export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH`
	> 如果您的`libmkldnn`没有装在`/usr/lib`下，请使用`find / -name libmkldnn.so.0`从根目录开始找到`libmkldnn.so.0`之后将路径填到以下命令[dir]的的位置：`export LD_LIBRARY_PATH=[dir]:$LD_LIBRARY_PATH`。

3. 使用以下指令将numpy的版本降至1.12.0-1.14.0之间：
	
	> 由于numpy支持造成numpy 1.15.0 及以上版本引发`shape warning`。

	`pip install -U numpy==1.14.0`
	> 如果遇到`Python.h: No such file or directory`请设置`python.h`路径到`C_INCLUDE_PATH/CPLUS_INCLUDE_PATH`
	    
	    

现在您已经完成通过`pip install` 来安装的PaddlePaddle的过程。


<br/><br/>
##### ***验证安装***
安装完成后您可以使用：`python` 进入Python解释器，然后使用`import paddle.fluid` 验证是否安装成功。

<br/><br/>
##### ***如何卸载PaddlePaddle***
请使用以下命令卸载PaddlePaddle：

* ***CPU版本的PaddlePaddle***: `pip uninstall paddlepaddle`

* ***GPU版本的PaddlePaddle***: `pip uninstall paddlepaddle-gpu`




<br/><br/>
### **MacOS下安装PaddlePaddle**

本说明将介绍如何在*64位台式机或笔记本电脑*以及MacOS系统下安装PaddlePaddle，我们支持的MacOS系统需满足以下要求。

请注意：在其他系统上的尝试可能会导致安装失败。

* *MacOS 10.12/10.13*

#### 确定要安装的PaddlePaddle版本

* 仅支持CPU的PaddlePaddle。



#### 选择如何安装PaddlePaddle
在MacOS的系统下我们提供3种不同的安装方式：

* Docker安装（不支持GPU版本）   
* Docker源码编译安装（不支持GPU版本）    


我们更加推荐**使用Docker进行安装**，因为我们在把工具和配置都安装在一个 Docker image 里，这样如果遇到问题，其他人可以复现问题以便帮助。另外，对于习惯使用Windows和MacOS的开发者来说，使用Docker就不用配置交叉编译环境了。需要强调的是：Docker 不会虚拟任何硬件，Docker container 里运行的编译工具实际上都是在本机的 CPU 和操作系统上直接运行的，性能和把编译工具安装在本机运行一样。        





<br/><br/>
##### ***使用Docker进行安装***

<!-- 我们更加推荐**使用Docker进行安装**，因为我们在把工具和配置都安装在一个 Docker image 里，这样如果遇到问题，其他人可以复现问题以便帮助。另外，对于习惯使用Windows和MacOS的开发者来说，使用Docker就不用配置交叉编译环境了。需要强调的是：Docker 不会虚拟任何硬件，Docker container 里运行的编译工具实际上都是在本机的 CPU 和操作系统上直接运行的，性能和把编译工具安装在本机运行一样。-->

为了更好的使用Docker并避免发生问题，我们推荐使用**最高版本的Docker**，关于**安装和使用Docker**的细节请参阅Docker[官方文档](https://docs.docker.com/install/)。
> 请注意，在MacOS系统下登陆docker需要使用您的dockerID进行登录，否则将出现`Authenticate Failed`错误。

如果已经**正确安装Docker**，即可以开始**使用Docker安装PaddlePaddle**

1. 使用以下指令拉取我们为您预安装好PaddlePaddle的镜像：


	* 对于需要**CPU版本的PaddlePaddle**的用户请使用以下指令拉取我们为您预安装好*PaddlePaddle For CPU*的镜像：

		`docker pull hub.baidubce.com/paddlepaddle/paddle:0.14.0`
		

	* 您也可以通过以下指令拉取任意的我们提供的Docker镜像：

		`docker pull hub.baidubce.com/paddlepaddle/paddle:[tag]`
		> （请把[tag]替换为[镜像表](#dockers)中的内容）
		
2. 使用以下指令用已经拉取的镜像构建并进入Docker容器：

	`docker run --name [Name of container] -it -v $PWD:/paddle <imagename> /bin/bash`

	> 上述命令中，--name [Name of container] 设定Docker的名称；-it 参数说明容器已和本机交互式运行； -v $PWD:/paddle 指定将当前路径（Linux中$PWD变量会展开为当前路径的[绝对路径](https://baike.baidu.com/item/绝对路径/481185)）挂载到容器内部的 /paddle 目录； `<imagename>` 指定需要使用的image名称，如果您需要使用我们的镜像请使用`hub.baidubce.com/paddlepaddle/paddle:[tag]` 注：tag的意义同第二步；/bin/bash是在Docker中要执行的命令。

3. （可选：当您需要第二次进入Docker容器中）使用如下命令使用PaddlePaddle：

	`docker start [Name of container]`
	> 启动之前创建的容器。

	`docker attach [Name of container]`
	> 进入启动的容器。

4. （可选：当您镜像中的numpy版本不匹配）在Docker中 使用如下命令安装numpy 1.14.0：
	
	`pip install numpy==1.14.0`
	
至此您已经成功使用Docker安装PaddlePaddle，您只需要进入Docker容器后运行PaddlePaddle即可，更多Docker使用请参见[Docker官方文档](https://docs.docker.com)。

> 注：PaddlePaddle Docker镜像为了减小体积，默认没有安装`vim`，您可以在容器中执行 `apt-get install -y vim` 安装后，在容器中编辑代码。
<!--TODO: When we support pip install mode on MacOS, we can write on this part -->



<br/><br/>
##### ***验证安装***
安装完成后您可以使用：`python` 进入python解释器，然后使用`import paddle.fluid` 验证是否安装成功。

<br/><br/>
##### ***如何卸载PaddlePaddle***
请使用以下命令卸载PaddlePaddle：

* ***CPU版本的PaddlePaddle***: `pip uninstall paddlepaddle`     




<br/><br/>
### **Windows下安装PaddlePaddle**

本说明将介绍如何在*64位台式机或笔记本电脑*以及Windows系统下安装PaddlePaddle，我们支持的Windows系统需满足以下要求。

请注意：在其他系统上的尝试可能会导致安装失败。

* *Windows 7/8 and Windows 10 专业版/企业版*

#### 确定要安装的PaddlePaddle版本

* Windows下我们目前仅提供支持CPU的PaddlePaddle。


#### 选择如何安装PaddlePaddle
在Windows系统下请使用我们为您提供的[一键安装包](http://paddle-windows.bj.bcebos.com/PaddlePaddle-windows.zip)进行安装
	
> 我们提供的一键安装包将基于Docker为您进行便捷的安装流程


我们之所以使用**基于Docker的安装方式**，是因为我们在把工具和配置都安装在一个 Docker image 里，这样如果遇到问题，其他人可以复现问题以便帮助。另外，对于习惯使用Windows和MacOS的开发者来说，使用Docker就不用配置交叉编译环境了。需要强调的是：Docker 不会虚拟任何硬件，Docker container 里运行的编译工具实际上都是在本机的 CPU 和操作系统上直接运行的，性能和把编译工具安装在本机运行一样。        



<!--从**源码编译安装**，在Windows下我们不支持**直接源码编译安装**，使用docker进行源码编译的过程将在文档的最后为您展示。-->




<br/><br/>
##### ***验证安装***
安装完成后您可以使用：`python` 进入python解释器，然后使用`import paddle.fluid` 验证是否安装成功。

<br/><br/>
##### ***如何卸载PaddlePaddle***
请使用以下命令卸载PaddlePaddle：

* ***CPU版本的PaddlePaddle***: `pip uninstall paddlepaddle`   






<br/><br/>
## **从源码编译PaddlePaddle**
我们也为您提供了从源码编译的方式，但不推荐您使用这种方式，这是因为您的本机环境多种多样，在编译源码时易出现复杂的本说明中覆盖以外问题而造成安装失败。
      
***       
### **Ubuntu下从源码编译PaddlePaddle**

本说明将介绍如何在*64位台式机或笔记本电脑*以及Ubuntu系统下编译PaddlePaddle，我们支持的Ubuntu系统需满足以下要求：

* Ubuntu 14.04/16.04/18.04（这涉及到相关工具是否能被正常安装）

#### 确定要编译的PaddlePaddle版本
* **仅支持CPU的PaddlePaddle**，如果您的系统没有 NVIDIA® GPU，则必须安装此版本。而此版本较GPU版本更加容易安
因此即使您的计算机上拥有GPU我们也推荐您先安装CPU版本的PaddlePaddle来检测您本地的环境是否适合。

* **支持GPU的PaddlePaddle**，为了使得PaddlePaddle程序运行的更加迅速，我们通常使用GPU对PaddlePaddle程序进行加速，但安装GPU版本的PaddlePaddle需要先拥有满足以下条件的NVIDIA® GPU（具体安装流程和配置请务必参见NVIDIA官方文档：[For CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[For cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)）
	* *CUDA 工具包9.0配合cuDNN v7*
	* *CUDA 工具包8.0配合cuDNN v7*
	* *GPU运算能力超过1.0的硬件设备*

#### 选择如何编译PaddlePaddle
在Ubuntu的系统下我们提供两种不同的编译方式：

* Docker源码编译
* 直接本机源码编译

我们更加推荐**使用Docker进行编译**，因为我们在把工具和配置都安装在一个 Docker image 里。这样如果遇到问题，其他人可以复现问题以便帮助。另外，对于习惯使用Windows和MacOS的开发者来说，使用Docker就不用配置交叉编译环境了。有人用虚拟机来类比 Docker。需要强调的是：Docker 不会虚拟任何硬件，Docker container 里运行的编译工具实际上都是在本机的 CPU 和操作系统上直接运行的，性能和把编译工具安装在本机运行一样。        



我们也提供了可以从**本机直接源码编译**的方法，但是由于在本机上的情况更加复杂，我们只对特定系统提供了支持。        

<a name="ubt_docker"></a>                         

<br/><br/>
##### ***使用Docker进行编译***
为了更好的使用Docker并避免发生问题，我们推荐使用**最高版本的Docker**，关于**安装和使用Docker**的细节请参阅Docker[官方文档](https://docs.docker.com/install/)


> 请注意，要安装和使用支持 GPU 的PaddlePaddle版本，您必须先安装[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)



当您已经**正确安装Docker**后你就可以开始**使用Docker编译PaddlePaddle**：

1. 请首先选择您希望储存PaddlePaddle的路径，然后在该路径下使用以下命令将PaddlePaddle的源码从github克隆到本地当前目录下名为Paddle的文件夹中：

	`git clone https://github.com/PaddlePaddle/Paddle.git`

2. 进入Paddle目录下： `cd Paddle`

3. 利用我们提供的镜像（使用该命令您可以不必提前下载镜像）：

	`docker run --name paddle-test -v $PWD:/paddle --network=host -it hub.baidubce.com/paddlepaddle/paddle:latest-dev /bin/bash`
	> --name paddle-test为您创建的Docker容器命名为paddle-test，-v $PWD:/paddle 将当前目录挂载到Docker容器中的/paddle目录下（Linux中$PWD变量会展开为当前路径的[绝对路径](https://baike.baidu.com/item/绝对路径/481185)），-it 与宿主机保持交互状态，`hub.baidubce.com/paddlepaddle/paddle:latest-dev` 使用名为`hub.baidubce.com/paddlepaddle/paddle:latest-dev`的镜像创建Docker容器，/bin/bash 进入容器后启动/bin/bash命令。

4. 进入Docker后进入paddle目录下：`cd paddle`

5. 切换到较稳定release分支下进行编译：

	`git checkout release/0.14.0`

6. 创建并进入/paddle/build路径下：

	`mkdir -p /paddle/build && cd /paddle/build`

7. 使用以下命令安装相关依赖：

	`pip install numpy==1.14.0`
	> 安装numpy 1.14.0，由于目前numpy1.15.0会引起大量warning，因此在numpy修复该问题前我们先使用numpy 1.14.0。

	`pip install protobuf==3.1.0`
	> 安装protobuf 3.1.0。

	`apt install patchelf`
	> 安装patchelf，PatchELF 是一个小而实用的程序，用于修改ELF可执行文件的动态链接器和RPATH。

8. 执行cmake：       
	
	>具体编译选项含义请参见[编译选项表](#Compile)<!--TODO: Link 编译选项表到这里-->


	*  对于需要编译**CPU版本PaddlePaddle**的用户：

		`cmake .. -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF`


	* 对于需要编译**GPU版本PaddlePaddle**的用户：

		`cmake .. -DWITH_FLUID_ONLY=ON -DWITH_GPU=ON -DWITH_TESTING=OFF`


9. 执行编译：

	`make -j$(nproc)`
	> 使用多核编译

10. 编译成功后进入`/paddle/build/python/dist`目录下找到生成的`.whl`包： `cd /paddle/build/python/dist`

11. 在当前机器或目标机器安装编译好的`.whl`包：

	`pip install （whl包的名字）`

至此您已经成功使用Docker安装PaddlePaddle，您只需要进入Docker容器后运行PaddlePaddle即可，更多Docker使用请参见[Docker官方文档](https://docs.docker.com)。

> 注：PaddlePaddle Docker镜像为了减小体积，默认没有安装`vim`，您可以在容器中执行 `apt-get install -y vim` 安装后，在容器中编辑代码。

恭喜您，现在您已经完成使用Docker编译PaddlePaddle的过程。            

<a name="ubt_source"></a>    
	
<br/><br/>
##### ***本机编译***

1. 检查您的计算机和操作系统是否符合我们支持的编译标准： `uname -m && cat /etc/*release`

2. 更新`apt`的源： `apt update`

2. 我们支持使用virtualenv进行编译安装，首先请使用以下命令创建一个名为`paddle-venv`的虚环境：

	* 安装Python-dev: `apt install python-dev`

	* 安装pip: `apt install python-pip` (请保证拥有9.0.1及以上版本的pip）

	* 安装虚环境`virtualenv`以及`virtualenvwrapper`并创建名为`paddle-venv`的虚环境：

		1.  `apt install virtualenv` 或 `pip install virtualenv`
		2.  `apt install virtualenvwrapper` 或 `pip install virtualenvwrapper`
		3.  找到`virtualenvwrapper.sh`： `find / -name virtualenvwrapper.sh`
		4.  查看`virtualenvwrapper.sh`中的安装方法： `cat virtualenvwrapper.sh`
		5.  按照`virtualenvwrapper.sh`中的安装方法安装`virtualwrapper`
		6.  创建名为`paddle-venv`的虚环境： `mkvirtualenv paddle-venv`


3. 进入虚环境：`workon paddle-venv`         


4. **执行编译前**请您确认在虚环境中安装有[编译依赖表](#third_party)中提到的相关依赖：<!--TODO：Link 安装依赖表到这里-->

	* 这里特别提供`patchELF`的安装方法，其他的依赖可以使用`apt install`或者`pip install` 后跟依赖名称和版本安装:

		`apt install patchelf`
		> 不能使用apt安装的用户请参见patchElF github[官方文档](https://gist.github.com/ruario/80fefd174b3395d34c14)

5. 将PaddlePaddle的源码clone在当下目录下的Paddle的文件夹中，并进入Padde目录下：

	- `git clone https://github.com/PaddlePaddle/Paddle.git`

	- `cd Paddle`

6. 切换到较稳定release分支下进行编译：

	`git checkout release/0.14.0`

7. 并且请创建并进入一个叫build的目录下：

	`mkdir build && cd build`

8. 执行cmake：      
	
	>具体编译选项含义请参见[编译选项表](#Compile)<!--TODO：Link 安装选项表到这里-->


	*  对于需要编译**CPU版本PaddlePaddle**的用户：

		`cmake .. -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF`.


	* 对于需要编译**GPU版本PaddlePaddle**的用户：(*仅支持ubuntu16.04/14.04*)

		1. 请确保您已经正确安装nccl2，或者按照以下指令安装nccl2（这里提供的是ubuntu 16.04，CUDA8，cuDNN7下nccl2的安装指令），更多版本的安装信息请参考NVIDIA[官方网站](https://developer.nvidia.com/nccl/nccl-download):      
			i. `wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb`            
			ii. `sudo apt-get install libnccl2=2.2.13-1+cuda8.0 libnccl-dev=2.2.13-1+cuda8.0` 
		
		2. 如果您已经正确安装了`nccl2`，就可以开始cmake了：

		`cmake .. -DWITH_FLUID_ONLY=ON -DWITH_GPU=ON -DWITH_TESTING=OFF`

9. 使用以下命令来编译：

	`make -j$(nproc)`

10. 编译成功后进入`/paddle/build/python/dist`目录下找到生成的`.whl`包： `cd /paddle/build/python/dist`

11. 在当前机器或目标机器安装编译好的`.whl`包：

	`pip install （whl包的名字）`

恭喜您，现在您已经完成使本机编译PaddlePaddle的过程了。

<br/><br/>
##### ***验证安装***
安装完成后您可以使用：`python` 进入Python解释器，然后使用`import paddle.fluid` 验证是否安装成功。

<br/><br/>
##### ***如何卸载PaddlePaddle***
请使用以下命令卸载PaddlePaddle：

* ***CPU版本的PaddlePaddle***: `pip uninstall paddlepaddle`

* ***GPU版本的PaddlePaddle***: `pip uninstall paddlepaddle-gpu`


<br/><br/>
### **CentOS下从源码编译PaddlePaddle**

本说明将介绍如何在*64位台式机或笔记本电脑*以及CentOS系统下编译PaddlePaddle，我们支持的Ubuntu系统需满足以下要求：

* CentOS 7 / 6（这涉及到相关工具是否能被正常安装）

#### 确定要编译的PaddlePaddle版本
* **仅支持CPU的PaddlePaddle**。

<!--* 支持GPU的PaddlePaddle，为了使得PaddlePaddle程序运行的更加迅速，我们通常使用GPU对PaddlePaddle程序进行加速，但安装GPU版本的PaddlePaddle需要先拥有满足以下条件的NVIDIA® GPU（具体安装流程和配置请务必参见NVIDIA官方文档：[For CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[For cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)）
	* *Cuda 工具包9.0配合cuDNN v7*
	* *Cuda 工具包8.0配合cuDNN v7*
	* *GPU运算能力超过1.0的硬件设备*-->

#### 选择如何编译PaddlePaddle
我们在CentOS的系统下提供2种的编译方式：

* Docker源码编译（不支持CentOS 6 / 7的GPU版本）
* 直接本机源码编译（不支持CentOS 6的全部版本以及CentOS 7的GPU版本）

我们更加推荐**使用Docker进行编译**，因为我们在把工具和配置都安装在一个 Docker image 里。这样如果遇到问题，其他人可以复现问题以便帮助。另外，对于习惯使用Windows和MacOS的开发者来说，使用Docker就不用配置交叉编译环境了。需要强调的是：Docker 不会虚拟任何硬件，Docker container 里运行的编译工具实际上都是在本机的 CPU 和操作系统上直接运行的，性能和把编译工具安装在本机运行一样。        



同样对于那些出于各种原因不能够安装Docker的用户我们也提供了可以从**本机直接源码编译**的方法，但是由于在本机上的情况更加复杂，因此我们只支持特定的系统。            

<a name="ct_docker"></a>


<br/><br/>
##### ***使用Docker进行编译***

为了更好的使用Docker并避免发生问题，我们推荐使用**最高版本的Docker**，关于**安装和使用Docker**的细节请参阅Docker[官方文档](https://docs.docker.com/install/)。   


<!--TODO add the following back when support gpu version on Cent-->

当您已经**正确安装Docker**后你就可以开始**使用Docker编译PaddlePaddle**啦：

1. 请首先选择您希望储存PaddlePaddle的路径，然后在该路径下使用以下命令将PaddlePaddle的源码从github克隆到本地当前目录下名为Paddle的文件夹中：

	`git clone https://github.com/PaddlePaddle/Paddle.git`

2. 进入Paddle目录下： `cd Paddle`

3. 利用我们提供的镜像（使用该命令您可以不必提前下载镜像）：

	`docker run --name paddle-test -v $PWD:/paddle --network=host -it hub.baidubce.com/paddlepaddle/paddle:latest-dev /bin/bash`
	> --name paddle-test为您创建的Docker容器命名为paddle-test，-v $PWD:/paddle 将当前目录挂载到Docker容器中的/paddle目录下（Linux中$PWD变量会展开为当前路径的[绝对路径](https://baike.baidu.com/item/绝对路径/481185)），-it 与宿主机保持交互状态，`hub.baidubce.com/paddlepaddle/paddle` 使用名为`hub.baidubce.com/paddlepaddle/paddle:latest-dev`的镜像创建Docker容器，/bin/bash 进入容器后启动/bin/bash命令。

4. 进入Docker后进入paddle目录下：`cd paddle`

5. 切换到较稳定release分支下进行编译：

	`git checkout release/0.14.0`

6. 创建并进入/paddle/build路径下：

	`mkdir -p /paddle/build && cd /paddle/build`

7. 使用以下命令安装相关依赖：

	`pip install numpy==1.14.0`
	> 安装numpy 1.14.0，由于目前numpy1.15.0会引起大量warning，因此在numpy修复该问题前我们先使用numpy 1.14.0。

	`pip install protobuf==3.1.0`
	> 安装protobuf 3.1.0。

	`apt install patchelf`
	> 安装patchelf，PatchELF 是一个小而实用的程序，用于修改ELF可执行文件的动态链接器和RPATH。

8. 执行cmake：        
	
	>具体编译选项含义请参见[编译选项表](#Compile)<!--TODO： Link 编译选项表到这里-->
	*  对于需要编译**CPU版本PaddlePaddle**的用户：

		`cmake .. -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF`
		
		
	>> 我们目前不支持CentOS下GPU版本PaddlePaddle的编译     
		
9. 执行编译：

	`make -j$(nproc)`
	> 使用多核编译

10. 编译成功后进入`/paddle/build/python/dist`目录下找到生成的`.whl`包： `cd /paddle/build/python/dist`

11. 在当前机器或目标机器安装编译好的`.whl`包：

	`pip install （whl包的名字）`

至此您已经成功使用Docker安装PaddlePaddle，您只需要进入Docker容器后运行PaddlePaddle即可，更多Docker使用请参见[Docker官方文档](https://docs.docker.com)。

> 注：PaddlePaddle Docker镜像为了减小体积，默认没有安装`vim`，您可以在容器中执行 `apt-get install -y vim` 安装后，在容器中编辑代码。

恭喜您，现在您已经完成使用Docker编译PaddlePaddle的过程。        



      

<a name="ct_source"></a>


<br/><br/>
##### ***本机编译***

1. 检查您的计算机和操作系统是否符合我们支持的编译标准： `uname -m && cat /etc/*release`

2. 更新`yum`的源： `yum update`, 并添加必要的yum源：`yum install -y epel-release`

3. 安装必要的工具`bzip2`以及`make`： `yum install -y bzip2` ， `yum install -y make`

2. 我们支持使用virtualenv进行编译安装，首先请使用以下命令创建一个名为`paddle-venv`的虚环境：

	* 安装Python-dev: `yum install python-devel`

	* 安装pip: `yum install python-pip` (请保证拥有9.0.1及以上的pip版本）

	* 安装虚环境`virtualenv`以及`virtualenvwrapper`并创建名为`paddle-venv`的虚环境：

		1.  `pip install virtualenv` 或 `pip install virtualenv`
		2.  `pip install virtualenvwrapper` 或 `pip install virtualenvwrapper`
		3.  找到`virtualenvwrapper.sh`： `find / -name virtualenvwrapper.sh`
		4.  查看`virtualenvwrapper.sh`中的安装方法： `cat vitualenvwrapper.sh`
		5.  安装`virtualwrapper`
		6.  创建名为`paddle-venv`的虚环境： `mkvirtualenv paddle-venv`


3. 进入虚环境：`workon paddle-venv`         


4. **执行编译前**请您确认在虚环境中安装有[编译依赖表](#third_party)中提到的相关依赖：<!--TODO：Link 安装依赖表到这里-->

	* 这里特别提供`patchELF`的安装方法，其他的依赖可以使用`yum install`或者`pip install` 后跟依赖名称和版本安装:

		`yum install patchelf`
		> 不能使用apt安装的用户请参见patchElF github[官方文档](https://gist.github.com/ruario/80fefd174b3395d34c14)

5. 将PaddlePaddle的源码clone在当下目录下的Paddle的文件夹中，并进入Padde目录下：

	- `git clone https://github.com/PaddlePaddle/Paddle.git`

	- `cd Paddle`

6. 切换到较稳定release分支下进行编译：

	`git checkout release/0.14.0`

7. 并且请创建并进入一个叫build的目录下：

	`mkdir build && cd build`

8. 执行cmake：     
	
	>具体编译选项含义请参见[编译选项表](#Compile)<!--TODO：Link 安装选项表到这里-->


	*  对于需要编译**CPU版本PaddlePaddle**的用户：

		`cmake .. -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF`.
	
		<!--Add CentOS7 GPU compile instruction here when we support it-->


9. 使用以下命令来编译：

	`make -j$(nproc)`

10. 编译成功后进入`/paddle/build/python/dist`目录下找到生成的`.whl`包： `cd /paddle/build/python/dist`

11. 在当前机器或目标机器安装编译好的`.whl`包：

	`pip install （whl包的名字）`

恭喜您，现在您已经完成使本机编译PaddlePaddle的过程了。



<br/><br/>
##### ***验证安装***
安装完成后您可以使用：`python` 进入Python解释器，然后使用`import paddle.fluid` 验证是否安装成功。

<br/><br/>
##### ***如何卸载PaddlePaddle***
请使用以下命令卸载PaddlePaddle：

* ***CPU版本的PaddlePaddle***: `pip uninstall paddlepaddle`




<br/><br/>
### **MacOS下从源码编译PaddlePaddle**

本说明将介绍如何在*64位台式机或笔记本电脑*以及MacOS系统下编译PaddlePaddle，我们支持的MacOS系统需满足以下要求：

* MacOS 10.12/10.13（这涉及到相关工具是否能被正常安装）

#### 确定要编译的PaddlePaddle版本
* **仅支持CPU的PaddlePaddle**。

<!--* 支持GPU的PaddlePaddle，为了使得PaddlePaddle程序运行的更加迅速，我们通常使用GPU对PaddlePaddle程序进行加速，但安装GPU版本的PaddlePaddle需要先拥有满足以下条件的NVIDIA® GPU（具体安装流程和配置请务必参见NVIDIA官方文档：[For CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[For cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)）
	* *Cuda 工具包9.0配合cuDNN v7*
	* *Cuda 工具包8.0配合cuDNN v7*
	* *GPU运算能力超过1.0的硬件设备*-->

#### 选择如何编译PaddlePaddle
在MacOS 10.12/10.13的系统下我们提供2种的编译方式：

<!--* 直接本机源码编译-->
* Docker源码编译       




我们更加推荐**使用Docker进行编译**，因为我们在把工具和配置都安装在一个 Docker image 里。这样如果遇到问题，其他人可以复现问题以便帮助。另外，对于习惯使用Windows和MacOS的开发者来说，使用Docker就不用配置交叉编译环境了。需要强调的是：Docker 不会虚拟任何硬件，Docker container 里运行的编译工具实际上都是在本机的 CPU 和操作系统上直接运行的，性能和把编译工具安装在本机运行一样。        


      




<a name="mac_docker"></a>



<br/><br/>
##### ***使用Docker进行编译***

为了更好的使用Docker并避免发生问题，我们推荐使用**最高版本的Docker**，关于**安装和使用Docker**的细节请参阅Docker[官方文档](https://docs.docker.com/install/)。
> 请注意，在MacOS系统下登陆docker需要使用您的dockerID进行登录，否则将出现`Authenticate Failed`错误。       


当您已经**正确安装Docker**后你就可以开始**使用Docker编译PaddlePaddle**啦：

1. 进入Mac的终端

1. 请选择您希望储存PaddlePaddle的路径，然后在该路径下使用以下命令将PaddlePaddle的源码从github克隆到本地当前目录下名为Paddle的文件夹中：

	`git clone https://github.com/PaddlePaddle/Paddle.git`

2. 进入Paddle目录下： `cd Paddle`

3. 利用我们提供的镜像（使用该命令您可以不必提前下载镜像）：

	`docker run --name paddle-test -v $PWD:/paddle --network=host -it hub.baidubce.com/paddlepaddle/paddle:latest-dev /bin/bash`
	> --name paddle-test为您创建的Docker容器命名为paddle-test，-v $PWD:/paddle 将当前目录挂载到Docker容器中的/paddle目录下（Linux中$PWD变量会展开为当前路径的[绝对路径](https://baike.baidu.com/item/绝对路径/481185)），-it 与宿主机保持交互状态，`hub.baidubce.com/paddlepaddle/paddle:latest-dev` 使用名为`hub.baidubce.com/paddlepaddle/paddle:latest-dev`的镜像创建Docker容器，/bin/bash 进入容器后启动/bin/bash命令。

4. 进入Docker后进入paddle目录下：`cd paddle`

5. 切换到较稳定release分支下进行编译：

	`git checkout release/0.14.0`

6. 创建并进入/paddle/build路径下：

	`mkdir -p /paddle/build && cd /paddle/build`

7. 使用以下命令安装相关依赖：

	`pip install numpy==1.14.0`
	> 安装numpy 1.14.0，由于目前numpy1.15.0会引起大量warning，因此在numpy修复该问题前我们先使用numpy 1.14.0。

	`pip install protobuf==3.1.0`
	> 安装protobuf 3.1.0。

	`apt install patchelf`
	> 安装patchelf，PatchELF 是一个小而实用的程序，用于修改ELF可执行文件的动态链接器和RPATH。

8. 执行cmake：      
	
	>具体编译选项含义请参见[编译选项表](#Compile)<!--TODO： Link 编译选项表到这里-->


	*  对于需要编译**CPU版本PaddlePaddle**的用户：

		`cmake .. -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF`
		> 我们目前不支持MacOS下GPU版本PaddlePaddle的编译



9. 执行编译：

	`make -j$(nproc)`
	> 使用多核编译

10. 编译成功后进入`/paddle/build/python/dist`目录下找到生成的`.whl`包： `cd /paddle/build/python/dist`

11. 在当前机器或目标机器安装编译好的`.whl`包：

	`pip install （whl包的名字）`

至此您已经成功使用Docker安装PaddlePaddle，您只需要进入Docker容器后运行PaddlePaddle即可，更多Docker使用请参见[Docker官方文档](https://docs.docker.com)。

> 注：PaddlePaddle Docker镜像为了减小体积，默认没有安装`vim`，您可以在容器中执行 `apt-get install -y vim` 安装后，在容器中编辑代码。

恭喜您，现在您已经完成使用Docker编译PaddlePaddle的过程。






<br/><br/>
##### ***验证安装***
安装完成后您可以使用：`python` 进入Python解释器，然后使用`import paddle.fluid` 验证是否安装成功。

<br/><br/>
##### ***如何卸载PaddlePaddle***
请使用以下命令卸载PaddlePaddle：

* ***CPU版本的PaddlePaddle***: `pip uninstall paddlepaddle`





<!-- TODO add it back when finish test <br/><br/>
### **Windows下编译PaddlePaddle**

本说明将介绍如何在*64位台式机或笔记本电脑*以及Windows系统编译PaddlePaddle，我们支持的Windows系统需满足以下要求。

请注意：在其他系统上的尝试可能会导致安装失败。

* *Windows 7/8 and Windows 10 专业版/企业版*

#### 确定要编译的PaddlePaddle版本

* 支持CPU的PaddlePaddle。如果您的计算机没有 NVIDIA® GPU，则只能安装此版本。如果您的计算机有GPU，
也推荐您先安装CPU版本的PaddlePaddle，来检测您本地的环境是否适合。

* 支持GPU的PaddlePaddle。为了使PaddlePaddle程序运行更加迅速，我们通过GPU对PaddlePaddle程序进行加速，但安装GPU版本的PaddlePaddle需要先拥有满足以下条件的NVIDIA® GPU（具体安装流程和配置请务必参见NVIDIA官方文档：[For CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[For cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)）
	* *CUDA 工具包9.0配合cuDNN v7*
	* *CUDA 工具包8.0配合cuDNN v7*
	* *GPU运算能力超过1.0的硬件设备*


#### 选择如何安装PaddlePaddle
在Windows系统下请使用我们为您提供的[一键安装包](http://paddle-windows.bj.bcebos.com/PaddlePaddle-windows.zip)进行安装
	
> 我们提供的一键安装包将基于Docker为您进行便捷的安装流程


我们之所以使用**基于Docker的编译方式**，是因为我们在把工具和配置都安装在一个 Docker image 里，这样如果遇到问题，其他人可以复现问题以便帮助。另外，对于习惯使用Windows和MacOS的开发者来说，使用Docker就不用配置交叉编译环境了。需要强调的是：Docker 不会虚拟任何硬件，Docker container 里运行的编译工具实际上都是在本机的 CPU 和操作系统上直接运行的，性能和把编译工具安装在本机运行一样。        



<br/><br/>
##### ***验证安装***
安装完成后您可以使用：`python` 进入python解释器，然后使用`import paddle.fluid` 验证是否安装成功。

<br/><br/>
##### ***如何卸载PaddlePaddle***
请使用以下命令卸载PaddlePaddle：

* ***CPU版本的PaddlePaddle***: `pip uninstall PaddlePaddle`

* ***GPU版本的PaddlePaddle***: `pip uninstall PaddlePaddle-gpu`-->





<a name="FAQ"></a>
</br></br>
## **FAQ**
- CentOS6下如何编译python2.7为共享库? 
	
	> 使用以下指令：
	
		./configure --prefix=/usr/local/python2.7 --enable-shared   
		make && make install   

<!--TODO please add more F&Q parts here-->

- Ubuntu18.04下libidn11找不到？
	
	> 使用以下指令：
	
		apt install libidn11   

- Ubuntu编译时出现大量的代码段不能识别？
	
	> 这可能是由于cmake版本不匹配造成的，请在gcc的安装目录下使用以下指令：
		
		apt install gcc-4.8 g++-4.8
		cp gcc gcc.bak
		cp g++ g++.bak
		rm gcc
		rm g++
		ln -s gcc-4.8 gcc
		ln -s g++-4.8 g++
- 遇到paddlepaddle*.whl is not a supported wheel on this platform？
	> 出现这个问题的主要原因是，没有找到和当前系统匹配的paddlepaddle安装包。 请检查Python版本是否为2.7系列。另外最新的pip官方源中的安装包默认是manylinux1标准， 需要使用最新的pip (>9.0.0) 才可以安装。您可以执行以下指令更新您的pip：     
	
	pip install --upgrade pip     

	> 或者：     
	
	python -c "import pip; print(pip.pep425tags.get_supported())"    

	> 如果系统支持的是 linux_x86_64 而安装包是 manylinux1_x86_64 ，需要升级pip版本到最新； 如果系统支持 manylinux1_x86_64 而安装包	 （本地）是 linux_x86_64， 可以重命名这个whl包为 manylinux1_x86_64 再安装。

- 使用Docker编译出现问题？
	
	> 请参照GitHub上[Issue12079](https://github.com/PaddlePaddle/Paddle/issues/12079)

- 什么是 Docker?

  如果您没有听说 Docker，可以把它想象为一个类似 virtualenv 的系统，但是虚拟的不仅仅是 Python 的运行环境。

- Docker 还是虚拟机？

  有人用虚拟机来类比 Docker。需要强调的是：Docker 不会虚拟任何硬件，Docker container 里运行的编译工具实际上都是在本机的 CPU 和操作系统上直接运行的，性能和把编译工具安装在本机运行一样。

- 为什么用 Docker?

  把工具和配置都安装在一个 Docker image 里可以标准化编译环境。这样如果遇到问题，其他人可以复现问题以便帮助。

  另外，对于习惯使用Windows和MacOS的开发者来说，使用Docker就不用配置交叉编译环境了。

- 可以选择不用Docker吗？

  当然可以。大家可以用把开发工具安装进入 Docker image 一样的方式，把这些工具安装到本机。这篇文档介绍基于 Docker 的开发流程，是因为这个流程比其他方法都更简便。

- 学习 Docker 有多难？

  理解 Docker 并不难，大概花十分钟看一下[这篇文章](https://zhuanlan.zhihu.com/p/19902938)。
  这可以帮您省掉花一小时安装和配置各种开发工具，以及切换机器时需要新安装的辛苦。别忘了 PaddlePaddle 更新可能导致需要新的开发工具。更别提简化问题复现带来的好处了。

- 可以用 IDE 吗？

  当然可以，因为源码就在本机上。IDE 默认调用 make 之类的程序来编译源码，我们只需要配置 IDE 来调用 Docker 命令编译源码即可。

  很多 PaddlePaddle 开发者使用 Emacs。他们在自己的 `~/.emacs` 配置文件里加两行

    (global-set-key "\C-cc" 'compile)
    (setq compile-command "docker run --rm -it -v $(git rev-parse --show-toplevel):/paddle paddle:dev")

  就可以按 `Ctrl-C` 和 `c` 键来启动编译了。

- 可以并行编译吗？

  是的。我们的 Docker image 运行一个 [Bash 脚本](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/paddle/scripts/paddle_build.sh)。这个脚本调用`make -j$(nproc)` 来启动和 CPU 核一样多的进程来并行编译。

- Docker 需要 sudo？

  如果用自己的电脑开发，自然也就有管理员权限（sudo）了。如果用公用的电脑开发，需要请管理员安装和配置好 Docker。此外，PaddlePaddle 项目在努力开始支持其他不需要 sudo 的集装箱技术，比如 rkt。

- 在 Windows/MacOS 上编译很慢？

  Docker 在 Windows 和 MacOS 都可以运行。不过实际上是运行在一个 Linux 虚拟机上。可能需要注意给这个虚拟机多分配一些 CPU 和内存，以保证编译高效。具体做法请参考[issue627](https://github.com/PaddlePaddle/Paddle/issues/627)。

- 磁盘不够？

  本文中的例子里，`docker run` 命令里都用了 `--rm` 参数，这样保证运行结束之后的 containers 不会保留在磁盘上。可以用 `docker ps -a` 命令看到停止后但是没有删除的 containers。 `docker build` 命令有时候会产生一些中间结果，是没有名字的 images，也会占用磁盘。可以参考 [这篇文章](https://zaiste.net/posts/removing_docker_containers) 来清理这些内容。

- 在DockerToolbox下使用book时`http://localhost:8888/`无法打开？
	
   需要将localhost替换成虚拟机ip，一般需要在浏览器中输入：`http://192.168.99.100:8888/`

- pip install gpu版本的PaddlePaddle后运行出现SegmentFault如下：
   
  	 @ 0x7f6c8d214436 paddle::platform::EnforceNotMet::EnforceNotMet()
	 
   	 @ 0x7f6c8dfed666 paddle::platform::GetCUDADeviceCount() 
	 
  	 @ 0x7f6c8d2b93b6 paddle::framework::InitDevices()
   
   出现这个问题原因主要是由于您的显卡驱动低于对应CUDA版本的要求，请保证您的显卡驱动支持所使用的CUDA版本






<a name="third_party"></a>
</br></br>
## 附录

### **编译依赖表**

<p align="center">
<table>
	<thead>
	<tr>
		<th> 依赖包名称 </th>
		<th> 版本 </th>
		<th> 说明 </th>
		<th> 安装命令 </th>
	</tr>
	</thead>
	<tbody>
	<tr>
		<td> CMake </td>
		<td> 3.4 </td>
		<td>  </td>
		<td>  </td>
	</tr>
	<tr>
		<td> GCC </td>
		<td> 4.8 / 5.4 </td>
		<td>  推荐使用CentOS的devtools2 </td>
		<td>  </td>
	</tr>
		<tr>
		<td> Python </td>
		<td> 2.7.x. </td>
		<td> 依赖libpython2.7.so </td>
		<td> <code> apt install python-dev </code> 或 <code> yum install python-devel </code></td>
	</tr>
	<tr>
		<td> SWIG </td>
		<td> 最低 2.0 </td>
		<td>  </td>
		<td> <code>apt install swig </code> 或 <code> yum install swig </code> </td>
	</tr>
	<tr>
		<td> wget </td>
		<td> any </td>
		<td>  </td>
		<td> <code> apt install wget </code>  或 <code> yum install wget </code> </td>
	</tr>
	<tr>
		<td> openblas </td>
		<td> any </td>
		<td>  </td>
		<td>  </td>
	</tr>
	<tr>
		<td> pip </td>
		<td> 最低9.0.1 </td>
		<td>  </td>
		<td> <code> apt install python-pip </code> 或 <code> yum install Python-pip </code> </td>
	</tr>
	<tr>
		<td> numpy </td>
		<td> 最低1.12.0，最高1.14.0 </td>
		<td>  </td>
		<td> <code> pip install numpy==1.14.0 </code> </td>
	</tr>
	<tr>
		<td> protobuf </td>
		<td> 3.1.0 </td>
		<td>  </td>
		<td> <code> pip install protobuf==3.1.0 </code> </td>
	</tr>
	<tr>
		<td> wheel </td>
		<td> any </td>
		<td>  </td>
		<td> <code> pip install wheel </code> </td>
	</tr>
	<tr>
		<td> patchELF </td>
		<td> any </td>
		<td>  </td>
		<td> <code> apt install patchelf </code> 或参见github <a href="https://gist.github.com/ruario/80fefd174b3395d34c14">patchELF 官方文档</a></td>
	</tr>
	<tr>
		<td> go </td>
		<td> >=1.8 </td>
		<td> 可选 </td>
		<td>  </td>
	</tr>
	</tbody>
</table>
</p>


***
<a name="Compile"></a>
</br></br>
### **编译选项表**

<p align="center">
<table>
	<thead>
	<tr>
		<th> 选项 </th>
		<th> 说明 </th>
		<th> 默认值 </th>
	</tr>
	</thead>
	<tbody>
	<tr>
		<td> WITH_GPU </td>
		<td> 是否支持GPU </td>
		<td> ON </td>
	</tr>
	<tr>
		<td> WITH_C_API </td>
		<td> 是否仅编译CAPI </td>
		<td>  OFF </td>
	</tr>
		<tr>
		<td> WITH_DOUBLE </td>
		<td> 是否使用双精度浮点数 </td>
		<td> OFF </td>
	</tr>
	<tr>
		<td> WITH_DSO </td>
		<td> 是否运行时动态加载CUDA动态库，而非静态加载CUDA动态库 </td>
		<td> ON </td>
	</tr>
	<tr>
		<td> WITH_AVX </td>
		<td> 是否编译含有AVX指令集的PaddlePaddle二进制文件 </td>
		<td> ON </td>
	</tr>
	<tr>
		<td> WITH_PYTHON </td>
		<td> 是否内嵌PYTHON解释器 </td>
		<td> ON </td>
	</tr>
	<tr>
		<td> WITH_STYLE_CHECK </td>
		<td> 是否编译时进行代码风格检查 </td>
		<td> ON </td>
	</tr>
	<tr>
		<td> WITH_TESTING </td>
		<td> 是否开启单元测试 </td>
		<td> OFF </td>
	</tr>
	<tr>
		<td> WITH_DOC </td>
		<td> 是否编译中英文文档 </td>
		<td> OFF </td>
	</tr>
	<tr>
		<td> WITH_SWIG_PY </td>
		<td> 是否编译PYTHON的SWIG接口，该接口可用于预测和定制化训练 </td>
		<td> Auto </td>
	<tr>
		<td> WITH_GOLANG </td>
		<td> 是否编译go语言的可容错parameter server </td>
		<td> OFF </td>
	</tr>
	<tr>
		<td> WITH_MKL </td>
		<td> 是否使用MKL数学库，如果为否则是用OpenBLAS </td>
		<td> ON </td>
	</tr>
   </tbody>
</table>
</p>





**BLAS**

PaddlePaddle支持 [MKL](https://software.intel.com/en-us/mkl) 和 [OpenBlAS](http://www.openblas.net) 两种BLAS库。默认使用MKL。如果使用MKL并且机器含有AVX2指令集，还会下载MKL-DNN数学库，详细参考[这里](https://github.com/PaddlePaddle/Paddle/tree/develop/doc/design/mkldnn#cmake) 。

如果关闭MKL，则会使用OpenBLAS作为BLAS库。

**CUDA/cuDNN**

PaddlePaddle在编译时/运行时会自动找到系统中安装的CUDA和cuDNN库进行编译和执行。 使用参数 `-DCUDA_ARCH_NAME=Auto` 可以指定开启自动检测SM架构，加速编译。

PaddlePaddle可以使用cuDNN v5.1之后的任何一个版本来编译运行，但尽量请保持编译和运行使用的cuDNN是同一个版本。 我们推荐使用最新版本的cuDNN。

**编译选项的设置**

PaddePaddle通过编译时指定路径来实现引用各种BLAS/CUDA/cuDNN库。cmake编译时，首先在系统路径（ `/usr/liby` 和 `/usr/local/lib` ）中搜索这几个库，同时也会读取相关路径变量来进行搜索。 通过使用`-D`命令可以设置，例如：

> `cmake .. -DWITH_GPU=ON -DWITH_TESTING=OFF -DCUDNN_ROOT=/opt/cudnnv5`

**注意**：这几个编译选项的设置，只在第一次cmake的时候有效。如果之后想要重新设置，推荐清理整个编译目录（ rm -rf ）后，再指定。


***
<a name="whls"></a>
</br></br>
### **安装包列表**   

<p align="center">
<table>
	<thead>
	<tr>
		<th> 版本号 </th>
		<th> 版本说明 </th>
	</tr>
	</thead>
	<tbody>
	<tr>
		<td> paddlepaddle-gpu==0.14.0 </td>
		<td> 使用CUDA 9.0和cuDNN 7编译的0.14.0版本 </td>
	</tr>
	<tr>
		<td> paddlepaddle-gpu==0.14.0.post87 </td>
		<td> 使用CUDA 8.0和cuDNN 7编译的0.14.0版本 </td>
	</tr>
		<tr>
		<td> paddlepaddle-gpu==0.14.0.post85 </td>
		<td> 使用CUDA 8.0和cuDNN 5编译的0.14.0版本 </td>
	</tr>
	<tr>
		<td> paddlepaddle-gpu==0.13.0 </td>
		<td> 使用CUDA 9.0和cuDNN 7编译的0.13.0版本 </td>
	</tr>
	<tr>
		<td> paddlepaddle-gpu==0.12.0 </td>
		<td> 使用CUDA 8.0和cuDNN 5编译的0.12.0版本 </td>
	</tr>
	<tr>
		<td> paddlepaddle-gpu==0.11.0.post87 </td>
		<td> 使用CUDA 8.0和cuDNN 7编译的0.11.0版本 </td>
	</tr>
	<tr>
		<td> paddlepaddle-gpu==0.11.0.post85 </td>
		<td> 使用CUDA 8.0和cuDNN 5编译的0.11.0版本 </td>
	</tr>
	<tr>
		<td> paddlepaddle-gpu==0.11.0 </td>
		<td> 使用CUDA 7.5和cuDNN 5编译的0.11.0版本 </td>
	</tr>
   </tbody>
</table>
</p>


您可以在 [Release History](https://pypi.org/project/paddlepaddle-gpu/#history) 中找到PaddlePaddle-gpu的各个发行版本。

***
<a name="dockers"></a>
</br></br>
### **安装镜像表及简介**   
<p align="center">
<table>
	<thead>
	<tr>
		<th> 版本号 </th>
		<th> 版本说明 </th>
	</tr>
	</thead>
	<tbody>
	<tr>
		<td> hub.baidubce.com/paddlepaddle/paddle:latest </td>
		<td> 最新的预先安装好PaddlePaddle CPU版本的镜像 </td>
	</tr>
	<tr>
		<td> hub.baidubce.com/paddlepaddle/paddle:latest-dev </td>
		<td> 最新的PaddlePaddle的开发环境 </td>
	</tr>
		<tr>
		<td> hub.baidubce.com/paddlepaddle/paddle:[Version] </td>
		<td> 将version换成具体的版本，历史版本的预安装好PaddlePaddle的镜像 </td>
	</tr>
	<tr>
		<td> hub.baidubce.com/paddlepaddle/paddle:latest-gpu </td>
		<td> 最新的预先安装好PaddlePaddle GPU版本的镜像 </td>
	</tr>
   </tbody>
</table>
</p>


您可以在 [DockerHub](https://hub.docker.com/r/paddlepaddle/paddle/tags/) 中找到PaddlePaddle的各个发行的版本的docker镜像。


***
<a name="ciwhls"></a>
</br></br>
### **多版本whl包列表**   
<p align="center">
<table>
	<thead>
	<tr>
		<th> 版本说明 </th>
		<th> cp27-cp27mu </th>
		<th> cp27-cp27m </th>
	</tr>
	</thead>
	<tbody>
	<tr>
		<td> cpu_avx_mkl </td>
		<td> <a href="https://guest@paddleci.ngrok.io/repository/download/Manylinux1_CpuAvxCp27cp27mu/.lastSuccessful/paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl">	paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl</a></td>
		<td> <a href="https://guest@paddleci.ngrok.io/repository/download/Manylinux1_CpuAvxCp27cp27mu/.lastSuccessful/paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl">	paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl</a></td>
	</tr>
	<tr>
		<td> cpu_avx_mkl </td>
		<td> <a href="https://guest@paddleci.ngrok.io/repository/download/Manylinux1_CpuAvxOpenblas/.lastSuccessful/paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl">	paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl</a></td>
		<td> <a href="https://guest@paddleci.ngrok.io/repository/download/Manylinux1_CpuAvxOpenblas/.lastSuccessful/paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl">	paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl</a></td>
	</tr>
		<tr>
		<td> cpu_noavx_openblas </td>
		<td> <a href="https://guest@paddleci.ngrok.io/repository/download/Manylinux1_CpuNoavxOpenblas/.lastSuccessful/paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl">	paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl</a></td>
		<td><a href="https://guest@paddleci.ngrok.io/repository/download/Manylinux1_CpuNoavxOpenblas/.lastSuccessful/paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl">	paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl</a></td>
	</tr>
	<tr>
		<td> cuda8.0_cudnn5_avx_mkl </td>
		<td> <a href="https://guest@paddleci.ngrok.io/repository/download/Manylinux1_Cuda80cudnn5cp27cp27mu/.lastSuccessful/paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl">	paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl</a></td>
		<td><a href="https://guest@paddleci.ngrok.io/repository/download/Manylinux1_Cuda80cudnn5cp27cp27mu/.lastSuccessful/paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl">	paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl</a></td>
	</tr>
	<tr>
		<td> cuda8.0_cudnn7_avx_mkl </td>
		<td> <a href="https://guest@paddleci.ngrok.io/repository/download/Manylinux1_Cuda8cudnn7cp27cp27mu/.lastSuccessful/paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl">	paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl</a></td>
		<td><a href="https://guest@paddleci.ngrok.io/repository/download/Manylinux1_Cuda8cudnn7cp27cp27mu/.lastSuccessful/paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl">	paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl</a></td>
	</tr>
   </tbody>
</table>
</p>        







<!--TODO this part should be in a new webpage-->

</br></br>

### 在Docker中执行PaddlePaddle训练程序     

***

假设您已经在当前目录（比如在/home/work）编写了一个PaddlePaddle的程序: `train.py` （可以参考
[PaddlePaddleBook](http://www.paddlepaddle.org/docs/develop/book/01.fit_a_line/index.cn.html)
编写），就可以使用下面的命令开始执行训练：

     cd /home/work
     docker run -it -v $PWD:/work hub.baidubce.com/paddlepaddle/paddle:0.14.0 /work/train.py

上述命令中，`-it` 参数说明容器已交互式运行；`-v $PWD:/work`
指定将当前路径（Linux中$PWD变量会展开为当前路径的绝对路径）挂载到容器内部的:`/work`
目录: `hub.baidubce.com/paddlepaddle/paddle:0.14.0` 指定需要使用的容器； 最后`/work/train.py`为容器内执行的命令，即运行训练程序。

当然，您也可以进入到Docker容器中，以交互式的方式执行或调试您的代码：

     docker run -it -v $PWD:/work hub.baidubce.com/paddlepaddle/paddle:0.14.0 /bin/bash
     cd /work
     python train.py

**注：PaddlePaddle Docker镜像为了减小体积，默认没有安装vim，您可以在容器中执行** `apt-get install -y vim` **安装后，在容器中编辑代码。**

</br></br>

### 使用Docker启动PaddlePaddle Book教程

***

使用Docker可以快速在本地启动一个包含了PaddlePaddle官方Book教程的Jupyter Notebook，可以通过网页浏览。
PaddlePaddle Book是为用户和开发者制作的一个交互式的Jupyter Notebook。
如果您想要更深入了解deep learning，PaddlePaddle Book一定是您最好的选择。
大家可以通过它阅读教程，或者制作和分享带有代码、公式、图表、文字的交互式文档。

我们提供可以直接运行PaddlePaddle Book的Docker镜像，直接运行：

`docker run -p 8888:8888 hub.baidubce.com/paddlepaddle/book`

国内用户可以使用下面的镜像源来加速访问：

`docker run -p 8888:8888 hub.baidubce.com/paddlepaddle/book`

然后在浏览器中输入以下网址：

`http://localhost:8888/`

就这么简单，享受您的旅程！如有其他问题请参见[FAQ](#FAQ)

</br></br>
### 使用Docker执行GPU训练

***

为了保证GPU驱动能够在镜像里面正常运行，我们推荐使用
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)来运行镜像。
请不要忘记提前在物理机上安装GPU最新驱动。

`nvidia-docker run -it -v $PWD:/work hub.baidubce.com/paddlepaddle/paddle:latest-gpu /bin/bash`

**注: 如果没有安装nvidia-docker，可以尝试以下的方法，将CUDA库和Linux设备挂载到Docker容器内：**   

     export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') \
     $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
     export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
     docker run ${CUDA_SO} \
      ${DEVICES} -it hub.baidubce.com/paddlepaddle/paddle:latest-gpu


**关于AVX：**

AVX是一种CPU指令集，可以加速PaddlePaddle的计算。最新的PaddlePaddle Docker镜像默认
是开启AVX编译的，所以，如果您的电脑不支持AVX，需要单独[编译](/build_from_source_cn.html) PaddlePaddle为no-avx版本。

以下指令能检查Linux电脑是否支持AVX：

`if cat /proc/cpuinfo | grep -i avx; then echo Yes; else echo No; fi`

如果输出是No，就需要选择使用no-AVX的镜像
