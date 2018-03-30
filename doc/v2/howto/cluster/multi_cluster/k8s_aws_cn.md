#kubernetes on aws

我们将向你展示怎么样在AWS的kubernetes集群上运行分布式paddlepaddle训练，让我们从核心概念开始

##分布式paddlepaddle训练的核心概念

###分布式训练任务

一个分布式训练任务可以看做是一个kubernetes任务
每一个kubernetes任务都有相应的配置文件，此配置文件指定了像任务的pods和环境变量信息
在分布式训练任务中，我们可以如下操作：
1. 在分布式文件系统中，准备分区数据和配置文件（在此次教学中，我们会用到亚马逊分布式存储服务（EFS））
2. 创建和提交一个kubernetes任务配置到集群中开始训练

###参数服务和训练

在paddlepaddle集群中有两个角色：参数服务（pserver）者和训练者， 每一个参数服务过程都会以一个全局模型碎片存在。每一个训练者都可以进行本地模型拷贝，并可以利用本地数据更新模型。在这个训练过程中，训练者发送模型更新到参数服务中，参数服务职责就是聚合这些更新，以便于训练者可以把全局模型同步到本地。
为了能够和pserver联通，训练者需要每一个pserver的IP地址。在kubernetes中利用服务发现机制（比如：DNS、hostname）要比静态的IP地址要好一些，因为任何一个pod都会被杀掉然后新的pod被重启到另一个不同IP地址的node上。现在我们可以先用静态的IP地址方式，这种方式是可以更改的。
	参数服务者和训练者一块被打包成一个docker镜像，这个镜像会运行在被kebernetes集群调度的pod中。
###训练者ID
	每一个训练过程都需要一个训练ID，以0作为基础值，作为命令行参数传递。训练过程因此用这个ID去读取数据分区。
###训练
	利用shell脚本进入容器，可以看到许多kubernetes预先定义好的环境变量。这里可以定义任务identity，在任务中identity可以用来远程访问包含所有pod的kubernetes apiserver服务。
	每一个pod通过ip来排序。每一个pod的序列作为“pod id”。因为我们会在每一个pod中运行训练和参数服务，可以用“pod id”作为训练ID。入口脚本详细工作流程如下：
	1. 查找apiserver得到pod信息，通过ip排序来分配一个trainer_id。
	2. 从EFS持久化卷中复制训练数据到容器中。
	3. 从环境变量中解析paddle pserver和 paddle trainer的启动参数，然后开始启动流程。
	4. 以trainer_id来训练将自动把结果写入到EFS卷中。
##AWS的kubernetes中的paddlepaddle
###选择AWS服务区域
	这个教程需要多个AWS服务工作在一个区域中。在AWS创建任何东西之前，请检查链接https://aws.amazon.com/about-aws/global-infrastructure/regional-product-services/ 选择一个可以提供如下服务的区域：EC2, EFS, VPS, CloudFormation, KMS, VPC, S3。在教程中我们使用“Oregon(us-west-2)”作为例子。
###创建aws账户和IAM账户
	在每一个aws账户下可以创建多个IAM用户。允许为每一个IAM用户赋予权限，作为IAM用户可以创建/操作aws集群。
	注册aws账户，请遵循用户指南。在AWS账户下创建IAM用户和用户组，请遵循用户指南
	请注意此教程需要如下的IAM用户权限：
	AmazonEC2FullAccess
	AmazonS3FullAccess
	AmazonRoute53FullAccess
	AmazonRoute53DomainsFullAccess
	AmazonElasticFileSystemFullAccess
	AmazonVPCFullAccess
	IAMUserSSHKeys
	IAMFullAccess
	NetworkAdministrator
	AWSKeyManagementServicePowerUser
###下载kube-aws and kubectl
####kube-aws
	在AWS中kube-aws是一个自动部署集群的CLI工具
####kube-aws完整性验证
	提示：如果你用的是非官方版本（e.g RC release）的kube-aws，可以跳过这一步骤。引入	coreos的应用程序签名公钥:
	gpg2 --keyserver pgp.mit.edu --recv-key FC8A365E
####指纹验证
	gpg2 --fingerprint FC8A365E
	正确的指纹是：18AD 5014 C99E F7E3 BA5F 6CE9 50BD D3E0 FC8A 365E
	我们可以从发布页面中下载kube-aws，教程使用0.9.1版本
	验证tar包的GPG签名：
	PLATFORM=linux-amd64
	 # Or
	PLATFORM=darwin-amd64
	gpg2 --verify kube-aws-${PLATFORM}.tar.gz.sig kube-aws-${PLATFORM}.tar.gz

####安装kube-aws
	解压：
	tar zxvf kube-aws-${PLATFORM}.tar.gz
	添加到环境变量
	mv ${PLATFORM}/kube-aws /usr/local/bin
####kubectl
	kubectl是一个操作kubernetes集群的命令行接口
	利用curl工具从kubernetes发布页面中下载kubectl：
	# OS X
	curl -O https://storage.googleapis.com/kubernetes-release/release/"$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)"/bin/darwin/amd64/kubectl
	# Linux
	curl -O https://storage.googleapis.com/kubernetes-release/release/"$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)"/bin/linux/amd64/kubectl
	为了能是kubectl运行必须将之添加到环境变量中（e.g. /usr/local/bin）：
	chmod +x ./kubectl
	sudo mv ./kubectl /usr/local/bin/kubectl
###配置AWS证书
	首先检查这里并安装AWS命令行工具
	然后配置aws账户信息
	aws configure
	添加如下信息:
	AWS Access Key ID: YOUR_ACCESS_KEY_ID
	AWS Secrete Access Key: YOUR_SECRETE_ACCESS_KEY
	Default region name: us-west-2
	Default output format: json
	YOUR_ACCESS_KEY_ID和YOUR_SECRETE_ACCESS_KEY是创建aws账户和IAM账户的IAM的key和密码
	描述任何运行在你账户中的实例来验证凭据是否工作：
	aws ec2 describe-instances

###定义集群参数
####EC2秘钥对
	秘钥对将认证ssh访问你的EC2实例。秘钥对的公钥部分将配置到每一个COREOS节点中。
	遵循 EC2 Keypair用户指南来创建EC2秘钥对。
	你可以使用创建好的秘钥对名称来配置集群。
	在同一工作区中秘钥对为EC2实例唯一码。在教程中使用 us-west-2 ，所以请确认在这个区域（Oregon）中创建秘钥对。
	在浏览器中下载一个key-name.pem文件用来访问EC2实例，我们待会会用到。
####KMS秘钥
	亚马逊的KMS秘钥在TLS秘钥管理服务中用来加密和解密集群。如果你已经有可用的KMS秘钥，你可以跳过创建新秘钥这一步，提供现存秘钥的ARN字符串。
	利用aws命令行创建kms秘钥：
	aws kms --region=us-west-2 create-key --description="kube-aws assets"
	{
	    "KeyMetadata": {
	        "CreationDate": 1458235139.724,
	        "KeyState": "Enabled",
	        "Arn": "arn:aws:kms:us-west-2:aaaaaaaaaaaaa:key/xxxxxxxxxxxxxxxxxxx",
	        "AWSAccountId": "xxxxxxxxxxxxx",
	        "Enabled": true,
	        "KeyUsage": "ENCRYPT_DECRYPT",
	        "KeyId": "xxxxxxxxx",
	        "Description": "kube-aws assets"
	    }
	}
	我们一会用到arn的值。
	在IAM用户许可中添加多个内联策略。
	进入IAM Console。点击users按钮，点击刚才创建的用户，然后点击Add inline policy按钮，选择Custom Policy。
	粘贴内联策略：
	(Caution: node_0, node_1, node_2 directories represents PaddlePaddle node and train_id, not the Kubernetes node){
	    "Version": "2012-10-17",
	    "Statement": [
	        {
	            "Sid": "Stmt1482205552000",
	            "Effect": "Allow",
	            "Action": [
	                "kms:Decrypt",
	                "kms:Encrypt"
	            ],
	            "Resource": [
	                "arn:aws:kms:*:AWS_ACCOUNT_ID:key/*"
	            ]
	        },
	        {
	            "Sid": "Stmt1482205746000",
	            "Effect": "Allow",
	            "Action": [
	                "cloudformation:CreateStack",
	                "cloudformation:UpdateStack",
	                "cloudformation:DeleteStack",
	                "cloudformation:DescribeStacks",
	                "cloudformation:DescribeStackResource",
	                "cloudformation:GetTemplate",
	                "cloudformation:DescribeStackEvents"
	            ],
	            "Resource": [
	                "arn:aws:cloudformation:us-west-2:AWS_ACCOUNT_ID:stack/MY_CLUSTER_NAME/*"
	            ]
	        }
	    ]
	}

	version：值必须是“2012-10-17”
	AWS_ACCOUNT_ID：你可以从命令行中获取：
	aws sts get-caller-identity --output text --query Account
	MY_CLUSTER_NAME：选择一个你喜欢的MY_CLUSTER_NAME，稍后会用到。
	请注意，堆栈名称必须是正则表达式： [a-zA-Z][-a-zA-Z0-9]， 在名称中不能有“_” 或者“-”，否则kube-aws在下面步骤中会抛出异常。
####外部DNS名称
	当集群被创建后，基于DNS名称控制器将会暴露安全的TLS API。
	DNS名称含有CNAME指向到集群DNS名称或者记录指向集群的IP地址。
	我们稍后会用到DNS名称，如果没有DNS名称的话，你可以选择一个（比如：paddle）还可以修改/etc/hosts用本机的DNS名称和集群IP关联。还可以在AWS上增加一个名称服务来关联paddle集群IP，稍后步骤中会查找集群IP。
####S3 bucket
	在启动kubernetes集群前需要创建一个S3 bucket
	在AWS上创建s3 bucket会有许多的bugs，所以使用s3 console。
	链接到 create bucket，确保在us-west-2 (Oregon)上创建一个唯一的BUCKET_NAME。
####初始化assets
	在本机创建一个目录用来存放产生的assets：
	$ mkdir my-cluster
	$ cd my-cluster
	利用KMS Arn、秘钥对名称和前一步产生的DNS名称来初始化集群的CloudFormation栈：
	kube-aws init \
	--cluster-name=MY_CLUSTER_NAME \
	--external-dns-name=MY_EXTERNAL_DNS_NAME \
	--region=us-west-2 \
	--availability-zone=us-west-2a \
	--key-name=KEY_PAIR_NAME \
	--kms-key-arn="arn:aws:kms:us-west-2:xxxxxxxxxx:key/xxxxxxxxxxxxxxxxxxx"

	
	MY_CLUSTER_NAME: the one you picked in KMS key
	MY_EXTERNAL_DNS_NAME: see External DNS name
	KEY_PAIR_NAME: see EC2 key pair
	--kms-key-arn: the “Arn” in KMS key
	这里的us-west-2a用于参数--availability-zone，但必须在AWS账户的有效可用区中。
	如果不能切换到其他的有效可用区（e.g., us-west-2a, or us-west-2b），请检查us-west-2a是支持aws ec2 --region us-west-2 describe-availability-zones。
	现在在asset目录中就有了集群的主配置文件cluster.yaml。
	默认情况下kube-aws会创建一个工作节点，修改cluster.yaml让workcount从1个节点变成3个节点.

####呈现asset目录内容
	在这个简单的例子中，你可以使用kuber-aws生成TLS身份和证书
	kube-aws render credentials --generate-ca
	下一步在asset目录中生成一组集群assets
	kube-aws render stack
	asserts(模板和凭证)用于创建、更新和当前目录被创建的kubernetes集群相关联。
###启动kubernetes集群
	创建一个在CloudFormation模板上定义好的实例
	现在让我们创建集群（在命令行中选择任意的 PREFIX）
	kube-aws up --s3-uri s3://BUCKET_NAME/PREFIX
	BUCKET_NAME：在S3 bucket上使用的bucket名称
####配置DNS
	你可以执行命令 kube-aws status来查看创建后集群的API
	$ kube-aws status
	Cluster Name:       paddle-cluster
	Controller DNS Name:    paddle-cl-ElbAPISe-EEOI3EZPR86C-531251350.us-west-2.elb.amazonaws.com
	如果你用DNS名称，在ip上设置任何记录或是安装CNAME点到Controller DNS Name (paddle-cl-ElbAPISe-EEOI3EZPR86C-531251350.us-west-2.elb.amazonaws.com)
####查询IP地址
	用命令dig去检查负载均衡器的域名来获取ip地址
	$ dig paddle-cl-ElbAPISe-EEOI3EZPR86C-531251350.us-west-2.elb.amazonaws.com
	;; QUESTION SECTION:
	;paddle-cl-ElbAPISe-EEOI3EZPR86C-531251350.us-west-2.elb.amazonaws.com. IN A
	;; ANSWER SECTION:
	paddle-cl-ElbAPISe-EEOI3EZPR86C-531251350.us-west-2.elb.amazonaws.com. 59 IN A 54.241.164.52
	paddle-cl-ElbAPISe-EEOI3EZPR86C-531251350.us-west-2.elb.amazonaws.com. 59 IN A 54.67.102.112
	在上面的例子中，54.241.164.52, 54.67.102.112这两个ip都将是工作状态
	如果你有DNS名称，设置记录到ip上，然后你可以跳过“Access the cluster”这一步
	如果没有自己的DNS名称
	更新本地的DNS关联
	编辑/etc/hosts文件用DNS关联IP
####在VPC上添加route53私有名称服务
	打开Route53 Console
	根据配置创建域名zone
	domain名称为：“paddle”
	类型：“Private hosted zone for amazon VPC”
	VPC ID: <Your VPC ID>
	添加记录
	点击zone中刚创建的“paddle”
	点击按钮“Create record set”
	名称： leave blank
	类型：A
	值：<kube-controller ec2 private ip>
	检查名称服务
	连接通过kube-aws via ssh创建的任何实例
	运行命令”host paddle“，看看是否ip为返回的kube-controller的私有IP
####进入集群
	集群运行后如下命令会看到：
	$ kubectl --kubeconfig=kubeconfig get nodes 
	NAME                                       STATUS    AGE
	ip-10-0-0-134.us-west-2.compute.internal   Ready     6m
	ip-10-0-0-238.us-west-2.compute.internal   Ready     6m
	ip-10-0-0-50.us-west-2.compute.internal    Ready     6m
	ip-10-0-0-55.us-west-2.compute.internal    Ready     6m

###集群安装弹性文件系统
	
	训练数据存放在AWS上的EFS分布式文件系统中
	1. 在 security group console为EFS创建一个安全组
	2. 可以看到paddle-cluster-sg-worker(在sg-055ee37d镜像中)安全组id
	图
	3. 增加安全组paddle-efs，以paddle-cluster-sg-worker的group id作为用户源和ALL TCP入栈规则。增加vpc paddle-cluster-vpc, 确保可用区是在Initialize Assets的时候用到的那一个。
	图
	4. 利用paddle-cluster-vpc私有网络在EFS console 中创建弹性文件系统
	图
###开始在AWS上进行paddlepaddle的训练
	配置kubernetes卷指向EFS
	首先需要创建一个持久卷PersistentVolume到EFS上
	用pv.yaml形式来保存
	
	apiVersion: v1
	kind: PersistentVolume
	metadata:
	  name: efsvol
	spec:
	  capacity:
	    storage: 100Gi
	  accessModes:
	    - ReadWriteMany
	  nfs:
	    server: EFS_DNS_NAME
	    path: "/"
	EFS_DNS_NAME：DNS名称最好能描述我们创建的paddle-efs，看起来像fs-2cbf7385.efs.us-west-2.amazonaws.com
	运行下面的命令来创建持久卷
	kubectl --kubeconfig=kubeconfig create -f pv.yaml
	下一步创建 PersistentVolumeClaim来声明持久卷
	用pvc.yaml来保存
	kind: PersistentVolumeClaim
	apiVersion: v1
	metadata:
	  name: efsvol
	spec:
	  accessModes:
	    - ReadWriteMany
	  resources:
	    requests:
	      storage: 50Gi
	运行下面命令来创建持久卷声明
	kubectl --kubeconfig=kubeconfig create -f pvc.yaml
####准备训练数据
	启动kubernetes job在我们创建的持久层上进行下载、保存并均匀拆分训练数据为3份
	用paddle-data-job.yaml保存
	apiVersion: batch/v1
	kind: Job
	metadata:
	  name: paddle-data
	spec:
	  template:
	    metadata:
	      name: pi
	    spec:
	      containers:
	      - name: paddle-data
	        image: paddlepaddle/paddle-tutorial:k8s_data
	        imagePullPolicy: Always
	        volumeMounts:
	        - mountPath: "/efs"
	          name: efs
	        env:
	        - name: OUT_DIR
	          value: /efs/paddle-cluster-job
	        - name: SPLIT_COUNT
	          value: "3"
	      volumes:
	        - name: efs
	          persistentVolumeClaim:
	            claimName: efsvol
	      restartPolicy: Never
	运行下面的命令来启动任务
	kubectl --kubeconfig=kubeconfig create -f paddle-data-job.yaml
	任务运行大概需要7分钟，可以使用下面命令查看任务状态，直到paddle-data任务的SUCCESSFUL状态为1时成功，这里here有怎样创建镜像的源码。
####开始训练
	现在可以开始运行paddle的训练任务，用addle-cluster-job.yaml进行保存
	apiVersion: batch/v1
	kind: Job
	metadata:
	  name: paddle-cluster-job
	spec:
	  parallelism: 3
	  completions: 3
	  template:
	    metadata:
	      name: paddle-cluster-job
	    spec:
	      volumes:
	      - name: efs
	        persistentVolumeClaim:
	          claimName: efsvol
	      containers:
	      - name: trainer
	        image: paddlepaddle/paddle-tutorial:k8s_train
	        command: ["bin/bash",  "-c", "/root/start.sh"]
	        env:
	        - name: JOB_NAME
	          value: paddle-cluster-job
	        - name: JOB_PATH
	          value: /home/jobpath
	        - name: JOB_NAMESPACE
	          value: default
	        - name: TRAIN_CONFIG_DIR
	          value: quick_start
	        - name: CONF_PADDLE_NIC
	          value: eth0
	        - name: CONF_PADDLE_PORT
	          value: "7164"
	        - name: CONF_PADDLE_PORTS_NUM
	          value: "2"
	        - name: CONF_PADDLE_PORTS_NUM_SPARSE
	          value: "2"
	        - name: CONF_PADDLE_GRADIENT_NUM
	          value: "3"
	        - name: TRAINER_COUNT
	          value: "3"
	        volumeMounts:
	        - mountPath: "/home/jobpath"
	          name: efs
	        ports:
	        - name: jobport0
	          hostPort: 7164
	          containerPort: 7164
	        - name: jobport1
	          hostPort: 7165
	          containerPort: 7165
	        - name: jobport2
	          hostPort: 7166
	          containerPort: 7166
	        - name: jobport3
	          hostPort: 7167
	          containerPort: 7167
	      restartPolicy: Never

	parallelism: 3, completions: 3意思是这个任务会同时开启3个paddlepaddle的pod，当pod启动后3个任务将被完成。
	env 参数代表容器的环境变量，在这里指定paddlepaddle的参数
	ports 指定TCP端口7164 - 7167和pserver进行连接，port从CONF_PADDLE_PORT (7164)到CONF_PADDLE_PORT + CONF_PADDLE_PORTS_NUM + CONF_PADDLE_PORTS_NUM_SPARSE - 1 (7167)
	我们使用多个端口密集和稀疏参数的更新来提高延迟
	运行下面命令来启动任务
	kubectl --kubeconfig=kubeconfig create -f paddle-claster-job.yaml
	检查pods信息
	$ kubectl --kubeconfig=kubeconfig get pods
	NAME                       READY     STATUS    RESTARTS   AGE
	paddle-cluster-job-cm469   1/1       Running   0          9m
	paddle-cluster-job-fnt03   1/1       Running   0          9m
	paddle-cluster-job-jx4xr   1/1       Running   0          9m
	检查指定pod的控制台输出
	kubectl --kubeconfig=kubeconfig log -f POD_NAME
	POD_NAME：任何一个pod的名称（e.g., paddle-cluster-job-cm469）
	运行kubectl --kubeconfig=kubeconfig describe job paddle-cluster-job来检查训练任务的状态，将会在大约20分钟完成
	pserver和trainer的细节都隐藏在docker镜像paddlepaddle/paddle-tutorial:k8s_train中，这里here有创建docker镜像的源码
####检查训练输出
	训练输出（模型快照和日志）将被保存在EFS上。我们可以用ssh登录到EC2的工作节点上，查看mount过的EFS和训练输出。
	1. ssh登录EC2工作节点
	chmod 400 key-name.pem
	ssh -i key-name.pem core@INSTANCE_IP
	INSTANCE_IP: EC2上kubernetes工作节点的公共IP地址，进入EC2 console中检查任何paddle-cluster-kube-aws-worker实例的public IP
	2. 挂载EFS
	mkdir efs
	sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 EFS_DNS_NAME:/ efs
	EFS_DNS_NAME：DNS名称最好能描述我们创建的paddle-efs，看起来像fs-2cbf7385.efs.us-west-2.amazonaws.com
	文件夹efs上有这结构相似的node信息：
	-- paddle-cluster-job
    |-- ...
    |-- output
    |   |-- node_0
    |   |   |-- server.log
    |   |   `-- train.log
    |   |-- node_1
    |   |   |-- server.log
    |   |   `-- train.log
    |   |-- node_2
    |   |   |-- server.log
    |   |   `-- train.log
    |   |-- pass-00000
    |   |   |-- ___fc_layer_0__.w0
    |   |   |-- ___fc_layer_0__.wbias
    |   |   |-- done
    |   |   |-- path.txt
    |   |   `-- trainer_config.lr.py
    |   |-- pass-00001...

    server.log 是pserver的log日志，train.log是trainer的log日志，模型快照和描述存放在pass-0000*

###kubernetes集群卸载或删除
	删除EFS
	到EFS console中删除创建的EFS卷
	删除安全组
	去Security Group Console删除安全组paddle-efs
	删除S3 bucket
	进入 S3 Console删除S3 bucket
	销毁集群
	kube-aws destroy
	命令会立刻返回，但需要大约5分钟来销毁集群
	可以进入CludFormation Console检查销毁的过程。
