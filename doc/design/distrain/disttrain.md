#核心概念
##master 

* 申请节点资源，维护节点列表 
	* 处理节点增减的问题 
* 对数据进行partition，分发对应的任务给trainer
	* push or pull？
* 维护task的队列信息
	* todo
	* pending
	* done
* 定时备份系统
	* checkpoint
	
##metasever（etcd,zookeeper)

* 保存task的队列数据  
* 保存psserver,trainer,master等节点列表信息
* 保证master只有一台提供决策

##psserver

* 保存参数
* 每个服务只保存一部分，所有server保存的数据是一个整体

##trainer

* 向master请求task，执行训练
* 保存参数信息到psserver
* 接受master对task的指令:
	* stop
	* re-request
	* start

##batch
* 数据训练的单位。用户指定的用来做训练的记录的数目。
* 由batchid唯一标识
	* batchid的生成可以在数据shuffle以后，用区间来定义。
		* id就是读取的办法？  

##parition
数据分发的单位。由batch组成。

* partition是一个数组，batch是其中的区间。
* 最好是系统支持区间读取，这样，分发的时候partition就是一个区间的指针。
	* 如hdfs
	* 如果由master发送实际的数据，这样速度会很慢。
	* 里边的batch可以是物理上连续或者不连续。如果是文件系统，连续有利于读取。
* 由paritionid唯一标识
	* 考虑到节点增减带来的repartition的问题？
		* parition的数目如果>>trainer的数目，这个问题可以不用考虑


##task
* 数据驱动系统,最小的单位是task。一个task对应一个batch(或者多个batch?)  
	* 考虑到数据多次迭代，接收的是partition。
* 由taskid唯一定义
	* taskid的生成规则？ 

##checkpoint
checkpoint就是系统的snapshot，需要保存的信息

* 参数值
	* 每个psserver保存的区间的范围
* 任务队列的信息

恢复系统的方式就是如果系统出现不能容忍的异常，则系统整体恢复到上一次正确的状态。丢失从正确到异常这段时间的操作。  

#同步训练
##iteration
##ring方式
* ["Bringing HPC Techniques to Deep Learning"]("http://research.baidu.com/bringing-hpc-techniques-deep-learning/")  
* 这种方式要求单个trainer保存所有的参数，对参数量超过内存的是否合适？  
* trainer就是psserver？

##传统方式

#异步训练
##极端情况？


#fault recoverable的考虑
##trainer异常
* 启动新的节点代替trainer
* 新节点load task信息
	* 同步系统需要等待该节点的完成
	* 异步不用？
	
##master异常
* 推举新的master节点，并load meserver中的meta信息
	* 如果有缩减节点的请求，master应该是最后被缩减或者不能被缩减的。 

部署情况：

* 3个或者多个
	* 采用多副本的形式。作为任务的维护者，只能有一个在提供服务，其他副本作为备份？

##psserver
###这是最大的单点。保存状态一般有两种方式:

* 流式记录
	* 记录更改
* checkpoint
	* 定时dump整体
 
考虑到参数巨大的量（重复）和更新的频率。采用checkpoint的方式比较符合我们的系统。比如每1个小时

* ps定时保存自己的参数到fs
	* 保存过程中系统不可变
* 出现异常，启动新的psserver以替代
* 系统load上一次的状态
	* master
	* ps
	* trainer

网络通信的消耗: 
恢复的时间： 

#架构图
#各个流程的详细设计
##正常训练流程
##psserver异常
##trainer异常
##master异常
##要求增加节点
##要求删除节点
