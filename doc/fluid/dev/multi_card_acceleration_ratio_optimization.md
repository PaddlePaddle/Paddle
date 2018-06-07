# SE-ResNeXt-152 多卡加速比调优过程
在对SE-ResNeXt-152模型进行多卡加速过程中发现多卡加速比未达到预期效果，为寻找其中原因而进行了多种猜想和尝试，由于该过程较为曲折，而且最终发现的原因又较为特殊，故将此过程进行记录。

## 背景描述
当前使用P40 GPU卡且每张卡的batch size为40，在单卡、四卡及八卡情况下训练SE-ResNeXt-152模型，每秒钟可处理的图像数分别为23.33、73.41、111.73。获得加速比分别是3.15和4.79，比期望值差的较多。

## 实验配置
- 硬件配置
   - 8卡 P40 
     - GPU卡之间的连接:
```
	GPU0	GPU1	GPU2	GPU3	GPU4	GPU5	GPU6	GPU7	mlx5_0	CPU Affinity
GPU0	 X 	PIX	PIX	PIX	PXB	PXB	PXB	PXB	SOC	0-13
GPU1	PIX	 X 	PIX	PIX	PXB	PXB	PXB	PXB	SOC	0-13
GPU2	PIX	PIX	 X 	PIX	PXB	PXB	PXB	PXB	SOC	0-13
GPU3	PIX	PIX	PIX	 X 	PXB	PXB	PXB	PXB	SOC	0-13
GPU4	PXB	PXB	PXB	PXB	 X 	PIX	PIX	PIX	SOC	0-13
GPU5	PXB	PXB	PXB	PXB	PIX	 X 	PIX	PIX	SOC	0-13
GPU6	PXB	PXB	PXB	PXB	PIX	PIX	 X 	PIX	SOC	0-13
GPU7	PXB	PXB	PXB	PXB	PIX	PIX	PIX	 X 	SOC	0-13
mlx5_0	SOC	SOC	SOC	SOC	SOC	SOC	SOC	SOC	 X

Legend:

  X   = Self
  SOC  = Connection traversing PCIe as well as the SMP link between CPU sockets(e.g. QPI)
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe switches (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing a single PCIe switch
  NV#  = Connection traversing a bonded set of # NVLinks
```
- 运行环境
  - Docker镜像：paddlepaddle/paddle:latest
- 模型配置: SE-ResNeXt-152  
  - batch size with one GPU: 40 

## 各种猜想
### 猜想一
卡与卡之间的通信导致随着卡数的增加训练时间变长。
##### 验证与结果
去掉卡与卡之间通信的操作，发现多卡加速比并无明显改善。
### 猜想二
执行引擎中任务调度问题。   
当前的执行引擎根据op的输入（in\_var）是否ready来决定是否执行该Op。在parallel executor中由一个队列（BlockingQueue）来存放这些ready的var，由一个线程从这个队列中取出所有ready的var，并对待执行的Op进行分析，如果某些op的输入已经ready了，就将这些Op放到一个线程池中运行，线程池完成op运行调用后将该op的输出（即已经是ready的var）放到这个队列中，其中线程池的大小为`dev_count x 4`。    
一开始认为线程切换的overhead较大，导致op的平均运行时间小于线程切换的时间，同时，队列（BlockingQueue）写的不够高效，导致大量线程在BlockingQueue这块发生了阻塞。因为SE-ResNeXt-152中包含将近4000个op，所以单卡情况下相当于每个线程运行需要0.4287 ms，四卡情况下为0.137ms，8卡情况下为0.084ms，并且每个op运行调用结束后都需要向队列中放数据。
##### 验证与结果
针对上问题重写执行引擎中的调度算法。
- 去掉线程池，由一组固定的线程完成op的运行调用；
- 减少队列（BlockingQueue）的访问次数。在开始运行模型时，有些Op的输入已经是ready的，将这些ready的Op放到一个全局的队列（BlockingQueue）中，每个线程在启动时从该队列中取出Op并运行，同时每个线程内部维护一个局部的队列（Queue），该队列用于存放当前op运行完后处于ready状态的Op，如此一来，该线程下一次运行的op首先从局部队列中获取，如果局部队列为空，则从全局的队列中获取。   

结果发现使用新的调度算法得到的加速比无明显改善。   
为进一步确定调度算法是否有问题，再进一步的验证中将`op->run`方法替换成sleep操作，而sleep的时长设置也很关键，这本次验证中将sleep的时长设置为每个op的平均运行时间84微秒。   
结果发现模型的加速比与卡数成线性关系，由此可得新的调度算法是没有问题的，问题可能在于某个op在多卡运行时会变慢。

### 猜想三
模型在运行过程中某个调用在多卡情况下会变慢。

##### 验证与结果

### 猜想四
框架中stream调用有问题。当前框架中存在的stream有eigen stream，cudnn stream，cuda stream。

##### 验证与结果

### 猜想五
CPU与GPU之前的通信

##### 验证与结果

## 结论
