# SE-ResNeXt-152 多卡加速比调优过程
在对SE-ResNeXt-152模型进行多卡加速过程中发现多卡加速比未达到预期效果，为寻找其中原因而进行了多种猜想和尝试，由于该过程较为曲折，且最终发现的原因较为特殊，故将此过程进行记录。

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
执行引擎中任务调度较慢。   
当前的执行引擎根据op的输入（input var）是否ready来决定是否执行该Op。在parallel executor中由一个队列（BlockingQueue）来存放这些ready的var，由一个线程从这个队列中取出所有ready的var，并对待执行的Op进行分析，如果某些op的输入已经ready了，就将这些Op放到一个线程池中运行，线程池完成op运行调用之后将该op的输出（即已经是ready的var）放到这个队列中，其中线程池的大小为`dev_count x 4`。    
一开始认为线程切换的overhead较大，导致op的平均运行时间小于线程切换的时间，同时，队列（BlockingQueue）写的不够高效，导致大量线程在BlockingQueue这块发生了阻塞。因为SE-ResNeXt-152中包含将近4000个op，所以单卡情况下相当于平均每个Op运行需要0.4287 ms，四卡情况下为0.137ms，8卡情况下为0.084ms，并且每个op运行调用结束后都需要向队列中放数据。
##### 验证与结果
针对上述可能重写执行引擎中的调度算法。
- 去掉线程池，由一组固定的线程完成op的运行调用；
- 减少队列（BlockingQueue）的访问次数。在开始运行模型时，有些Op的输入已经是ready的，将这些ready的Op放到一个全局的队列（BlockingQueue）中，每个线程在启动时从该队列中取出Op并运行，同时每个线程内部维护一个局部的队列（Queue），该队列用于存放当前op运行完后处于ready状态的Op，如此一来，该线程下一次运行的op首先从局部队列中获取，如果局部队列为空，则从全局的队列中获取。   

结果发现使用新的调度算法得到的加速比无明显改善。   
为进一步确定调度算法是否有问题，需进一步的验证，即将`op->run`方法替换成sleep操作，而sleep的时长设置也很关键，这本次验证中将sleep的时长设置为每个op的平均运行时间。   
结果发现模型的加速比与卡数成线性关系，由此可得新的调度算法是没有问题的，问题可能在于某个op在多卡运行时会变慢。

### 猜想三
模型在运行过程中某个调用在多卡情况下会变慢。出现这种现象的可能原因是某个调用是加锁的，所以多线程情况下需要可能会出现拥挤。

##### 验证与结果
程序运行中需调用的加锁操作有`RecordEvent`和`NewScope`等，将这些锁去除后发现多卡加速比并无明显改善。进而猜想可能是调用的某个Op在多卡情况下会变慢。

### 猜想四
某个Op在多卡情况下。虽然SE-ResNeXt-152在8卡情况下由32000个op，但是也就20多种Op，所以可以通过二分法找到多卡情况下会变慢的Op。

##### 验证与结果
为验证上述猜想，将模型中用得到的20多种op通过二分法排查，即一部分op调用run方法，另一部分op调用sleep方法。经过若干次尝试之后发现在没有conv2d和conv2d_grad情况下，模型基本能达到线性加速。同时也发现这两个操作的执行时间占模型运行时间的大部分。    
用于conv2d和conv2d_grad操作的kernle为cudnn，为方便操作，将这两个操作的kernel换成GPU版的Gemm，并且这种情况下也能复现过上述问题。
由于conv2d和conv2d_grad的GPU kernel中包含的操作由im2col、col2im和gemm，前两个是手写的CUDA kernel，最后一个调用的cuBlas。所以猜想可能是kernel的调用方式有问题。

### 猜想五
Kernel的调用方式有问题。目前框架中使用的stream有Eigen stream，CUDNN stream，CUDA stream，所以猜想可能是某地方将stream设置错误。

##### 验证与结果
对conv2d、conv2d\_grad以及device\_context部分进行代码检查，同时在框架之外单独写了一个CUDA程序，程序中通过main函数调用im2col（GPU代码）每个线程在输入大小下同的情况下连续调用im2col操作1000次，分析单卡和多卡情况下运行时间，结果发现两个种情况下程序的运行时间几乎相等。由此得出conv2d、conv2d\_grad以及device\_context是没有问题的。     
单独写的CUDA程序与SE-ResNeXt-152中调用im2col的方式有些差异，单独写的CUDA程序输入的数据长度较大，且长度都是一样的，而SE-ResNeXt-152模型在调用im2col是输入的数据长度有大有小，可能在输入的数据长度较小时会出现多卡变慢现象。

### 猜想六
输入数据长度较小，导致CUDA kernel在多卡情况下运行会变慢。因为GPU和CPU之间是通过PCI-e进行通信的


##### 验证与结果

## 结论
