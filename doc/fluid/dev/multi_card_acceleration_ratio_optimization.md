# SE-ResNeXt-152 多卡加速比调优过程
在对SE-ResNeXt-152模型进行多卡加速过程中发现多卡加速比未达到预期效果，为寻找其中原因进行了多种猜想和尝试，由于该过程较为曲折，且最终发现的原因较为特殊，故将此过程进行记录。

## 背景描述
当前使用P40 GPU卡且每张卡的batch size为40，在单卡、四卡及八卡情况下训练SE-ResNeXt-152模型，每秒钟可处理的图像数分别为23.33、73.41、111.73，获得加速比分别是3.15和4.79，比期望值差的较多。

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

## 各种猜想与验证
### 猜想一
卡与卡之间的通信导致随着卡数的增加训练时间变长。
##### 验证与结果
去掉卡与卡之间通信的操作，发现多卡加速比并无明显改善。同时也反映出在该情况下，卡与卡之间通信占用的时间较少。
### 猜想二
执行引擎中任务调度较慢。   
当前的执行引擎根据op的输入（input var）是否ready来决定是否执行该Op。在parallel executor中由一个队列（BlockingQueue）来存放这些ready的var，由一个线程从这个队列中取出所有ready的var，并对待执行的Op进行分析，如果某些op的输入已经ready了，就将这些Op放到一个线程池中运行，线程池完成op运行调用之后将该op的输出（即已经是ready的var）放到这个队列中，其中线程池的大小为`dev_count x 4`。    
起初认为线程切换的overhead较大，导致op的平均运行时间小于线程切换的时间，同时，队列（BlockingQueue）写的不够高效，导致大量线程在BlockingQueue这块发生了阻塞。因为SE-ResNeXt-152中包含将近4000个op，所以单卡情况下相当于平均每个Op运行时间为0.4287 ms，四卡情况下为0.137ms，8卡情况下为0.084ms，并且每个op运行调用结束后都需要向队列中放入已经ready的var。
##### 验证与结果
针对上述可能重写执行引擎中的调度算法。
- 去掉线程池，由一组固定的线程完成op的运行调用；
- 减少队列（BlockingQueue）的访问次数。在开始运行模型时，有些Op的输入已经是ready的，将这些ready的Op放到一个全局的队列（BlockingQueue）中，每个线程在启动时从该队列中取出Op并运行，同时每个线程内部维护一个局部的队列（Queue），该队列用于存放当前op运行完后处于ready状态的Op，如此一来，该线程下一次运行的op首先从局部队列中获取，如果局部队列为空，则从全局的队列中获取。   

结果发现使用新的调度算法得到的加速比无明显改善。   
为确定调度算法是否有问题，需进一步的验证，即将`op->run`方法替换成sleep操作，而sleep的时长设置也很关键，这本次验证中将sleep的时长设置为每个op的平均运行时间。   
结果发现模型的加速比与卡数成线性关系，由此可得新的调度算法是没有问题的，问题可能在于某个op在多卡运行时会变慢。

### 猜想三
模型在运行过程中某个调用在多卡情况下会变慢。出现这种现象的可能原因是某个调用是加锁的，所以多线程情况下可能会出现拥挤。

##### 验证与结果
程序运行中需调用的加锁操作有`RecordEvent`和`NewScope`等，将这些锁去除后发现多卡加速比并无明显改善。进而猜想可能是调用的某个Op在多卡情况下会变慢。

### 猜想四
某些Op在多卡情况下会变慢。虽然SE-ResNeXt-152在8卡情况下有32000个Op，但是也就20多种Op，所以理论上可以通过二分查找法快速确定多卡情况下会变慢的Op。

##### 验证与结果
为验证上述猜想，将模型中用得到的20多种Op进行二分法排查，即一部分op调用run方法，另一部分op调用sleep方法。经过若干次尝试之后发现在没有conv2d和conv2d_grad情况下，模型基本能达到线性加速。同时也发现这两个操作的执行时间占模型运行时间的绝大部分。    
由于conv2d和conv2d_grad操作的kernle为cudnn，为方便操作，将这两个操作的kernel换成GPU版的GEMM，并且这种情况下也能复现加速比问题。  
由于conv2d和conv2d_grad的GPU kernel中包含的操作有im2col、col2im和gemm，前两个是开发者写的CUDA kernel，最后一个调用的cuBlas。所以猜想可能是kernel的调用方式有问题。

### 猜想五
Kernel的调用方式有问题。目前框架中使用的stream有Eigen stream，CUDNN stream，CUDA stream，所以猜想可能是某地方将stream设置错误。

##### 验证与结果
对conv2d、conv2d\_grad以及device\_context部分进行代码检查，同时在框架之外单独写了一个CUDA程序，程序中通过main函数调用im2col（GPU代码）操作，且每个线程在输入大小相同的情况下连续调用im2col操作1000次，分析单卡和多卡情况下运行时间，结果发现两个种情况下程序的运行时间几乎相等。由此得出conv2d、conv2d\_grad以及device\_context是没有问题的。    

进一步分析会发现单独写的CUDA程序与SE-ResNeXt-152中调用im2col的方式有些差异，单独写的CUDA程序输入的数据量较大，而SE-ResNeXt-152模型在调用im2col是输入的数据量有大有小，可能在输入的数据量较小时会出现多卡变慢现象。

### 猜想六
输入数据量的大小导致conv2d在多卡情况下会变慢。在kernel调用时需要将kernel的参数值从CPU端传到GPU端，而该传递过程需要通过PCI-e，所以kernel的运行时间大致包括：参数传递的时间、kernel启动的时间、kernel计算的时间。因此在kernel要处理的数据量较小的情况下，参数传递和kernel启动的时间就会凸显，而参数传递需要经过PCI-e，所以在多卡运行时由于PCI-e带宽的限制导致多卡情况下加速比较低。
在PaddlePaddle用的绝大多数是CUDA Runtime API，CUDA Runtime API是CUDA driver API上层的一个C++软件库。一般情况下用CUDA Runtime API启动的Kernel函数为__global__函数，调用方式为：`func<<<>>>(...)`，NVCC在编译CUDA代码时会将__global__函数替换成其他的一些函数。比如`im2col`的定义形式为：
```
template <class T>
__global__ void im2col(const T* data_im, int num_outs, int im_height,
                       int im_width, int dilation_h, int dilation_w,
                       int filter_height, int filter_width, int stride_height,
                       int stride_width, int padding_height, int padding_width,
                       int col_height, int col_width, T* data_col) {...}
```
程序调用该Kernel函数的表达式为`im2col<<<x,x,x,x,>>>(...)`，NVCC在编译时会产生的部分调用为：
```
static void __device_stub__Z6im2colIfEvPKT_iiiiiiiiiiiiiPS0_(const float *__par0, int __par1, int __par2, int __par3, int __par4, int __par5, int __par6, int __par7, int __par8, int __par9, int __par10, int __par11, int __par12, int __par13, float *__par14){if (cudaSetupArgument((void *)(char *)&__par0, sizeof(__par0), (size_t)0UL) != cudaSuccess) return;if (cudaSetupArgument((void *)(char *)&__par1, sizeof(__par1), (size_t)8UL) != cudaSuccess) return;if (cudaSetupArgument((void *)(char *)&__par2, sizeof(__par2), (size_t)12UL) != cudaSuccess) return;if (cudaSetupArgument((void *)(char *)&__par3, sizeof(__par3), (size_t)16UL) != cudaSuccess) return;if (cudaSetupArgument((void *)(char *)&__par4, sizeof(__par4), (size_t)20UL) != cudaSuccess) return;if (cudaSetupArgument((void *)(char *)&__par5, sizeof(__par5), (size_t)24UL) != cudaSuccess) return;if (cudaSetupArgument((void *)(char *)&__par6, sizeof(__par6), (size_t)28UL) != cudaSuccess) return;if (cudaSetupArgument((void *)(char *)&__par7, sizeof(__par7), (size_t)32UL) != cudaSuccess) return;if (cudaSetupArgument((void *)(char *)&__par8, sizeof(__par8), (size_t)36UL) != cudaSuccess) return;if (cudaSetupArgument((void *)(char *)&__par9, sizeof(__par9), (size_t)40UL) != cudaSuccess) return;if (cudaSetupArgument((void *)(char *)&__par10, sizeof(__par10), (size_t)44UL) != cudaSuccess) return;if (cudaSetupArgument((void *)(char *)&__par11, sizeof(__par11), (size_t)48UL) != cudaSuccess) return;if (cudaSetupArgument((void *)(char *)&__par12, sizeof(__par12), (size_t)52UL) != cudaSuccess) return;if (cudaSetupArgument((void *)(char *)&__par13, sizeof(__par13), (size_t)56UL) != cudaSuccess) return;if (cudaSetupArgument((void *)(char *)&__par14, sizeof(__par14), (size_t)64UL) != cudaSuccess) return;{ volatile static char *__f __attribute__((unused)); __f = ((char *)((void ( *)(const float *, int, int, int, int, int, int, int, int, int, int, int, int, int, float *))im2col<float> )); (void)cudaLaunch(((char *)((void ( *)(const float *, int, int, int, int, int, int, int, int, int, int, int, int, int, float *))im2col<float> ))); };}
```
上述调用的功能是整合参数并将其发送到GPU端。

##### 验证与结果
为更接近模型的运行环境，在验证程序中加入任务调度策略，并且连续调用conv2d操作，同时将输入的大小设为`64, 3, 224, 224`，Filter大小为`128, 3, 3, 3}`，分别使用单卡和8卡运行，验证程序：[`Debug multicards`](https://github.com/PaddlePaddle/Paddle/compare/develop...reyoung:debug_multicards?expand=1)。   
结果发现在单卡情况下GPU的使用率约为90%\~100%，但是在8卡情况下GPU的使用率仅为30%\~40%。同时，在将CUDA Kernel的函数体删除，即Kernel中没有计算操作，该情况下单卡和8卡的GPU使用率与Kernel的函数体不被删除时的结果类似。由此可验证上述猜想是对的。

## 结论
综合上述所有实验得出的结论：在模型较大且输入数据较小时，多卡情况下卡与卡之间的计算并非完全独立。同时上述一系列的实验在一定程度上验证了Paddle中CUDA Kernel调用方式是没有问题的，并且执行引擎中的调度算法的性能也是可以接受的。   
基于上述所有猜想和验证，认为进一步提升SE-ResNeXt-152多卡加速比的方法有两个：改善硬件环境和对小Kernel做融合。
