# 用Docker编译和测试PaddlePaddle

## 需要的软硬件

为了开发PaddlePaddle，我们需要

1. 一台电脑，可以装的是 Linux, BSD, Windows 或者 MacOS 操作系统，以及
1. Docker。

不需要依赖其他任何软件了。即便是 Python 和 GCC 都不需要，因为我们会把所有编译工具都安装进一个 Docker image 里。

## 总体流程

1. 获取源码

   ```bash
   git clone https://github.com/paddlepaddle/paddle
   ```

2. 安装开发工具到 Docker image 里

   ```bash
   cd paddle; docker build -t paddle:dev .
   ```

   请注意这个命令结尾处的 `.`；它表示 `docker build` 应该读取当前目录下的 [`Dockerfile`文件](https://github.com/PaddlePaddle/Paddle/blob/develop/Dockerfile)，按照其内容创建一个名为 `paddle:dev` 的 Docker image，并且把各种开发工具安装进去。

3. 编译

   以下命令启动一个 Docker container 来执行 `paddle:dev` 这个 Docker image，同时把当前目录（源码树根目录）映射为 container 里的 `/paddle` 目录，并且运行 `Dockerfile` 描述的默认入口程序 [`build.sh`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/scripts/docker/build.sh)。这个脚本调用 `cmake` 和 `make` 来编译 `/paddle` 里的源码，结果输出到 `/paddle/build`，也就是本地的源码树根目录里的 `build` 子目录。

   ```bash
   docker run --rm -v $PWD:/paddle paddle:dev
   ```

   上述命令编译出一个 CUDA-enabled 版本。如果我们只需要编译一个只支持 CPU 的版本，可以用

   ```bash
   docker run --rm -e WITH_GPU=OFF -v $PWD:/paddle paddle:dev
   ```

4. 运行单元测试

   用本机的第一个 GPU 来运行包括 GPU 单元测试在内的所有单元测试：

   ```bash
   NV_GPU=0 nvidia-docker run --rm -v $PWD:/paddle paddle:dev bash -c "cd /paddle/build; ctest"
   ```

   如果编译的时候我们用了 `WITH_GPU=OFF` 选项，那么编译过程只会产生 CPU-based 单元测试，那么我们也就不需要 nvidia-docker 来运行单元测试了。我们只需要：

   ```bash
   docker run --rm -v $PWD:/paddle paddle:dev bash -c "cd /paddle/build; ctest"
   ```

   有时候我们只想运行一个特定的单元测试，比如 `memory_test`，我们可以

   ```bash
   nvidia-docker run --rm -v $PWD:/paddle paddle:dev bash -c "cd /paddle/build; ctest -V -R memory_test"
   ```

5. 清理

   有时候我们会希望清理掉已经下载的第三方依赖以及已经编译的二进制文件。此时只需要：

   ```bash
   rm -rf build
   ```

## 为什么要 Docker 呀？

- 什么是 Docker?

  如果您没有听说 Docker，可以把它想象为一个类似 virtualenv 的系统，但是虚拟的不仅仅是 Python 的运行环境。

- Docker 还是虚拟机？

  有人用虚拟机来类比 Docker。需要强调的是：Docker 不会虚拟任何硬件，Docker container 里运行的编译工具实际上都是在本机的 CPU 和操作系统上直接运行的，性能和把编译工具安装在本机运行一样。

- 为什么用 Docker?

  把工具和配置都安装在一个 Docker image 里可以标准化编译环境。这样如果遇到问题，其他人可以复现问题以便帮助。

  另外，对于习惯使用Windows和MacOS的开发者来说，使用Docker就不用配置交叉编译环境了。

- 我可以选择不用Docker吗？

  当然可以。大家可以用把开发工具安装进入 Docker image 一样的方式，把这些工具安装到本机。这篇文档介绍基于 Docker 的开发流程，是因为这个流程比其他方法都更简便。

- 学习 Docker 有多难？

  理解 Docker 并不难，大概花十分钟看一下[这篇文章](https://zhuanlan.zhihu.com/p/19902938)。这可以帮您省掉花一小时安装和配置各种开发工具，以及切换机器时需要新安装的辛苦。别忘了 PaddlePaddle 更新可能导致需要新的开发工具。更别提简化问题复现带来的好处了。

- 我可以用 IDE 吗？

  当然可以，因为源码就在本机上。IDE 默认调用 make 之类的程序来编译源码，我们只需要配置 IDE 来调用 Docker 命令编译源码即可。

  很多 PaddlePaddle 开发者使用 Emacs。他们在自己的 `~/.emacs` 配置文件里加两行

  ```emacs
  (global-set-key "\C-cc" 'compile)
  (setq compile-command
   "docker run --rm -it -v $(git rev-parse --show-toplevel):/paddle paddle:dev")
  ```

  就可以按 `Ctrl-C` 和 `c` 键来启动编译了。

- 可以并行编译吗？

  是的。我们的 Docker image 运行一个 [Bash 脚本](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/scripts/docker/build.sh)。这个脚本调用 `make -j$(nproc)` 来启动和 CPU 核一样多的进程来并行编译。

## 可能碰到的问题

- Docker 需要 sudo

  如果用自己的电脑开发，自然也就有管理员权限（sudo）了。如果用公用的电脑开发，需要请管理员安装和配置好 Docker。此外，PaddlePaddle 项目在努力开始支持其他不需要 sudo 的集装箱技术，比如 rkt。

- 在 Windows/MacOS 上编译很慢

  Docker 在 Windows 和 MacOS 都可以运行。不过实际上是运行在一个 Linux 虚拟机上。可能需要注意给这个虚拟机多分配一些 CPU 和内存，以保证编译高效。具体做法请参考[这个issue](https://github.com/PaddlePaddle/Paddle/issues/627)。

- 磁盘不够

  本文中的例子里，`docker run` 命令里都用了 `--rm` 参数，这样保证运行结束之后的 containers 不会保留在磁盘上。可以用 `docker ps -a` 命令看到停止后但是没有删除的 containers。`docker build` 命令有时候会产生一些中间结果，是没有名字的 images，也会占用磁盘。可以参考[这篇文章](https://zaiste.net/posts/removing_docker_containers/)来清理这些内容。
