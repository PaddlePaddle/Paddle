因为我们不提供非Ubuntu的bulid支持，所以如果用户用其他操作系统，比如CoreOS、CentOS、MacOS X、Windows，开发都得在docker里。所以需要能build本地修改后的代码。

我们可能需要两个 Docker images：

1. development image：不包括源码，但是包括开发环境（预先安装好各种工具），也就是说Dockerfile.dev里既不需要  COPY 也不需要 RUN git clone。虽然这个image和源码无关，但是不同版本的源码需要依赖不同的第三方库，所以这个image的tag里还是要包含git branch/tag name，比如叫做 `paddlepaddle/paddle:dev-0.10.0rc1`，这里的0.10.0.rc1是一个branch name，其中rc是release candidate的意思。正是发布之后就成了master branch里的一个tag，叫做0.10.0。

1. production image： 不包括编译环境，也不包括源码，只包括build好的libpaddle.so和必要的Python packages，用于在Kubernetes机群上跑应用的image。比如叫做 `paddlepaddle/paddle:0.10.0rc1`。

从1.生成2.的过程如下：

1. 在本机（host）上开发。假设源码位于 `~/work/paddle`。

1. 用dev image build 我们的源码：
   ```bash
   docker run -it -p 2022:22 -v $PWD:/paddle paddlepaddle/paddle:dev-0.10.0rc1  /paddle/build.sh
   ```  
   注意，这里的 `-v ` 参数把host上的源码目录里的内容映射到了container里的`/paddle` 目录；而container里的 `/paddle/build.sh` 就是源码目录里的 `build.sh`。上述命令调用了本地源码中的 bulid.sh 来build了本地源码，结果在container里的 `/paddle/build` 目录里，也就是本地的源码目录里的 `build` 子目录。

1. 我们希望上述 `build.sh` 脚本在 `build` 子目录里生成一个Dockerfile，使得我们可以运行：
   ```bash
   docker build -t paddle  ./build
   ```
   来生成我们的production image。
   
1. 有了这个production image之后，我们可能会希望docker push 到dockerhub.com的我们自己的名下，然后可以用来启动本地或者远程（Kubernetes）jobs：

   ```bash
   docker tag paddle yiwang/paddle:did-some-change
   docker push
   paddlectl run yiwang/paddle:did-some-change /paddle/demo/mnist/train.py
   ```

   其中 paddlectl 应该是我们自己写的一个脚本，调用kubectl来在Kubernetes机群上启动一个job的。


曾经的讨论背景：   
["PR 1599"](https://github.com/PaddlePaddle/Paddle/pull/1599)  
["PR 1598"](https://github.com/PaddlePaddle/Paddle/pull/1598)
