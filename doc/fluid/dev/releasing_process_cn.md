# PaddlePaddle发行规范

PaddlePaddle使用git-flow branching model做分支管理，使用[Semantic Versioning](http://semver.org/)标准表示PaddlePaddle版本号。

PaddlePaddle每次发新的版本，遵循以下流程:

1. 从`develop`分支派生出新的分支，分支名为`release/版本号`。例如，`release/0.10.0`
1. 将新分支的版本打上tag，tag为`版本号rc.Patch号`。第一个tag为`0.10.0rc1`，第二个为`0.10.0rc2`，依次类推。
1. 对这个版本的提交，做如下几个操作:
  * 使用Regression Test List作为检查列表，测试本次release的正确性。
	  * 如果失败，记录下所有失败的例子，在这个`release/版本号`分支中，修复所有bug后，Patch号加一，到第二步
	* 修改`python/setup.py.in`中的版本信息,并将`istaged`字段设为`True`。
	* 将这个版本的python wheel包发布到pypi。
	* 更新Docker镜像（参考后面的操作细节）。
1. 第三步完成后，将`release/版本号`分支合入master分支，将master分支的合入commit打上tag，tag为`版本号`。同时再将`master`分支合入`develop`分支。
1. 协同完成Release Note的书写。

需要注意的是:

* `release/版本号`分支一旦建立，一般不允许再从`develop`分支合入`release/版本号`。这样保证`release/版本号`分支功能的封闭，方便测试人员测试PaddlePaddle的行为。
* 在`release/版本号`分支存在的时候，如果有bugfix的行为，需要将bugfix的分支同时merge到`master`, `develop`和`release/版本号`这三个分支。

## 发布wheel包到pypi

1. 使用[PaddlePaddle CI](https://paddleci.ngrok.io/project.html?projectId=Manylinux1&tab=projectOverview)
完成自动化二进制编译，参考下图，选择需要发布的版本（通常包含一个CPU版本和一个GPU版本），点击"run"右侧的"..."按钮，可以
弹出下面的选择框，在第二个tab (Changes)里选择需要发布的分支，这里选择0.11.0，然后点击"Run Build"按钮。
	<img src="https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/doc/fluid/images/ci_build_whl.png">
1. 等待编译完成后可以在此页面的"Artifacts"下拉框中找到生成的3个二进制文件，分别对应CAPI，`cp27m`和`cp27mu`的版本。
1. 由于pypi.python.org目前遵循[严格的命名规范PEP 513](https://www.python.org/dev/peps/pep-0513)，在使用twine上传之前，需要重命名wheel包中platform相关的后缀，比如将`linux_x86_64`修改成`manylinux1_x86_64`。
1. 上传：
```
cd build/python
pip install twine
twine upload dist/[package to upload]
```

* 注：CI环境使用 https://github.com/PaddlePaddle/buildtools 这里的DockerImage作为编译环境以支持更多的Linux
  发型版，如果需要手动编译，也可以使用这些镜像。这些镜像也可以从 https://hub.docker.com/r/paddlepaddle/paddle_manylinux_devel/tags/ 下载得到。
* pypi不支持覆盖上传，所以一个版本号的wheel包发布之后，不可以更改。下一个wheel包需要更新版本号才可以上传。

## 发布Docker镜像

上述PaddlePaddle CI编译wheel完成后会自动将Docker镜像push到DockerHub，所以，发布Docker镜像只需要对自动push的镜像打上
版本号对应的tag即可：

```
docker pull [镜像]:latest
docker tag [镜像]:latest [镜像]:[version]
docker push [镜像]:[version]
```

需要更新的镜像tag包括：

* `[version]`: CPU版本
* `[version]-openblas`: openblas版本
* `[version]-gpu`: GPU版本（CUDA 8.0 cudnn 5）
* `[version]-gpu-[cudaver]-[cudnnver]`: 不同cuda, cudnn版本的镜像

之后可进入 https://hub.docker.com/r/paddlepaddle/paddle/tags/ 查看是否发布成功。

## PaddlePaddle 分支规范

PaddlePaddle开发过程使用[git-flow](http://nvie.com/posts/a-successful-git-branching-model/)分支规范，并适应github的特性做了一些区别。

* PaddlePaddle的主版本库遵循[git-flow](http://nvie.com/posts/a-successful-git-branching-model/)分支规范。其中:
	* `master`分支为稳定(stable branch)版本分支。每一个`master`分支的版本都是经过单元测试和回归测试的版本。
	* `develop`分支为开发(develop branch)版本分支。每一个`develop`分支的版本都经过单元测试，但并没有经过回归测试。
	* `release/版本号`分支为每一次Release时建立的临时分支。在这个阶段的代码正在经历回归测试。

* 其他用户的fork版本库并不需要严格遵守[git-flow](http://nvie.com/posts/a-successful-git-branching-model/)分支规范，但所有fork的版本库的所有分支都相当于特性分支。
	* 建议，开发者fork的版本库使用`develop`分支同步主版本库的`develop`分支
	* 建议，开发者fork的版本库中，再基于`develop`版本fork出自己的功能分支。
	* 当功能分支开发完毕后，向PaddlePaddle的主版本库提交`Pull Reuqest`，进而进行代码评审。
		* 在评审过程中，开发者修改自己的代码，可以继续在自己的功能分支提交代码。

* BugFix分支也是在开发者自己的fork版本库维护，与功能分支不同的是，BugFix分支需要分别给主版本库的`master`、`develop`与可能有的`release/版本号`分支，同时提起`Pull Request`。

## PaddlePaddle回归测试列表

本列表说明PaddlePaddle发版之前需要测试的功能点。

### PaddlePaddle Book中所有章节

PaddlePaddle每次发版本首先要保证PaddlePaddle Book中所有章节功能的正确性。功能的正确性包括验证PaddlePaddle目前的`paddle_trainer`训练和纯使用`Python`训练（V2和Fluid）模型正确性。

<table>
<thead>
<tr>
<th></th>
<th>新手入门章节 </th>
<th> 识别数字</th>
<th> 图像分类</th>
<th>词向量</th>
<th> 情感分析</th>
<th>语意角色标注</th>
<th> 机器翻译</th>
<th>个性化推荐</th>
</tr>
</thead>

<tbody>
<tr>
<td>API.V2 + Docker + GPU </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
</tr>

<tr>
<td> API.V2 + Docker + CPU </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
</tr>

<tr>
<td>`paddle_trainer` + Docker + GPU </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
</tr>

<tr>
<td>`paddle_trainer` + Docker + CPU </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
</tr>

<tr>
<td> API.V2 + Ubuntu + GPU</td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
</tr>

<tr>
<td>API.V2 + Ubuntu + CPU </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
</tr>

<tr>
<td> `paddle_trainer` + Ubuntu + GPU</td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
</tr>

<tr>
<td> `paddle_trainer` + Ubuntu + CPU</td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
</tr>
</tbody>
</table>
