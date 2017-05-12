# Desgin Doc的一点总结
## 前言
故事还要从前两天我提交的[FileManager](https://github.com/PaddlePaddle/Paddle/pull/2013)的Design Doc PR开始。

[FileManager](https://github.com/PaddlePaddle/Paddle/pull/2013)这个文档我肯定是用心写了，我也清楚地记得，提交之前也仔细的检查了(这点自觉性还是有的)，结果，我突然发现，我的PR被Comments刷屏了；这还是其次，更要命的是，当网络速度稍慢的时候会提示我：Comments 太多啦！页面打不开！！哦。简直有点人间惨剧的味道！我要把这些和之前的Comments以及一些典型的Comments总结一下，避免同样的问题，同时也希望对您有用。
link
我觉得里边有几个基本的原则：
  
- 做事情要精益求精  
  这个是做事情的基础，也是我们讨论的基础。美，多是相似的，丑才是千奇百怪。
  
  精益求精是一种态度，态度比什么都重要。<sup>[yi](#yi)</sup>
- 节约别人的时间  
  同时也是给自己节约时间
  
- 有礼貌，有感恩的心
  > 我发现诸多之前提了comment，但是没有改，也没有回复。这个不太好吧。<sup>[ying](#ying)</sup>   
  > 每个comment都必须回复。这是开源社区的基本礼貌。别人帮了忙，应该说谢谢。<sup>[yi](#yi)</sup>

我的理解，写Doc精益求精要从基础开始，首先要规范；其次要有提纲挈领的东西，让人一眼基本明白要做的事情；然后，讲述事情结构要清晰，要分层，用最小化涉及的原则，该讲的要解释，不该讲的不讲；接下来逻辑要清晰，要流畅，不要太突兀。。。当然还有很多，比如文笔要好！：）

锻炼一下，中国人办事也不一定那么糙的，对吧！ ：）

## 基础规范

- 语言选择：中文 or 英文？  
  最好用英文，因为外国同学无法阅读中文。但是，语言是用来交流的，如果您认为英文无法完善的表达自己的意思的时候，还是用中文吧！
  
  重要的是，不管哪种文，***英文的拼写要对，中文要没有错别字***！
  
  如果文章用的是中文，那么请用中文的标点符号。反之，用英文的标点符号。***而且要保持统一、而且要保持完整***。<sup>[wuyi](#wuyi)</sup>
  
  例如这个[link](https://github.com/PaddlePaddle/Paddle/pull/2013#discussion_r114951817)
  
  还有这个[link](https://github.com/PaddlePaddle/Paddle/pull/2013#discussion_r115093563)
  
  (请原谅我多加了一个link做锚文本，Sphinx有一个bug:用中文的会报错。下边的处理类似。)

- 缩写的全称   
  一个缩写第一次出现的时候，要把他们的未缩写形式写出来。如果该单词代表的概念比较复杂，请在术语解释中把他写清楚！
  
  例如这个[link](https://github.com/PaddlePaddle/Paddle/pull/2013#discussion_r115093329)  
  
- 英语写法错误   
  <sup>[yi](#yi)</sup>总结了一下在code review的时候经常见的一些英语写法错误： Python写成python，TensorFlow写成Tensorflow，Travis CI 写成 Travis-CI，ReLU写成Relu，unit test写成unittest，MD5写成md5。

  大小写简单的规则如下：  
  - 英文的缩写是要用大写的。  
    例如这个[link](https://github.com/PaddlePaddle/Paddle/pull/2013#discussion_r115091985)
  - 英文的句子的首字母是要大写的。
  - 一个专有的名词第一个字母一般是要大写的。
 
  yiwang推荐了一个工具：[grammer](https://www.grammarly.com/)，可以作为插件安装到Chrome中，自动检查拼写和语法问题。
  
- 不要提交没有用的文件  
  例如这个 [link](https://github.com/PaddlePaddle/Paddle/pull/1964#discussion_r114414822)
  
- 提交的代码要经过验证  
  提交的代码都没有验证过是几个意思？
  
- 参考资料要设置超链，链接到对方  
  不设置的话俗称Copy。可以用`<sup>[name](#name)</sup>`设置MarkDown文件中的引用。一如此文中很多地方出现的那样。
  
- 给别人的东西步骤是要可以执行的  
  看这个[link](https://github.com/wangkuiyi/ipynb)是如何写步骤的，或者这个[link](https://github.com/PaddlePaddle/Paddle/pull/1602#issuecomment-285964510)（虽然只是一个Comment）。
  
- 链接可以直接跳到精确的位置  
  如果能一步到位就不要让别人走两步，节约大家彼此的时间。

## 提纲挈领
### 目标要明确
一般开头的一段话就要用简短的几句话把文档要描述的目标写清楚，别人看了之后，心中有了一个大概：这个文档准备讲什么。

例如这个[link](https://github.com/PaddlePaddle/Paddle/tree/develop/doc/design/cluster_train#objective)
### 架构图
一个系统或者模块的设计文档，首先要有架构图。***要有架构图，要有架构图，要有架构图***。。。。这句话可以多重复几遍。俗称，无图无真相或者一图胜千言。最起码有一个架构图说明各个模块的关系。图的表现能力远远超过了文字，特别是模块关系相关的和流程相关的场景下。

可以看一下这个文档中的图 [link](https://github.com/PaddlePaddle/Paddle/tree/develop/doc/design/cluster_train)

顺便提一句，里边的图都是用[OmniGraffle](https://www.omnigroup.com/omnigraffle)画的，就是贵了点：$99。“99你买了不吃亏，99你买了不上当。”<sup>[wuyi](#wuyi)</sup>

### 层次结构清晰
代码层面上，我们为了开发大的项目，很自然的把代码分模块、分文件。文档也是如此。每个文档或者章节说明他应该要想说的事情，尽量的减少涉及的范围。涉及的范围越少，需要阐述、解释的东西就越少，理解起来需要的背景知识就越少，就越好理解，写的人出错的概率也越少。。。

- 整体分层
  - 系统设计
  - 模块设计
  - 接口设计
  - 部署
  
相互之间需要尽量少的“越权”。例如这个[link](https://github.com/PaddlePaddle/Paddle/pull/2013#discussion_r115147388)

另外一个容易忽视的问题是，文档内部的层次结构。MarkDown文件不像Doc一样可以自动生成目录页，一个部分如果太多，就会让看得人失去层次感。以这个文档为例 [link](https://github.com/PaddlePaddle/Paddle/tree/develop/doc/design/cluster_train)。这个文档我也写过，只是把`Fault Recovery`的部分和前边正常的`Training Job`合到一起去了，结果发现越写越乱，后来看到Helin写的分层之后的文档，感觉流畅多了。

## 逻辑要清晰、流畅
- 概念的出现不要突兀。  
  文档的最前边有一个术语的解释，把文档中提到的概念先解释一下，这样，别人在看到那些概念的时候不会觉得很突兀。同时，前后要呼应。  
  
  例如这个[link](https://github.com/PaddlePaddle/Paddle/pull/2013#discussion_r114952115)
  
  如果概念或者名词后边没有出现，该删除还是删除了吧！
  
- 多种方案的选择要简述原因。  
  例如这个[link](https://github.com/PaddlePaddle/Paddle/pull/2013#discussion_r115147115)
  
- Design doc中不应该有“？” <sup>[wuyi](#wuyi)</sup>   
  应该都是陈述句的描述。有不确定的问题可以提Issue来讨论获得结论。
  对于自己不确定的地方，与其含混而过不如找人讨论先搞一个靠谱的可以讨论的。绝大多数情况下，这种含混而过的都会被Reivew给纠出来。即便就不出来，岂不是自己给自己埋一个坑？
  
- 文档当中不要黏贴大量的代码  
  代码一般都是细节，改变的非常的快，文档可能很快就失效了，需要重新的修正。另外，最主要的是大段的代码会让人的思路中断，陷于实现的细节当中。
  
- 不准备实现的就不要写了  
  最多放到放到`Future`中展望一下。

## 文笔要好
啊呀，不想当作家的程序员不是好程序员。这个当然比较难，要看大家的“学好数理化，走遍天下都不怕”的基本功的“深厚”程度了：）

顺便推荐一下公众号：老万故事会[link](https://freewechat.com/profile/MzI1MDQ3NTAxOQ==)，一个文章和代码写的一样好的人<sup>[yi](#yi)</sup>。

## 如何提高写文档的效率
这段其实本来没想到加的，开完组会之后听老大讲提高工作效率的事情有点感想，就写了。

我不是很怀疑自己写代码的效率，但是严重怀疑自己写文档的效率。感觉写文档比写代码烧脑多了。现在想想，最主要的点在于文档的结构和分层问题。

提高效率最好的办法是什么？是确定范围，很多东西不用讲了，当然效率就提高上去了。

写系统设计文档，主要需要表现模块间的关系和通信，特别是特定场景下的他们是如何配合的。这个过程中需要把关键的概念说清楚，例如这个[link](https://github.com/PaddlePaddle/Paddle/tree/develop/doc/design/cluster_train)中，`Trainjob`和`Fault Recovery`主要是讲模块间关系和配合的，[`task`](https://github.com/PaddlePaddle/Paddle/tree/develop/doc/design/cluster_train#task)作为一个关键的概念讲了。其他的部分，稍显细节的都可以放到模块设计中去讲。

另外一个，认真≠ 犹豫，也≠ 纠结。

如果感到了纠结，那说明没找到问题的根本。我在写文件上传的时候对做不做缓存优化纠结了很长时间，请教了Helin一会就讨论完毕了。如果感到了纠结，那是需要跟别人请教。不纠结的地方，下决断要果断。

## 参考
- <a name=yi>WangYi</a>
- <a name=WuYi>WuYi</a>
- <a name=Helin>Helin</a>
- <a name=YanXu>YanXu</a>
- <a name=CaoYing>CaoYing</a>
