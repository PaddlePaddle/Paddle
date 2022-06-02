[33mcommit 1bfbcfaf5ad9e737b19556cdbfa2b068ec9439b2[m[33m ([m[1;36mHEAD -> [m[1;32mscatter_mlu_lfy[m[33m)[m
Author: Fan Zhang <frank08081993@gmail.com>
Date:   Thu Jun 2 15:54:10 2022 +0800

    [XPUPS] modify BKCL comm op register (#43028)
    
    * Adapt XPUPS - 1st version - 3.24
    
    * Adapt XPUPS - update XPU PushSparse -  2nd version - 3.24
    
    * Adapt XPUPS - add XPU PullSparseOp - 3nd version - 3.25
    
    * refactor heter comm kernel
    
    * update. test=develop
    
    * Adapt XPUPS - modify by compilation - 4th version - 3.27
    
    * update calc_shard_offset. test=develop
    
    * update xpu kernel. test=develop
    
    * update args of calc_shard_offset
    
    * update. test=develop
    
    * remove customGradMerger
    
    * update. test=develop
    
    * heter_comm update
    
    * heter_comm update
    
    * update calc_shard_offset. test=develop
    
    * heter_comm update
    
    * update args of calc_shard_offset
    
    * update. test=develop
    
    * remove customGradMerger
    
    * update. test=develop
    
    * fix. test=develop
    
    * update. test=develop
    
    * update. test=develop
    
    * update optimizer kernel
    
    * Adapt XPUPS - use WITH_XPU_KP and modify wrapper kernel function - 5th version - 3.30
    
    * update. test=develop
    
    * update pslib.cmake
    
    * update. test=develop
    
    * update. test=develop
    
    * update. test=develop
    
    * update. test=develop
    
    * update. test=develop
    
    * Adapt XPUPS - modify by kp compilation  - 6th version - 3.30
    
    * update. test=develop
    
    * update. test=develop
    
    * update. test=develop
    
    * update optimizer kernel
    
    * update. test=develop
    
    * update. test=develop
    
    * update. test=develop
    
    * update. test=develop
    
    * update. test=develop
    
    * update. test=develop
    
    * update. test=develop
    
    * update. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * used by minxu
    
    * update heter_comm_inl
    
    * fix. test=develop
    
    * Adapt XPUPS - modify by kp compilation  - 7th version - 3.30
    
    * fix. test=develop
    
    * add optimizer kernel. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * 3.31 update
    
    * Adapt XPUPS - update kp compilation path  - 8th version - 3.31
    
    * add optimizer kernel. test=develop
    
    * fix kunlun not support size_t. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix kunlun not support size_t. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * update heter_comm_kernel.kps 3.31
    
    * fix. test=develop
    
    * fix. test=develop
    
    * update heter_comm_kernel.kps 3.31
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * update heter_comm.h 3.31
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * update hashtable. test=develop
    
    * update. test=develop
    
    * Adapt XPUPS - update by kp compilation  - 9th version - 4.1
    
    * update hashtable. test=develop
    
    * fix. test=develop
    
    * update hashtable 4.1
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * Adapt XPUPS - update by kp compilation  - 10th version - 4.1
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * update. test=develop
    
    * modify by compilation 4.1
    
    * update. test=develop
    
    * update. test=develop
    
    * fix. test=develop
    
    * modify by compilation 4.1
    
    * update. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * modify by compilation 4.1
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * modify by compilation 4.1 19:30
    
    * fix. test=develop
    
    * update ps_gpu_wrapper.kps 4.1
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * Adapt XPUPS - update by kp compilation  - 11th version - 4.1
    
    * fix. test=develop
    
    * Adapt XPUPS - update by kp compilation  - 12nd version - 4.2
    
    * fix. test=develop
    
    * fix. test=develop
    
    * modify by compilation 4.2
    
    * 4.2 update
    
    * fix. test=develop
    
    * template init. test=develop
    
    * update 4.6
    
    * fix. test=develop
    
    * template init. test=develop
    
    * 4.6 modify by compilation
    
    * hashtable template init. test=develop
    
    * hashtable template init. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=devlop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=devlop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * Adapt XPUPS - update by kp compilation  - 13nd version - 4.7
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * 4.11 update
    
    * fix. test=develop
    
    * fix. test=develop
    
    * 4.11 update
    
    * update by pre-commit
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * 4.12 update
    
    * fix. test=develop
    
    * Adapt XPUPS - update by kp compilation  - 14th version - 4.13
    
    * 4.13 update
    
    * 4.14 update
    
    * 4.14 update
    
    * 4.14 update
    
    * 4.14 modify by merged latest compilation
    
    * retry CI 4.14
    
    * 4.15 pass static check
    
    * 4.15 modify by gpups CI
    
    * 3.16 update by gpups CI - modify ps_gpu_wrapper.h
    
    * 4.16 update
    
    * 4.16 pass xpu compile
    
    * 4.16 retry CI
    
    * 4.16 update
    
    * Adapt XPUPS - adapt BKCL comm for XPUPS - 4.24
    
    * update by compilation
    
    * Adapt XPUPS - register PSGPUTrainer for XPUPS - 4.25
    
    * update device_worker_factory
    
    * Adapt XPUPS - split heter_ps into .cu and .cc - 4.27
    
    * Adapt XPUPS - register pull_box_sparse op under XPU_KP - 4.28
    
    * update
    
    * 5.7 modify ps_gpu_wrapper pull_sparse
    
    * 5.11 update ps_gpu_wrapper CopyKeysKernel
    
    * 5.13 modify calc_shard_offset_kernel & fill_shard_key_kernel
    
    * modify fill_dvals_kernel & PullCopy & c_sync_calc_stream - 5.18
    
    * modify PushCopy & fill_shard_grads_kernel & register push_box_sparse - 5.19
    
    * Adapt XPUPS - modify BKCL comm op register - 5.26
    
    * Adapt XPUPS - modify BKCL comm op register - 5.27
    
    * Adapt XPUPS - modify BKCL comm op register - 5.27v2
    
    * Adapt XPUPS - modify BKCL comm op register - 5.27v3
    
    * Adapt XPUPS - modify c_comm_init_all_op to adapt BKCL init - 5.30
    
    * Adapt XPUPS - modify c_comm_init_all_op to adapt BKCL init v2 - 5.30
    
    * Adapt XPUPS - modify c_comm_init_all_op to adapt BKCL init v3 - 5.30
    
    * Adapt XPUPS - modify c_comm_init_all_op to adapt BKCL init v4 - 5.31
    
    Co-authored-by: zmxdream <zhangminxu01@baidu.com>

[33mcommit 030b23da86cfd26e072642e27ee50c8a23544b91[m
Author: Tomasz Socha <tomasz.socha@intel.com>
Date:   Thu Jun 2 09:35:10 2022 +0200

    Fix for Bfloat16 placement pass. (#43109)
    
    * Fix bfloat16 placement pass
    
    * Make it nicer
    
    * Fix leftovers
    
    * Style

[33mcommit 990c5e7f15da02d0484359e811305f6e8e0dd682[m
Author: Zhang Zheng <32410583+ZzSean@users.noreply.github.com>
Date:   Thu Jun 2 14:36:00 2022 +0800

    Support head_dim = 96 in fused_multi_transformer for PLATO-XL (#43120)
    
    * Support head_dim = 96 in fused_multi_transformer in PLATO-XL
    
    * add notes

[33mcommit 041000c2001c106c1cd571da2663577b6f82f429[m
Author: Lux et Veritas <1004239791@qq.com>
Date:   Thu Jun 2 14:17:46 2022 +0800

    [MLU]add mlu kernel for squeeze and squeeze2 (#43094)
    
    Co-authored-by: liupeiyu <liupeiyu@cambricon.com>

[33mcommit fe911a512775f348778c57b3b6d0168046b5e294[m
Author: Chenxiao Niu <ncxinhanzhong@gmail.com>
Date:   Thu Jun 2 14:17:36 2022 +0800

    add concat_grad mlu kernel. (#43117)

[33mcommit d999049f6cb66aa7a7a7c80a577a374e680f0481[m
Author: ziyoujiyi <73728031+ziyoujiyi@users.noreply.github.com>
Date:   Thu Jun 2 12:46:17 2022 +0800

    add federated learning parameter server(fl-ps) mode (#42682)
    
    * back fl
    
    * delete ssl cert
    
    * .
    
    * make warning
    
    * .
    
    * unittest paral degree
    
    * solve unittest
    
    * heter & multi cloud commm ready
    
    * .
    
    * .
    
    * fl-ps v1.0
    
    * .
    
    * support N + N mode
    
    * .
    
    * .
    
    * .
    
    * .
    
    * delete print
    
    * .
    
    * .
    
    * .
    
    * .

[33mcommit 2810dfea475e811d7a919d3ee9c5317e0f865da3[m
Author: Wangzheee <634486483@qq.com>
Date:   Thu Jun 2 12:21:17 2022 +0800

    [Paddle-Inference] new general transformer inference support (#43077)
    
    * new general transformer inference support

[33mcommit 0cb9dae580e806adff91f62af959d46ed6317307[m
Author: Zhang Zheng <32410583+ZzSean@users.noreply.github.com>
Date:   Thu Jun 2 12:05:39 2022 +0800

    Delete inplace strategy in group_norm_fwd (#43137)
    
    * Delete inplace strategy in group_norm_fwd
    
    * fix

[33mcommit 0f1be6e050e1c0c2fbc643f9458353a9991b1bc6[m
Author: wanghuancoder <wanghuan29@baidu.com>
Date:   Thu Jun 2 12:00:15 2022 +0800

    [Eager] first run accumulation node (#43134)
    
    * first run accumulation node

[33mcommit ceb2040675ff9ab9014ec18f650cac2a3468e371[m
Author: Siming Dai <908660116@qq.com>
Date:   Thu Jun 2 11:12:40 2022 +0800

    Support hetergraph reindex (#43128)
    
    * support heter reindex
    
    * add unittest, fix bug
    
    * add comment
    
    * delete empty line
    
    * refine example
    
    * fix codestyle
    
    * add disable static

[33mcommit 2bfe8b2c8584ab116c8faa5f7c6c1a09f5d024d0[m
Author: Jackwaterveg <87408988+Jackwaterveg@users.noreply.github.com>
Date:   Thu Jun 2 10:57:43 2022 +0800

    [Dataloader]Add prefetch_factor in dataloader (#43081)
    
    * fix usage of prefetch_factor
    
    * add assert
    
    * add docstring and change prefetch_factor when num_workers=0
    
    * fix doc

[33mcommit 67163fb42ccc104dbc92affc253519b678f59567[m
Author: Guoxia Wang <mingzilaochongtu@gmail.com>
Date:   Thu Jun 2 10:45:17 2022 +0800

    fix the bug of margin cross entropy loss for eager mode (#43161)

[33mcommit 85baa3c0272d6b56a5e8fb0d59d4ed4222f4abe2[m
Author: Li Min <11663212+limin2021@users.noreply.github.com>
Date:   Thu Jun 2 10:11:58 2022 +0800

    Extend forward fast layer_norm kernel to support more dimensions. (#43118)
    
    * extend forward fast_ln_kernel to support more column values.

[33mcommit 8c7cb3d6a9cd95b7f552e48202b0d778c15cb4f7[m
Author: zhaoyingli <86812880+zhaoyinglia@users.noreply.github.com>
Date:   Thu Jun 2 10:07:27 2022 +0800

    [AutoParallel] engine.prepare only once (#43093)
    
    * prepare only once

[33mcommit 7ba843e6b07693bb228d48e9701662492f33c29d[m
Author: zhaoyingli <86812880+zhaoyinglia@users.noreply.github.com>
Date:   Thu Jun 2 10:06:43 2022 +0800

    bug fix (#43153)

[33mcommit d05b940a9d1d0432bf9585c3547e8472ea0ea457[m
Author: sneaxiy <32832641+sneaxiy@users.noreply.github.com>
Date:   Thu Jun 2 09:39:44 2022 +0800

    Support CUDA Graph for partial graph in dygraph mode (#42786)
    
    * support CUDAGraph for partial graph
    
    * add ut
    
    * fix ci
    
    * fix ut again because of eager mode
    
    * fix kunlun ci
    
    * fix win ci

[33mcommit 126248acb35a0c199cd8d5057932738bdd52cdc1[m
Author: xiongkun <xiongkun03@baidu.com>
Date:   Wed Jun 1 21:46:31 2022 +0800

    fix memory leakage (#43141)

[33mcommit 56ae33b669ab4359d0aeb280c7440d8433fbd6d4[m
Author: YuanRisheng <yuanrisheng@baidu.com>
Date:   Wed Jun 1 19:58:28 2022 +0800

    Add yaml and unittest for instance_norm op (#43060)
    
    * add yaml
    
    * fix infrt compile bugs

[33mcommit b23914c26819cadf76cb47a4a7ec173ce88c3212[m
Author: Aganlengzi <aganlengzi@gmail.com>
Date:   Wed Jun 1 19:55:30 2022 +0800

    [fix] split nanmedian fluid deps (#43135)

[33mcommit ef79403e7e9aa732eb463790a64ae6db3312fe0a[m
Author: Qi Li <qili93@qq.com>
Date:   Wed Jun 1 19:31:26 2022 +0800

    Revert "[IPU] support paddle.distributed.launch with IPUs (#43087)" (#43138)
    
    This reverts commit e680d581c4ff906e84ae273d2c2b3dbee96ee9db.

[33mcommit 2dac35f3a8d2313948ef3d6b380e1ade9723633d[m
Author: Qi Li <qili93@qq.com>
Date:   Wed Jun 1 18:12:10 2022 +0800

    [NPU] fix npu runtime error of HCCLParallelContext, test=develop (#43116)

[33mcommit 1f6d25d8f6405c794a14375acc64d380d36bd0a3[m
Author: BrilliantYuKaimin <91609464+BrilliantYuKaimin@users.noreply.github.com>
Date:   Wed Jun 1 17:31:16 2022 +0800

    修复 paddle.bernoulli 英文文档 (#42912)
    
    * Update random.py
    
    * test=document_fix
    
    * test=document_fix
    
    * Update random.py

[33mcommit 67b9b51b0eb3f8e66624b11ee3600563031c0297[m
Author: Guoxia Wang <mingzilaochongtu@gmail.com>
Date:   Wed Jun 1 17:08:14 2022 +0800

    support nccl api for bfloat16, required >= cudnn 10.1, nccl >= 2.10.3 (#43147)

[33mcommit 048b00132e9e6b86d21e520319cbe13de0dc2098[m
Author: sneaxiy <32832641+sneaxiy@users.noreply.github.com>
Date:   Wed Jun 1 16:55:51 2022 +0800

    Make fuse_gemm_epilogue support transpose_x and transpose_y (#40558)
    
    * support weight transpose
    
    * add ut
    
    * add template
    
    * fix transpose error
    
    * fix transpose_comment
    
    * add api tests
    
    * add skipif
    
    * add doc

[33mcommit 07993044026e4e25e8eda0d6edb2f2306ee18025[m
Author: YUNSHEN XIE <1084314248@qq.com>
Date:   Wed Jun 1 16:55:33 2022 +0800

    remove skip ci directly when the pr is approved (#43130)

[33mcommit 13add8231a616025d3f3e556dfa034c94610a62a[m
Author: Zhou Wei <1183042833@qq.com>
Date:   Wed Jun 1 16:25:39 2022 +0800

    Unify sparse api in paddle.incubate (#43122)

[33mcommit 664758fa776405b9a2e2a6199e008f820341a621[m
Author: Sing_chan <51314274+betterpig@users.noreply.github.com>
Date:   Wed Jun 1 16:20:17 2022 +0800

    code format check upgrade step1: pre-commit, remove-ctrlf, pylint (#43103)

[33mcommit f59bcb1c781038b871154118f31658c0fff8b16a[m
Author: JZ-LIANG <jianzhongliang10@gmail.com>
Date:   Wed Jun 1 15:43:47 2022 +0800

    [AutoParallel & Science] Miscellaneous improvements  (#43139)
    
    * adapt for 10 loss
    
    * partitioner support optimizer

[33mcommit ff1789ca5eba03fe47764cdfec7791f6d149eea4[m
Author: BrilliantYuKaimin <91609464+BrilliantYuKaimin@users.noreply.github.com>
Date:   Wed Jun 1 15:19:10 2022 +0800

    Update creation.py (#42915)

[33mcommit 2aea0db8a5d014a84ef83a755f2878db979bf100[m
Author: houj04 <35131887+houj04@users.noreply.github.com>
Date:   Wed Jun 1 14:25:49 2022 +0800

    update xpu cmake: xdnn 0601 (#43051)
    
    * update xpu cmake: xdnn 0527. test=kunlun
    
    * update to xdnn 0531.
    
    * update to xdnn 0531. test=kunlun
    
    * update to xdnn 0601. test=kunlun

[33mcommit dc26d07b60414df7984e9664788b9d760d50169f[m
Author: zhangchunle <clzhang_cauc@163.com>
Date:   Wed Jun 1 14:07:21 2022 +0800

    Unittest parallel (#43042)
    
    unittest parallel
    
    Co-authored-by: zhangbo9674 <zhangbo54@baidu.com>

[33mcommit c4b7c4852e85673b2ced5f1d5ba24ae575aa1c75[m
Author: Ruibiao Chen <chenruibiao@baidu.com>
Date:   Wed Jun 1 12:22:35 2022 +0800

    Add pinned memory to host memory stats (#43096)
    
    * Add pinned memory to HostMemoryStats
    
    * Add macro for WrapStatAllocator
    
    * Fix CI errors

[33mcommit 0e10f247d609f5755e29f5a940d2e43c43fd17a6[m
Author: zhiboniu <31800336+zhiboniu@users.noreply.github.com>
Date:   Wed Jun 1 12:08:33 2022 +0800

    fluid code transfer in nn.functional (#42808)

[33mcommit 77bae9a45b4870006b1f3b12ee9ffdc319864a89[m
Author: Guoxia Wang <mingzilaochongtu@gmail.com>
Date:   Wed Jun 1 11:36:24 2022 +0800

    fix the bug of adamw which set the attribute in param group not working (#43013)
    
    * fix the bug of adamw which set the attribute in param group not working
    
    * fix undefined variable
    
    * fix api example typo
    
    * add unittest
    
    * fix unittest typo

[33mcommit 81622708a7c904092185ef04897b1e81629f51a6[m
Author: huzhiqiang <912790387@qq.com>
Date:   Wed Jun 1 10:32:03 2022 +0800

     [revert] revert inference accelarate #43125

[33mcommit bd01836016137dc9564f6c26bf4fb5c3b19ff950[m
Author: caozhou <48191911+Caozhou1995@users.noreply.github.com>
Date:   Wed Jun 1 10:22:06 2022 +0800

    add some comp op costs (#43114)

[33mcommit 010aba33ee5655555ce1e9bf92e9596828d446ae[m
Author: Yulong Ao <aoyulong@baidu.com>
Date:   Wed Jun 1 10:18:26 2022 +0800

    [Auto Parallel] Add miscellaneous improvements (#43108)
    
    * [Auto Parallel] Add the parallel tuner
    
    * [Auto Parallel] Improve the parallel tuner and fix some bugs
    
    * upodate cost model
    
    * update import Resharder by dist op
    
    * update cost model
    
    * fix comp cost bug
    
    * update cost model
    
    * [Auto Parallel] Amend the dist attr for #processses=1
    
    * update cost model and tuner
    
    * update cost model and tuner
    
    * update cost model and tuner
    
    * update cluster
    
    * update reshard
    
    * [Auto Parallel] Add the estimation from the cost model
    
    * [Auto Parallel] Reimplement the backup and restore functions
    
    * [Auto Parallel] Fix the bugs of the parallel tuner
    
    * [Auto Parallel] Update the engine api and dist context
    
    * [Auto Parallel] Work around the high order grad problem
    
    * [Auto Parallel] Add some miscellaneous improvements
    
    * [Auto Parallel] Add a unittest for DistributedContext
    
    Co-authored-by: caozhou <caozhou@radi.ac.cn>

[33mcommit 5f2c251c75b11b6bb311a68482a9bd7fe5107d83[m
Author: chentianyu03 <chentianyu03@baidu.com>
Date:   Wed Jun 1 10:03:49 2022 +0800

    [Yaml]add conv3d, depthwise_conv2d yaml (#42807)
    
    * add conv3d yaml
    
    * add conv3d_grad, conv3d_double_grad
    
    * add final_state_conv3d test case
    
    * add conv3d double test case
    
    * add depthwise_conv2d grad yaml
    
    * add depthwise_conv2d double grad test case
    
    * modify the order of args
    
    * add depthwise_conv2d_grad_grad config

[33mcommit 4b89120bf55e48cdc78ceca8c7dadcf349b14060[m
Author: Sławomir Siwek <slawomir.siwek@intel.com>
Date:   Tue May 31 16:13:41 2022 +0200

    Remove mkldnn attributes from base ops (#42852)
    
    * remove attrs from base op
    
    * fix typos
    
    * remove brelu
    
    * undo removing code related to matmul
    
    * remove whitespaces
    
    * undo changes in matmul
    
    * remove empty line

[33mcommit 941942755d2bc650360dfda1e48cd057c27ecbdc[m
Author: pangyoki <pangyoki@126.com>
Date:   Tue May 31 21:50:43 2022 +0800

    add double_grad and triple_grad inplace info in backward.yaml (#43124)
    
    * add double_grad and triple_grad inplace info in backward.yaml
    
    * only generate inplace api in forward

[33mcommit 462ae0054a7be6708d631b888523aed76f376c1a[m
Author: wanghuancoder <wanghuan29@baidu.com>
Date:   Tue May 31 21:47:39 2022 +0800

    [Eager] Fix Full Zero (#43048)
    
    * fix full zero
    
    * fix full zero
    
    * fix full zero
    
    * fix full zero
    
    * refine
    
    * refine
    
    * refine

[33mcommit d70e45bc51e607677069d9cf3cc154dac5934bdf[m
Author: Sing_chan <51314274+betterpig@users.noreply.github.com>
Date:   Tue May 31 21:34:59 2022 +0800

    put set error_code infront to avoid being skipped (#43014)

[33mcommit c9e7c407612e3746c4a218344d0b2be8916a7a6f[m
Author: Chen Weihang <chenweihang@baidu.com>
Date:   Tue May 31 18:37:45 2022 +0800

    [Phi] Polish assign kernel copy impl (#43061)
    
    * fix assign kernel copy impl
    
    * fix test failed

[33mcommit 172739d4935c727d9a20c54236ed08691e8f4d1d[m
Author: BrilliantYuKaimin <91609464+BrilliantYuKaimin@users.noreply.github.com>
Date:   Tue May 31 17:21:31 2022 +0800

    test=document_fix Verified (#42919)

[33mcommit cb195fa0c349b0592974dbb206f0f708552db943[m
Author: cambriconhsq <106155938+cambriconhsq@users.noreply.github.com>
Date:   Tue May 31 16:28:38 2022 +0800

    [MLU] add mlu kernel for abs op (#43099)

[33mcommit e680d581c4ff906e84ae273d2c2b3dbee96ee9db[m
Author: yaozhixin <zhixiny@graphcore.ai>
Date:   Tue May 31 16:25:40 2022 +0800

    [IPU] support paddle.distributed.launch with IPUs (#43087)
    
    * [IPU] support paddle.distributed.launch with IPUs
    
    * add device_num to env_args_mapping

[33mcommit 48409529b68b5767e2465222a235700ec25a367d[m
Author: David Nicolas <37790151+liyongchao911@users.noreply.github.com>
Date:   Tue May 31 15:54:55 2022 +0800

    update RandomCrop class code annotation; test=document_fix (#42428)
    
    * update RandomCrop class code annotation; test=document_fix
    
    * update adjust_brightness api in functional.py test=document_fix
    
    * udpate uniform api in random.py
    
    * update transforms.py

[33mcommit 632027d74a3199e89bde2568a6ab344777fd7be3[m
Author: BrilliantYuKaimin <91609464+BrilliantYuKaimin@users.noreply.github.com>
Date:   Tue May 31 15:54:47 2022 +0800

    test=document_fix (#42922)

[33mcommit e9589e354fd90965272bc5fed18303037179f3bc[m
Author: Chen Weihang <chenweihang@baidu.com>
Date:   Tue May 31 15:26:28 2022 +0800

    [Eager] Polish append op using for model perf (#43102)
    
    * polish append op using
    
    * fix var error
    
    * fix group norm impl

[33mcommit f9e55dee9cc1c7cac70bd87200d228aec931deea[m
Author: Aganlengzi <aganlengzi@gmail.com>
Date:   Tue May 31 14:52:36 2022 +0800

    [NPU] fix arg_max and reduce_max (#42887)
    
    * fix arg_max and reduce_max
    
    * add arg_max ut

[33mcommit 21e1d10f26b5e58139a75c2da067446fb4425e68[m
Author: thunder95 <290844930@qq.com>
Date:   Tue May 31 14:44:32 2022 +0800

    【PaddlePaddle Hackathon 2】16 新增 API RRelu (#41823)
    
    * rrelu逻辑部分
    
    * unregistered op kernel (unresolved)
    
    * commit before merge
    
    * 丰富测试用例
    
    * 修复rrelu-sig的bug
    
    * 修复cpu环境测试
    
    * 修改拼写错误
    
    * 修改code format
    
    * 尝试优化测试用例timeout的问题
    
    * 优化测试用例
    
    * 移除seed, 优化随机函数
    
    * update en doc for rrelu
    
    * fix rrelu en docs, test=document_fix
    
    * add paper link for en docs, test=document_fix
    
    * udpate en doc
    
    * add r,test=document_fix

[33mcommit 6319dd830f5bfb1ab57a0584176ac83132f6b20a[m
Author: Haohongxiang <86215757+haohongxiang@users.noreply.github.com>
Date:   Tue May 31 14:37:05 2022 +0800

    fix bugs (#43115)

[33mcommit a4bb38cbb8b64bb36a40fd68b035c41adf20076f[m
Author: xiongkun <xiongkun03@baidu.com>
Date:   Tue May 31 14:35:30 2022 +0800

    [EinsumOp] Make EinsumOp support bfloat16. (#43085)
    
    * change einsum_v2 as default and add new flags: FLAG_einsum_opt=1|0
    
    * make EInsumOP support bf16
    
    * add unittest for BF16
    
    * add condition for test_BF16
    
    * fix bugs
    
    * fix

[33mcommit 0ae8a2d67623f33c13f2dc14141587619cc3ba7e[m
Author: Leo Chen <39020268+leo0519@users.noreply.github.com>
Date:   Tue May 31 14:21:14 2022 +0800

    Fix the underflow of fp16 fake quantize operators (#43088)
    
    Co-authored-by: Ryan Jeng <rjeng@nvidia.com>

[33mcommit 4700a08e99d232d2597a135ec655252f4a29cdd6[m
Author: Jiabin Yang <360788950@qq.com>
Date:   Tue May 31 13:41:44 2022 +0800

    Support backward prune for eager intermidiate (#43111)
    
    * support is empty
    
    * fix error
    
    * fix code error
    
    * change to fake empty
    
    * using fake empty first
    
    * using fake empty first
    
    * Support backward prune in fluid

[33mcommit 6749711976817c1df3d57733b8699a5b6855e933[m
Author: Li Min <11663212+limin2021@users.noreply.github.com>
Date:   Tue May 31 12:44:48 2022 +0800

    Rename dropout is test (#43098)
    
    * replace dropout_is_test with is_test.
    * improve atol on a100.

[33mcommit ae45d981181b44783c61a21d808b54cc5148dc02[m
Author: Weilong Wu <veyron_wu@163.com>
Date:   Tue May 31 12:01:09 2022 +0800

    [Eager] fix collective_global_gather (#43090)
    
    * [Eager] fix collective_global_gather
    
    * fix eager_ode = 1

[33mcommit 2785f8762ed24316b71e9ae0dab4a639b01b19fe[m
Author: zyfncg <zhangyunfei07@baidu.com>
Date:   Tue May 31 11:40:02 2022 +0800

    add embedding yaml (#43029)
    
    * add embedding yaml
    
    * fix infermeta bug
    
    * fix bug of selected_rows infer_meta
    
    * fix selected_rows
    
    * add unittest

[33mcommit b779d2b8bb2dbe17987f7c490c487f3a430ea582[m
Author: Wilber <jiweibo@baidu.com>
Date:   Tue May 31 11:27:12 2022 +0800

    fix slice plugin (#43110)

[33mcommit 12d8a567b5bfecd284ff856f7471699ed3da0af7[m
Author: jakpiase <jakpia21@gmail.com>
Date:   Mon May 30 19:25:19 2022 +0200

    OneDNN md-in-tensor refactoring part 5: Memory descriptor enabled for elementwises, reductions and expand_v2 ops (#43036)
    
    * enabled md in elementwises, reductions and expand_v2
    
    * CI fix for invalid numpy copy
    
    * fixed formatting
    
    * CI rerun
    
    * changes after review

[33mcommit 13a21cf7a45f4b740b010b57b309fee5357ff32b[m
Author: Chenxiao Niu <ncx_bupt@163.com>
Date:   Mon May 30 22:36:02 2022 +0800

    [mlu] add one_hot_v2 mlu kernel (#43025)

[33mcommit dceccd9d1b9ccc8e0f352932401f18864dc49f47[m
Author: Li Min <11663212+limin2021@users.noreply.github.com>
Date:   Mon May 30 22:02:21 2022 +0800

    Add fused_bias_dropout_residual_ln op and layer. (#43062)
    
    * add fused_bias_dropout_residual_ln op and layer.

[33mcommit e1e0deed64bd879357b9fc28ff68770f8eae87a6[m
Author: heliqi <1101791222@qq.com>
Date:   Mon May 30 08:48:10 2022 -0500

    fix scale_matmul fuse pass (#43089)

[33mcommit 1448520d45d18c7272332f1d10247ab1c287b234[m
Author: shentanyue <34421038+shentanyue@users.noreply.github.com>
Date:   Mon May 30 21:39:23 2022 +0800

    [TensorRT] Fix delete fill_constant pass (#43053)
    
    * update lite compile cmake
    
    * Update delete_fill_constant_op_pass.cc
    
    * Update analysis_config.cc

[33mcommit ed2886de81de7fd4457a6e69bed435212c15404d[m
Author: pangyoki <pangyoki@126.com>
Date:   Mon May 30 20:36:01 2022 +0800

    support backward inplace in eager fluid dygraph mode (#43054)
    
    * support backward inplace in eager fluid mode
    
    * fix
    
    * fix
    
    * optimize format
    
    * little change

[33mcommit 3d56d41918f2d58e0dcb190b450318228b04afcb[m
Author: pangyoki <pangyoki@126.com>
Date:   Mon May 30 19:08:09 2022 +0800

    add backward inplace api (#42965)

[33mcommit fdcdbec5330efbe850d648f9444d60ce7881f4dc[m
Author: crystal <62974595+Zjq9409@users.noreply.github.com>
Date:   Mon May 30 17:56:51 2022 +0800

    Implement fused_gate_attention operator for AlphaFold. (#42018)

[33mcommit 17b8446d459bc3ddde7eee71d04e5ed4c986fbc5[m
Author: zhaoyingli <86812880+zhaoyinglia@users.noreply.github.com>
Date:   Mon May 30 17:46:12 2022 +0800

    [AutoParallel] use original id in grad_op_id_to_op_id (#42992)
    
    * use original id in dist_op_context.grad_op_id_to_op_id
    
    * del assert
    
    * remove redundant map

[33mcommit f87fa3c0e5d0ebf89b336cf16c4d1eb0b8767b25[m
Author: thunder95 <290844930@qq.com>
Date:   Mon May 30 16:38:45 2022 +0800

    【PaddlePaddle Hackathon 2】15 新增 API Nanmedian (#42385)
    
    * nanmedian op
    
    * 修改cuda kernel的bug
    
    * 修复count_if在其他硬件平台不兼容
    
    * 修复某些cpu硬件不兼容
    
    * 修复某些cpu硬件不兼容
    
    * 修复isnan判断
    
    * 兼容numpy低版本不支持全部nan的情况
    
    * 兼容numpy低版本不支持全部nan的情况
    
    * fix code example
    
    * fix api comment error
    
    * 修改反向传播逻辑以及c++处理逻辑
    
    * 完成修改建议
    
    * typo pre_dim
    
    * update en docs, test=document_fix
    
    * remove numpy in en doc, test=document_fix
    
    * add r,test=document_fix
    
    * 添加api到all
    
    * follow advice from chenwhql

[33mcommit 5df922621017f1983d11e76808b8e962d6f1b96d[m
Author: huzhiqiang <912790387@qq.com>
Date:   Mon May 30 16:02:46 2022 +0800

    [Framework]accelerate inference period (#42400)

[33mcommit 8cc40f4702c5cf0e8c88b13e17d8461938f7298a[m
Author: levi131 <83750468+levi131@users.noreply.github.com>
Date:   Mon May 30 15:59:08 2022 +0800

    enhance check for current block and docstring for prim2orig interface (#43063)
    
    * enhance check for current block docstring for prim2orig interface
    
    * refine if else syntax

[33mcommit 586f9429bb3a9086a0f66279c5883b27fb31f293[m
Author: cambriconhsq <106155938+cambriconhsq@users.noreply.github.com>
Date:   Mon May 30 15:48:46 2022 +0800

    [MLU]add mlu kernel for log_softmax op (#43040)

[33mcommit 2d6dd55f8148ceb8c136b0a8d18d4f50713667e1[m[33m ([m[1;31morigin/develop[m[33m, [m[1;31morigin/HEAD[m[33m, [m[1;32mdevelop[m[33m)[m
Author: tianshuo78520a <707759223@qq.com>
Date:   Mon May 30 15:39:54 2022 +0800

    Update Coverage docker (#43078)

[33mcommit 4b9e9949e24b54d68d360f425707237cb428029e[m
Author: Qi Li <qili93@qq.com>
Date:   Mon May 30 14:23:55 2022 +0800

    fix build error on Sunway, test=develop (#43071)

[33mcommit 806073d6b765a15cd14cab31521973c7cf8456d6[m
Author: limingshu <61349199+JamesLim-sy@users.noreply.github.com>
Date:   Mon May 30 14:23:40 2022 +0800

    Optimize memcpy operation in Eigh  (#42853)
    
    * 1st commit
    
    * fix usless change in header transpose_kernel_h file
    
    * add sync

[33mcommit 3591a2528038b17b90390ea7bdb8c0e5eabee7d9[m
Author: Sing_chan <51314274+betterpig@users.noreply.github.com>
Date:   Mon May 30 14:16:01 2022 +0800

    cant just exit, because the new api has no doc in develop;test=document_fix (#43083)

[33mcommit cd3d0911038355bdba8a5533960cc99400ae16ee[m
Author: WangZhen <23097963+0x45f@users.noreply.github.com>
Date:   Mon May 30 12:09:27 2022 +0800

    [Dy2St]Fix cond_block_grad error when handle no need grad vras (#43034)
    
    * Fix cond_block_grad error when handle no need grad vras
    
    * Add comment and UT

[33mcommit 849d937b9863e97cc72002284d94099981a9c752[m
Author: Aganlengzi <aganlengzi@gmail.com>
Date:   Mon May 30 11:43:28 2022 +0800

    [fix] addmm supports 1-d input (#42959)
    
    * addmm supports 1-d input
    
    * fix coverage
    
    * fix
    
    * more ut

[33mcommit 114a5d214977507c20c2b8f770301e3187f3ab04[m
Author: zyfncg <zhangyunfei07@baidu.com>
Date:   Mon May 30 10:39:37 2022 +0800

    Make data transform inplaced when tensor is on GPUPinned (#43055)
    
    * make data transform inplace when tensor is on gpupinned in new dygraph
    
    * fix unittest

[33mcommit 4fd334f5cc501d5ef92003d48a3a0b23d5cef33e[m
Author: tianshuo78520a <707759223@qq.com>
Date:   Mon May 30 10:38:07 2022 +0800

    CI check Coverage build size (#42145)

[33mcommit a1d87776ac500b1a3c3250dd9897f103515909c6[m
Author: zhangchunle <clzhang_cauc@163.com>
Date:   Mon May 30 10:30:24 2022 +0800

    rm serial mode in exclusive case (#43073)

[33mcommit 8cc2e28c7ed3c4826de9c82f60368b06bd111918[m
Author: ShenLiang <1422485404@qq.com>
Date:   Sat May 28 16:32:41 2022 +0800

    [Bug Fix]Fix global_scatter/global_gather in ProcessGroup (#43027)
    
    * fix alltoall
    
    * rename utest

[33mcommit 9eb18c75a39816c91d8456ae455fe403ac62d451[m
Author: Jiabin Yang <360788950@qq.com>
Date:   Fri May 27 21:39:22 2022 +0800

    [Eager] Support is empty (#43032)
    
    * support is empty
    
    * fix error
    
    * fix code error
    
    * change to fake empty
    
    * using fake empty first
    
    * using fake empty first

[33mcommit 4d32f417a435446d06541ae951edc2404e97e74c[m
Author: Weilong Wu <veyron_wu@163.com>
Date:   Fri May 27 21:36:35 2022 +0800

    [Eager] Support EagerParamBase init by 'shape'(Tensor) (#43045)

[33mcommit 6d78524c27732fdc4f3505815d392d8f24b2dca8[m
Author: zyfncg <zhangyunfei07@baidu.com>
Date:   Fri May 27 20:47:18 2022 +0800

    [Phi] Change optional tensor from `optional<const Tensor&>` to `optional<Tensor>` (#42939)
    
    * refactor the optional tensor
    
    * remove optiona<MetaTensor> in InferMeta
    
    * fix bug
    
    * fix optional<vector<Tensor>>
    
    * fix bug
    
    * fix rmsprop
    
    * fix amp of eager_gen
    
    * polish code
    
    * fix deleted code
    
    * fix merge conflict
    
    * polish code
    
    * remove is_nullopt_
    
    * fix merge conflict
    
    * fix merge conflict

[33mcommit 2d87300809ae75d76f5b0b457d8112cb88dc3e27[m
Author: Aurelius84 <zhangliujie@baidu.com>
Date:   Fri May 27 18:07:09 2022 +0800

    [Dy2Stat]Replace paddle.jit.dy2stat with _jst (#42947)
    
    * [Dy2Stat]Replace paddle.jit.dy2stat with _jst
    
    * [Dy2Stat]Replace paddle.jit.dy2stat with _jst
    
    * refine code style
    
    * refine code style

[33mcommit a76f2b33d287d5f7faec7b8fe08eb8d611dc7175[m
Author: zhangbo9674 <82555433+zhangbo9674@users.noreply.github.com>
Date:   Fri May 27 16:02:59 2022 +0800

    Refine trunc uinttest logic  (#43016)
    
    * refine trunc uinttest
    
    * refine unittest
    
    * refine ut
    
    * refine fp64 grad check

[33mcommit ba157929e5ff15b69b18aa19cc1ab71c8fdb64bf[m
Author: wanghuancoder <wanghuan29@baidu.com>
Date:   Fri May 27 15:14:02 2022 +0800

    cast no need buffer (#42999)

[33mcommit 3d9fe71e3c043b715134f9991a85a6afb4cd6423[m
Author: Haipeng Wang <wanghaipeng03@baidu.com>
Date:   Fri May 27 14:52:51 2022 +0800

    experimental nvcc-lazy-module-loading (#43037)
    
    * experimental nvcc-lazy-module-loading
    
    * remove two empty last line from two files

[33mcommit 668e235cef7d1ee20d3a721e430103e508121604[m
Author: xiongkun <xiongkun03@baidu.com>
Date:   Fri May 27 13:54:39 2022 +0800

    change einsum_v2 as default and add new flags: FLAG_einsum_opt=1|0 (#43010)

[33mcommit 905d857ca8c41efca52bc817d9a99892fdf948b3[m
Author: Baibaifan <39549453+Baibaifan@users.noreply.github.com>
Date:   Fri May 27 11:17:37 2022 +0800

    fix_sharding_timeout (#43002)

[33mcommit 21f11d350cc348c5c2509d0935b7c2344c3d2f76[m
Author: Ruibiao Chen <chenruibiao@baidu.com>
Date:   Fri May 27 10:51:30 2022 +0800

    Support memory stats for CPU (#42945)
    
    * Support memory stats for CPU
    
    * Add UTs
    
    * Fix typos
    
    * Fix typos

[33mcommit b2b78cd416f8bd7d27cf3a18fccc8bf6d6f56cb5[m
Author: YuanRisheng <yuanrisheng@baidu.com>
Date:   Thu May 26 21:31:07 2022 +0800

    move instance_norm_double_grad (#43021)

[33mcommit 6af32a7fe57095619021d202ffbba37337fc5f19[m
Author: yaoxuefeng <yaoxuefeng@baidu.com>
Date:   Thu May 26 17:02:10 2022 +0800

    delete id 0 (#42951)
    
    delete id 0 in gpups

[33mcommit eb15e9a7aa51cfda6441b0648efdc3db76b4546d[m
Author: zhupengyang <zhu_py@qq.com>
Date:   Thu May 26 16:57:24 2022 +0800

    enhance yolo_box_fuse_pass (#42926)

[33mcommit 18323a463ae57447922207a9c8433dc81db5b330[m
Author: tianshuo78520a <707759223@qq.com>
Date:   Thu May 26 16:12:58 2022 +0800

    fix protobuf error (#43009)

[33mcommit cc272afb7e4ffde063a2876b3b13deeda9c45310[m
Author: YuanRisheng <yuanrisheng@baidu.com>
Date:   Thu May 26 14:21:52 2022 +0800

    [Phi]Refactor InstanceNormKernel and InstanceNormGradKernel (#42978)
    
    * move instance_norm
    
    * change mutable_data
    
    * fix compile bugs

[33mcommit 8f7f3ac9f2a0209959d0fe3bd8c8f50744f03b64[m
Author: danleifeng <52735331+danleifeng@users.noreply.github.com>
Date:   Thu May 26 12:42:32 2022 +0800

    [GPUPS]fix dymf gpups pscore (#42991)

[33mcommit 52ff3f4869b41e706536711803b664a58c156cf7[m
Author: ShenLiang <1422485404@qq.com>
Date:   Thu May 26 11:30:17 2022 +0800

    fix pipeline on processgroup (#42989)

[33mcommit 5b86e190f5143b2c4f7db37bbe7fa08ac5fe5301[m
Author: zlsh80826 <rewang@nvidia.com>
Date:   Thu May 26 11:23:55 2022 +0800

    Use all sitepackages path as the library/include path (#42940)

[33mcommit 3ee1b99b73ce56550342cb2fdb104f15b13704fb[m
Author: Leo Chen <chenqiuliang@baidu.com>
Date:   Thu May 26 10:36:29 2022 +0800

    remove Wno-error=parentheses-equality (#42993)

[33mcommit f70a734f289cd7a81410b94eb959bc2ba9e7ae0e[m
Author: Wangzheee <634486483@qq.com>
Date:   Wed May 25 21:11:44 2022 +0800

    fix_multi_int8 (#42977)

[33mcommit 657abd517f3930b37c2a665dc1ef5c8140252504[m
Author: jakpiase <jakpia21@gmail.com>
Date:   Wed May 25 14:51:52 2022 +0200

    OneDNN md-in-tensor refactoring part 4: Memory descriptor enabled for more ops (#42946)
    
    * added support for md in more ops
    
    * fixed typo

[33mcommit c6f98fa0ec9068ee93eead3beb6cce8a377f1342[m
Author: onecatcn <kaiwang85@qq.com>
Date:   Wed May 25 16:17:29 2022 +0800

    fix an bug in metrics.py; test=document_fix (#42976)
    
    PR types
    Bug fixes
    
    PR changes
    Docs
    
    Describe
    修复 paddle.metric.accuracy 文档，对应的中文文档修复为 https://github.com/PaddlePaddle/docs/pull/4811
    the file was editted based on the discussion in the issue:
    INT32 Failed on paddle.metric.accuracy: https://github.com/PaddlePaddle/Paddle/issues/42845

[33mcommit f1f79b0d9d18cebcf8b89775d2b066d6fdd04199[m
Author: Leo Chen <chenqiuliang@baidu.com>
Date:   Wed May 25 12:32:49 2022 +0800

    fix maybe-uninitialized warning (#42902)
    
    * fix maybe-uninitialized warning
    
    * fix compile
    
    * fix xpu compile
    
    * fix npu compile
    
    * fix infer compile
    
    * fix compile
    
    * fix compile

[33mcommit 45d7a3ea304090bd5cd0910450c8ffa7aee771a6[m
Author: danleifeng <52735331+danleifeng@users.noreply.github.com>
Date:   Wed May 25 12:32:07 2022 +0800

    [GPUPS]fix gpups pscore (#42967)

[33mcommit b685905474cd8c114b02787da7ebb9237d5b41ee[m
Author: Qi Li <qili93@qq.com>
Date:   Wed May 25 11:12:33 2022 +0800

    fix compile error on Loongson CPU, test=develop (#42953)

[33mcommit cbb241369f21d4002289649d0ea242d429b86c2b[m
Author: fwenguang <95677191+fwenguang@users.noreply.github.com>
Date:   Wed May 25 11:11:50 2022 +0800

    [MLU] adapt coalesce_tensor op for mlu (#42873)

[33mcommit 71b046cda4d2c1751cfbc280e3695261f12fe8b4[m
Author: xiongkun <xiongkun03@baidu.com>
Date:   Wed May 25 10:58:15 2022 +0800

    [EinsumOp] Optimize the backward speed of EinsumOp (#42663)
    
    * change logic for optimize
    
    * modifty
    
    * optimize the backward speed of EinsumOp
    
    * add cache optimizer for einsum op
    
    * EinsumOp: fix new dygraph mode error
    
    * fix bug
    
    * change Cache->InnerCache
    
    * fix code
    
    * fix
    
    * add nan inf utils for einsum op
    
    * add as_extra
    
    * Compatible with v2.3 EinsumOp
    
    * remove dispensable

[33mcommit e5fc68b2c34cc068274d33d127ecfda75e4ed4c2[m
Author: Ming-Xu Huang <mingh@nvidia.com>
Date:   Wed May 25 10:17:27 2022 +0800

    Dynamic graph support to Automatic SParsity. (#41177)
    
    * Dynamic graph support to Automatic SParsity.
    
    1. Added dynamic support to ASP module (paddle.fluid.contrib.sparsity).
    2. Added ASP related unit-tests regards to above changes.
    3. Put ASP module under paddle.static for now, waiting for APIs confirmation from Paddle.
    
    * Modified documents of functions to have correct examples.
    
    * Update in_dygraph_mode to paddle.in_dynamic_mode()
    
    * Modified documents of functions and added comments
    
    * Minor changes.
    
    * Fix example errors in asp API.
    
    * Code Change for Review
    
    1. Added more examples in documents.
    2. Chaged test_asp_pruning_static.
    
    * Minor changes
    
    * Update ASP function documents.
    
    * Update ASP function documents.
    
    * Reduce test case size of asp pruning due CI time limit.
    
    * Update time limitation to some asp UTs.
    
    * Fix sample code errors.
    
    * Fix sample code errors.
    
    * Fix sample code errors.
    
    * Update time limitation to parts of ASP UTs.
    
    * Update UTs to fit with CI.
    
    * Reduce problem size in python/paddle/fluid/tests/unittests/asp/test_fleet_with_asp_dynamic.py
    
    * Added paddle.asp
    
    * Fixed type casting error of OpRole.Optimize in new dygraph mode.
    
    * Made set_excluded_layers be compatible with 2.2
    
    * Fix example code of calculate_density.
    
    * Update code examples.
    
    * Move paddle.asp to paddle.incubate.asp
    
    * Fixed an example error of calculate_density

[33mcommit 4218957b202cedb7d52686f1ad555015e664f636[m
Author: Zhangjingyu06 <92561254+Zhangjingyu06@users.noreply.github.com>
Date:   Wed May 25 09:59:13 2022 +0800

    modify xpu.cmake *test=kunlun (#42962)

[33mcommit 53e503830cff4b3bcf00e99c8e368ffc62a115d7[m
Author: Baibaifan <39549453+Baibaifan@users.noreply.github.com>
Date:   Wed May 25 09:57:32 2022 +0800

    [Dygraph]fix_sharding3_offload (#42955)
    
    * fix_sharding3_offload
    
    * fix_fp16dtype_bug

[33mcommit 07dab9da12231b09271e0f05057458f391d948e4[m
Author: zyfncg <zhangyunfei07@baidu.com>
Date:   Tue May 24 22:21:52 2022 +0800

    fix namespace parser in eager_code_gen (#42957)

[33mcommit 4d7a9eef4237c2780ca5799805c74dcf90b3ceb8[m
Author: YuanRisheng <yuanrisheng@baidu.com>
Date:   Tue May 24 19:21:44 2022 +0800

    [Phi]Move grad_add op kernel into phi and delete elementwise_add_op file (#42903)
    
    * move grad_add
    
    * fix unittest bugs
    
    * fix compile bugs

[33mcommit 9e5acc1faebd0ab8f03f7e8b82fac29c63de3464[m
Author: jakpiase <jakpia21@gmail.com>
Date:   Tue May 24 10:30:23 2022 +0200

    updated paddle_bfloat to v0.1.7 (#42865)

[33mcommit b5ec9ca0cbf3f2fc0fd19a9b8159469855ce0c8d[m
Author: Allen Guo <alleng@graphcore.ai>
Date:   Tue May 24 16:30:02 2022 +0800

    upgrade to sdk2.5.1 (#42950)
    
    * upgrade to sdk2.5.1

[33mcommit d4cdfa55cbe682d54993445773d689024fbcdafd[m
Author: chentianyu03 <chentianyu03@baidu.com>
Date:   Tue May 24 15:39:33 2022 +0800

    [Yaml]add pad/pad3d/squeeze/unsqueeze yaml and test case (#42774)
    
    * add pad3d_double_grad yaml and test case
    
    * add squeeze and unsqueeze double grad
    
    * add double grad config
    
    * add pad_grad and pad_double_grad yaml
    
    * add pad_double_grad in config

[33mcommit de735a9a819cd2c53d115e99b25a422ede0614d9[m
Author: Feiyu Chan <chenfeiyu@baidu.com>
Date:   Tue May 24 15:05:14 2022 +0800

    fix cmake command, rm -> remove (#42927)

[33mcommit f8931c97985fac563dd095a6e81326ee4cfa8fb5[m
Author: Fan Zhang <frank08081993@gmail.com>
Date:   Tue May 24 14:39:27 2022 +0800

    [XPUPS] Modify XPU Kernel (#42745)
    
    * Adapt XPUPS - 1st version - 3.24
    
    * Adapt XPUPS - update XPU PushSparse -  2nd version - 3.24
    
    * Adapt XPUPS - add XPU PullSparseOp - 3nd version - 3.25
    
    * refactor heter comm kernel
    
    * update. test=develop
    
    * Adapt XPUPS - modify by compilation - 4th version - 3.27
    
    * update calc_shard_offset. test=develop
    
    * update xpu kernel. test=develop
    
    * update args of calc_shard_offset
    
    * update. test=develop
    
    * remove customGradMerger
    
    * update. test=develop
    
    * heter_comm update
    
    * heter_comm update
    
    * update calc_shard_offset. test=develop
    
    * heter_comm update
    
    * update args of calc_shard_offset
    
    * update. test=develop
    
    * remove customGradMerger
    
    * update. test=develop
    
    * fix. test=develop
    
    * update. test=develop
    
    * update. test=develop
    
    * update optimizer kernel
    
    * Adapt XPUPS - use WITH_XPU_KP and modify wrapper kernel function - 5th version - 3.30
    
    * update. test=develop
    
    * update pslib.cmake
    
    * update. test=develop
    
    * update. test=develop
    
    * update. test=develop
    
    * update. test=develop
    
    * update. test=develop
    
    * Adapt XPUPS - modify by kp compilation  - 6th version - 3.30
    
    * update. test=develop
    
    * update. test=develop
    
    * update. test=develop
    
    * update optimizer kernel
    
    * update. test=develop
    
    * update. test=develop
    
    * update. test=develop
    
    * update. test=develop
    
    * update. test=develop
    
    * update. test=develop
    
    * update. test=develop
    
    * update. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * used by minxu
    
    * update heter_comm_inl
    
    * fix. test=develop
    
    * Adapt XPUPS - modify by kp compilation  - 7th version - 3.30
    
    * fix. test=develop
    
    * add optimizer kernel. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * 3.31 update
    
    * Adapt XPUPS - update kp compilation path  - 8th version - 3.31
    
    * add optimizer kernel. test=develop
    
    * fix kunlun not support size_t. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix kunlun not support size_t. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * update heter_comm_kernel.kps 3.31
    
    * fix. test=develop
    
    * fix. test=develop
    
    * update heter_comm_kernel.kps 3.31
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * update heter_comm.h 3.31
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * update hashtable. test=develop
    
    * update. test=develop
    
    * Adapt XPUPS - update by kp compilation  - 9th version - 4.1
    
    * update hashtable. test=develop
    
    * fix. test=develop
    
    * update hashtable 4.1
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * Adapt XPUPS - update by kp compilation  - 10th version - 4.1
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * update. test=develop
    
    * modify by compilation 4.1
    
    * update. test=develop
    
    * update. test=develop
    
    * fix. test=develop
    
    * modify by compilation 4.1
    
    * update. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * modify by compilation 4.1
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * modify by compilation 4.1 19:30
    
    * fix. test=develop
    
    * update ps_gpu_wrapper.kps 4.1
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * Adapt XPUPS - update by kp compilation  - 11th version - 4.1
    
    * fix. test=develop
    
    * Adapt XPUPS - update by kp compilation  - 12nd version - 4.2
    
    * fix. test=develop
    
    * fix. test=develop
    
    * modify by compilation 4.2
    
    * 4.2 update
    
    * fix. test=develop
    
    * template init. test=develop
    
    * update 4.6
    
    * fix. test=develop
    
    * template init. test=develop
    
    * 4.6 modify by compilation
    
    * hashtable template init. test=develop
    
    * hashtable template init. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=devlop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=devlop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * Adapt XPUPS - update by kp compilation  - 13nd version - 4.7
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * 4.11 update
    
    * fix. test=develop
    
    * fix. test=develop
    
    * 4.11 update
    
    * update by pre-commit
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * fix. test=develop
    
    * 4.12 update
    
    * fix. test=develop
    
    * Adapt XPUPS - update by kp compilation  - 14th version - 4.13
    
    * 4.13 update
    
    * 4.14 update
    
    * 4.14 update
    
    * 4.14 update
    
    * 4.14 modify by merged latest compilation
    
    * retry CI 4.14
    
    * 4.15 pass static check
    
    * 4.15 modify by gpups CI
    
    * 3.16 update by gpups CI - modify ps_gpu_wrapper.h
    
    * 4.16 update
    
    * 4.16 pass xpu compile
    
    * 4.16 retry CI
    
    * 4.16 update
    
    * Adapt XPUPS - adapt BKCL comm for XPUPS - 4.24
    
    * update by compilation
    
    * Adapt XPUPS - register PSGPUTrainer for XPUPS - 4.25
    
    * update device_worker_factory
    
    * Adapt XPUPS - split heter_ps into .cu and .cc - 4.27
    
    * Adapt XPUPS - register pull_box_sparse op under XPU_KP - 4.28
    
    * update
    
    * 5.7 modify ps_gpu_wrapper pull_sparse
    
    * 5.11 update ps_gpu_wrapper CopyKeysKernel
    
    * 5.13 modify calc_shard_offset_kernel & fill_shard_key_kernel
    
    * modify fill_dvals_kernel & PullCopy & c_sync_calc_stream - 5.18
    
    * modify PushCopy & fill_shard_grads_kernel & register push_box_sparse - 5.19
    
    Co-authored-by: zmxdream <zhangminxu01@baidu.com>

[33mcommit ebf486acb8accd341cf19dc9667f365de0bdd57d[m
Author: kuizhiqing <kuizhiqing@baidu.com>
Date:   Tue May 24 11:16:57 2022 +0800

    [launch] fix timeout reset (#42941)

[33mcommit a5ad2659131fb0e753690d93311f6c842cfc46e2[m
Author: Zhangjingyu06 <92561254+Zhangjingyu06@users.noreply.github.com>
Date:   Tue May 24 11:00:03 2022 +0800

    modify xpu.cmake *test=kunlun (#42928)

[33mcommit d3c6afbff5933e306920dd351e0cfe0791b6d10a[m
Author: Ruibiao Chen <chenruibiao@baidu.com>
Date:   Tue May 24 10:01:30 2022 +0800

    Add type() interface for paddle::variant (#42943)
    
    * Add type() interface for variant
    
    * Fix CI errors

[33mcommit c60acca4a26264a98785da351f75ca7065edb407[m
Author: zyfncg <zhangyunfei07@baidu.com>
Date:   Mon May 23 22:43:52 2022 +0800

    Add assign_out_ yaml (#42833)
    
    * add assign_out_ yaml
    
    * fix final_state_assign
    
    * fix inplace bug
    
    * add inplace_check_blacklist for assign
    
    * fix merge conflict

[33mcommit c921a812bdb08ce8d3abfc472cb492462f740d71[m
Author: Chen Weihang <chenweihang@baidu.com>
Date:   Mon May 23 22:07:24 2022 +0800

    fix conv nd error (#42933)

[33mcommit 615d931c0a08f9a41d4e2a7a2f55cba07e691dc9[m
Author: Jiabin Yang <360788950@qq.com>
Date:   Mon May 23 22:06:38 2022 +0800

    Support to onnx test (#42698)
    
    * support to onnx test
    
    * add comments
    
    * remove log
    
    * remove log
    
    * update paddle2onnx version

[33mcommit e3ee2ad845d6169f2596ec850a6527aca4330478[m
Author: xiongkun <xiongkun03@baidu.com>
Date:   Mon May 23 22:02:01 2022 +0800

    sync stop_gradient in ParamBase. Fix the Different Behavior between Eval and Train (#42899)

[33mcommit fba94b9f1efee2530dab9e69cb35e28c3ac92a06[m
Author: Weilong Wu <veyron_wu@163.com>
Date:   Mon May 23 21:46:53 2022 +0800

    [Eager] Remove _enable_legacy for bfgs (#42936)

[33mcommit d414af940a956b51c0586b14f5b65265284bfe1a[m
Author: Jacek Czaja <jacek.czaja@intel.com>
Date:   Mon May 23 09:38:36 2022 +0200

    [Internal reviewing] NHWC fix to am_vocoder model for oneDNN 2.6 (#42729)
    
    * - prototype of reimplemented fixes
    
    * - compilation fixes
    
    * - compilation fix
    
    * - cosmetic info
    
    * - hopefully fix
    
    * - compilation fix
    
    * - supported for nested blocking of cache clearing
    
    * - fix
    
    * - Unit test to changes
    
    * - Compilation fix to windows (hopefully)
    
    * - Moved resetting layout to ResetBlob
    
    * - fixes after review

[33mcommit 0211a833a42cb7a2e378a1f172798b65632d276d[m
Author: YuanRisheng <yuanrisheng@baidu.com>
Date:   Mon May 23 15:32:19 2022 +0800

    Add double grad yaml for celu/sqrt/rsqrt/square op (#42895)
    
    * add double grad yaml
    
    * fix bugs when compile infrt

[33mcommit e5ebd347af93c698fead20d1f09aa577f89263e5[m
Author: pangyoki <pangyoki@126.com>
Date:   Mon May 23 15:29:38 2022 +0800

    support backward inplace for eager dygraph mode (#42795)
    
    * support inplace in backward
    
    * fix final_state_linear
    
    * fix format of backward_inplace_map
    
    * little change
    
    * add subtract in yaml
    
    * fix hook mem leak
    
    * fix hook use_count
    
    * little format change
    
    * fix
    
    Co-authored-by: JiabinYang <360788950@qq.com>

[33mcommit 2cb61405abcab502c07be750151ed0773175094e[m
Author: xiongkun <xiongkun03@baidu.com>
Date:   Mon May 23 14:23:25 2022 +0800

    add is_train into the cache key (#42889)
    
    * add is_train into the cache key
    
    * fix unittest error
    
    * add unittest
    
    * remove import

[33mcommit fa6b3c9a47c55b6bff5923c3e956e0b1cf3ab732[m
Author: zyfncg <zhangyunfei07@baidu.com>
Date:   Mon May 23 14:19:20 2022 +0800

    [Phi] Remove Storage (#42872)
    
    * remove storage
    
    * add glog include
    
    * add glog include
    
    * add glog include

[33mcommit 9aed83272c369fb77a24606693cbb8a17d2baaeb[m
Author: Ruibiao Chen <chenruibiao@baidu.com>
Date:   Mon May 23 12:56:25 2022 +0800

    Reduce test case for test_tensordot (#42885)
    
    * Reduce test case for test_tensordot
    
    * Fix CI errors

[33mcommit 65f705e1011f63c349813d7368d55b35df03ad82[m
Author: Weilong Wu <veyron_wu@163.com>
Date:   Mon May 23 12:06:39 2022 +0800

    [Eager] Support sharding_parallel under eager (#42910)

[33mcommit c0001a2433c1058ebfd21df22fe0f86146f16610[m
Author: yaoxuefeng <yaoxuefeng@baidu.com>
Date:   Mon May 23 11:49:19 2022 +0800

    Acc name (#42906)
    
    add dymf support of gpups

[33mcommit 3b488baea74edbffe895be7b42801edab57513ec[m
Author: Zhou Wei <1183042833@qq.com>
Date:   Mon May 23 11:48:42 2022 +0800

    remove is_init_py of RandomGenerator, and use Global RandomGenerator by default (#42876)
    
    * remove is_init_py of RandomGenerator, and use Global Generator if not OP seed
    
    * fix comment

[33mcommit 2b4977f20cbe962599c55ab57c99f0c2043bf478[m
Author: pangyoki <pangyoki@126.com>
Date:   Mon May 23 11:19:54 2022 +0800

    fix final_state_linear (#42820)

[33mcommit 9827c8b58b8cac88ae0db47aa193891f221ce5cb[m
Author: Sing_chan <51314274+betterpig@users.noreply.github.com>
Date:   Mon May 23 10:46:33 2022 +0800

    improve error info when no sample code found (#42742)
    
    * test=document_fix
    
    * exit 1 if no sample code found since api must have sample code;test=document_fix
    
    * test normal input;test=document_fix
    
    * delete test code;test=document_fix

[33mcommit 106083aa5f9641af029f7a678533cfb494a1c236[m
Author: shixingbo <90814748+bmb0537@users.noreply.github.com>
Date:   Mon May 23 10:42:02 2022 +0800

    Fix a bug in BroadcastConfig for KP XPU2 rec model   (#42866)

[33mcommit 2ffb337183f8d970c8b6eca002963061f48afba6[m
Author: Zuza Gawrysiak <zuzanna.gawrysiak@intel.com>
Date:   Sun May 22 16:35:19 2022 +0200

    Quantize elementwise sub (#42854)
    
    * Add elementwise_sub quantization
    
    * Remove unnecessary comments
    
    * Specify names for tests
    
    * Remove comments
    
    * Remove comments leftovers

[33mcommit 7b6bf28184cb53c78957d9e21d414b25b2c9bb41[m
Author: pangyoki <pangyoki@126.com>
Date:   Sat May 21 04:23:03 2022 +0800

    delete PADDLE_WITH_TESTING in memory_block_desc (#41817)
    
    * delete PADDLE_WITH_TESTING in memory_block_desc
    
    * test FLAGS_allocator_strategy=naive_best_fit
    
    * delete flag naive_best_fit

[33mcommit 0d878f1a696274297393114e58c7dd33564c79fc[m
Author: niuliling123 <51102941+niuliling123@users.noreply.github.com>
Date:   Fri May 20 23:05:21 2022 +0800

    Delete ElementwiseKernel in BroadcastKernel  (#42779)

[33mcommit c5d3bc0e1b0204d587f2aadce54742a3c4617cbb[m
Author: zyfncg <zhangyunfei07@baidu.com>
Date:   Fri May 20 21:26:07 2022 +0800

    support heterogeneous tensor for kernel in yaml (#42898)

[33mcommit 7306d1fba1efefe48b9bc151800ec3a42f5336ee[m
Author: WangXi <wangxi16@baidu.com>
Date:   Fri May 20 20:25:31 2022 +0800

    fix fused_attention_op cacheKV InferShape (#42900)

[33mcommit f36a9464bf19b07eb613471b4f39c7c1c756fe47[m
Author: Leo Chen <chenqiuliang@baidu.com>
Date:   Fri May 20 20:02:11 2022 +0800

    use fp32 compute type for cublasGemmStridedBatchedEx with fp16 input/output (#42851)
    
    * use fp32 compute type for cublasGemmStridedBatchedEx with fp16 input/output
    
    * add flags to control compute type
    
    * default to false
    
    * add unit test
    
    * default to true

[33mcommit 4a48e3d14070ee7cea0c61a4db3c1549a9bc8975[m
Author: Sing_chan <51314274+betterpig@users.noreply.github.com>
Date:   Fri May 20 16:43:29 2022 +0800

    【doc CI】simplify doc check log info (#42879)
    
    * simplify doc check log info;test=document_fix
    
    * test sample code error;test=document_fix
    
    * delete test code;test=document_fix

[33mcommit 191c441a0b4cf3eefd83adf136bc13150e31cf24[m
Author: YuanRisheng <yuanrisheng@baidu.com>
Date:   Fri May 20 16:22:16 2022 +0800

    move activation kernel (#42880)

[33mcommit d8b691242d02b4117eb4b06985cd0553946bac12[m
Author: Weilong Wu <veyron_wu@163.com>
Date:   Fri May 20 16:10:07 2022 +0800

    [Eager] Make CreateInferMeta more robust (#42871)

[33mcommit 723c4ae76d8c4123e614110da3c9ce22ad094f51[m
Author: Jiabin Yang <360788950@qq.com>
Date:   Fri May 20 16:08:26 2022 +0800

    fix hook mem leak (#42857)

[33mcommit 75db5b86f4124cb506b6aa82911111675109bef3[m
Author: xiaoguoguo626807 <100397923+xiaoguoguo626807@users.noreply.github.com>
Date:   Fri May 20 15:43:35 2022 +0800

    [Hackathon No.5] tril_indices OP (#41639)
    
    * add tril_indices cpu kernal
    
    * modify tril_indice cpu op
    
    * modify bug
    
    * modify bug
    
    * add tril_indices python api
    
    * add tril_indices python api
    
    * resolve conflict
    
    * add tril_indices test
    
    * modify details
    
    * add tril_indices.cu
    
    * pythonapi pass
    
    * save tril_indices
    
    * CPU tril_indices pass
    
    * delete vlog
    
    * modify test_tril_indices_op.py
    
    * delete tril_indices_kernel.cc.swp
    
    * delete tril_indice.cu
    
    * modify code style
    
    * add newline in creation.py
    
    * modify creation.py linux newline
    
    * delete annotation
    
    * check code style
    
    * check .py style add final_state??
    
    * modify code style
    
    * add gpu_tril_indices
    
    * modify gpu_compiled_juage
    
    * modify gpu judge
    
    * code style
    
    * add test example
    
    * modify english document
    
    modify english document
    
    modify english document
    
    modify document
    
    modify document
    
    * modify pram name
    
    * modify pram name
    
    * modify pram
    
    * reduce test ex

[33mcommit 1f76eabfe184b62bc6e6c49b854b7ddd8caa14b6[m
Author: zhaocaibei123 <48509226+zhaocaibei123@users.noreply.github.com>
Date:   Fri May 20 13:53:10 2022 +0800

    fix Wtype-limits (#42676)
    
    * fix Wtype-limits
    
    * fix
    
    * remove -Wno-error=type-limits

[33mcommit 11ce7eb11674fdea4dcecafdd1ace065d79447c2[m
Author: Leo Chen <chenqiuliang@baidu.com>
Date:   Fri May 20 13:52:49 2022 +0800

    add approval for changing warning flag (#42875)
    
    * add approval for changing warning flag
    
    * test for approval
    
    * revert changes

[33mcommit 56a8b3e30fa40876a4f03466ee243b6e51ac1514[m
Author: yaoxuefeng <yaoxuefeng@baidu.com>
Date:   Fri May 20 11:57:57 2022 +0800

    add dymf accessor support (#42881)

[33mcommit 5efc4146d3d3db1a7789364d1d04d444cabf5368[m
Author: zhupengyang <zhu_py@qq.com>
Date:   Fri May 20 11:35:07 2022 +0800

    add arg_max tensorrt converter, fix identity_scale_op_clean_pass (#42850)

[33mcommit 5d1bbecb0096bfd8ea0935183439763cf08d2a12[m
Author: zn <96479180+kangna-qi@users.noreply.github.com>
Date:   Fri May 20 11:02:51 2022 +0800

    [MLU]support to spawn processes on mlu (#41787)

[33mcommit 2caee61ff99aa368dd895c9c46aa6701edeac676[m
Author: Feiyu Chan <chenfeiyu@baidu.com>
Date:   Fri May 20 10:30:31 2022 +0800

    add files and directories generated during codegen for operators into gitignore (#42874)

[33mcommit 3f6192900a822321b2bfc2a982ba025788a36265[m
Author: yaoxuefeng <yaoxuefeng@baidu.com>
Date:   Fri May 20 00:31:43 2022 +0800

    merge dymf branch (#42714)
    
    merge dymf branch

[33mcommit e726960aa0751ec3ec33a49dba5679c8e7530c2d[m
Author: qipengh <huangqipeng@cambricon.com>
Date:   Thu May 19 19:47:57 2022 +0800

    [MLU] add lookup_table_v2 and unstack op (#42847)

[33mcommit 313f5d018fc74ef8d462bf945d31bc128176156d[m
Author: Rui Li <me@lirui.tech>
Date:   Thu May 19 18:43:09 2022 +0800

    Fix PD_INFER_DECL redefine (#42731)
    
    Signed-off-by: KernelErr <me@lirui.tech>

[33mcommit b522ca52df114fe63bf992732708e7fd071fe8ad[m
Author: jakpiase <jakpia21@gmail.com>
Date:   Thu May 19 11:19:09 2022 +0200

    OneDNN md-in-tensor refactoring part 3: Changes in quantize and dequantize (#42766)
    
    * added md support inside (de)quantizes
    
    * added missing file
    
    * changed paddle enforce text
    
    * another paddle enforce change
    
    * same as before
    
    * removed broken tests

[33mcommit 6d0e4e4a41381ceca5bd1439026f600a01db8196[m
Author: Sing_chan <51314274+betterpig@users.noreply.github.com>
Date:   Thu May 19 14:59:19 2022 +0800

    【CI】run all demo ci before exit in windows (#42700)
    
    * run all demo ci before exit;test=document_fix;test=windows_ci_inference
    
    * fix bug;test=document_fix;test=windows_ci_inference
    
    * improve log
    
    * commetn test code
    
    * modify according to zhouwei's comments

[33mcommit 4427f1b1726bbe148d2c9663b841f939fd0eeda8[m
Author: zyfncg <zhangyunfei07@baidu.com>
Date:   Thu May 19 14:52:21 2022 +0800

    [Phi] Change the output format of C++ backward api (Part2) (#42545)
    
    * change the output format of C++ backward api
    
    * fix merge conflict
    
    * fix sparse api code auto-gen
    
    * fix eager_gen bug
    
    * fix bug of output is null
    
    * fix bug of conv2d_grad_impl
    
    * fix optional grad
    
    * fix bug of eager-gen double_grad
    
    * fix bug
    
    * fix multiply_double_grad bug
    
    * fix bug of higher order derivative
    
    * fix bug of FillZeroForEmptyGradInput
    
    * remove redundant vector in grad_node
    
    * fix bug of test_deformable_conv_v1_op
    
    * fix bug of test_deformable_conv_v1_op
    
    * some refacotr

[33mcommit 892f6850583d830e45d165c814daf622056c1c6c[m
Author: Aganlengzi <aganlengzi@gmail.com>
Date:   Thu May 19 14:20:51 2022 +0800

    [NPU] minor changes for version control to support version without suffix (#42856)

[33mcommit 148582fef3c34f1984e623456c5a0cf276438a42[m
Author: danleifeng <52735331+danleifeng@users.noreply.github.com>
Date:   Thu May 19 14:18:07 2022 +0800

    【GPUPS】add ctr_dymf_accessor for pscore (#42827)

[33mcommit 7a171e3c8db75b928192ffb7c96ab6a11d2c5d50[m
Author: zyfncg <zhangyunfei07@baidu.com>
Date:   Thu May 19 14:08:31 2022 +0800

    [Phi] Remove shared_storage (#42821)
    
    * remove shared_storage
    
    * fix bug
    
    * fix rnn bug

[33mcommit 155fe05bbfb1ebfe24ba7ecaf96f38a01931997b[m
Author: Zhengyang Song <songzy_thu@163.com>
Date:   Thu May 19 12:24:11 2022 +0800

    Fix typos in the comment doc of SimpleRNN, LSTM, GRU: hidden_size -> input_size. (#42770)
    
    test=document_fix

[33mcommit ca359fec89735e34152d1fe90da3255bed32456d[m
Author: Chen Weihang <chenweihang@baidu.com>
Date:   Thu May 19 12:01:59 2022 +0800

    [CompileOpt] Refine enforce code and remove boost/variant include (#41093)
    
    * refine enforce code
    
    * refine enforce code
    
    * fix compile failed
    
    * fix infrt failed

[33mcommit 68babef1ba032c67d662e614ca96526b97ca8658[m
Author: seemingwang <seemingwang@users.noreply.github.com>
Date:   Thu May 19 11:24:49 2022 +0800

    distribute label evenly among partitions in graph engine (#42846)
    
    * enable graph-engine to return all id
    
    * change vector's dimension
    
    * change vector's dimension
    
    * enlarge returned ids dimensions
    
    * add actual_val
    
    * change vlog
    
    * fix bug
    
    * bug fix
    
    * bug fix
    
    * fix display test
    
    * singleton of gpu_graph_wrapper
    
    * change sample result's structure to fit training
    
    * recover sample code
    
    * fix
    
    * secondary sample
    
    * add graph partition
    
    * fix pybind
    
    * optimize buffer allocation
    
    * fix node transfer problem
    
    * remove log
    
    * support 32G+ graph on single gpu
    
    * remove logs
    
    * fix
    
    * fix
    
    * fix cpu query
    
    * display info
    
    * remove log
    
    * remove empyt file
    
    * distribute labeled data evenly in graph engine
    
    Co-authored-by: DesmonDay <908660116@qq.com>
