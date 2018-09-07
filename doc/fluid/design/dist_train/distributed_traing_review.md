# Parallelism, Asynchronous,  Synchronous, Codistillation


For valuable models, it’s worth using more hardware resources to reduce the training time and improve the final model quality. This doc discuss various solutions, their empirical results and some latest researches.

# Model Parallelism
In some situations, larger and more complex models can improve the model quality. Sometimes, such models cannot fit in one device. Sometimes, parts of the model can be executed in parallel to improve speed. Model Parallelism address the issues by partitioning a single model and place the shards on several devices for execution.

A common way of model parallelism is partition the logic of “gradient application” to parameter servers, while leaving the forward and backward computation at training servers.

More flexible model parallelism is challenging. For example, multi-level-single-direction LSTM can be partitioned by layers, while such solution is not helpful for bi-directional LSTM. Different models can have quite different ways of partitioning and the benefits also depend on the underlying hardware. Framework needs to provide flexible APIs for user to define the customized partition scheme. For example, in TensorFlow, user can use tf.device() to specify the device placement. In MxNet, mx.AttrScope(ctx_group='dev1') does similar things. Recent research proposes to automatically find the optimal partition scheme with Reinforcement Learning, which is essentially solution space search algorithm that could cost a lot of extra hardware sources.

# Data Parallelism
Data Parallelism runs the same model on multiple devices, each taking in a partition of the input batch. It’s more commonly used for a few reasons. It generally applies to common SGD mini-batch training. Compared with model parallelism, which requires users to carefully partition their model and tune for good performance, data parallelism usually involves no more than calling an extra API and speed up is more predictable.

# Asynchronous Training
In asynchronous training, it usually involves a set of trainers and a set of parameter servers. The parameter servers collectively hold a single copy of shared parameters. While the trainers each holds a unique copy of model and trains the model independently. Each trainer pulls parameters from parameter servers and sends gradients to the parameter servers independently. Similarly the parameter servers applies the gradients to parameters as soon as the gradients are received and sends parameters whenever they are requested.

In theory, asynchronous training is not safe and unstable. Each trainer is very likely using stale copy of parameters and parameters are also likely to apply stale gradients. However, in practice, especially for large-scale nonconvex optimization, it is effective [1]. Compared with synchronous solution, which will be discussed later, asynchronous distributed training is easier to implement and scales to a few dozen workers without losing much performance due to network communication or other overhead. Besides, asynchronous training can make progress even in case of random trainer failure in the cluster.

Many production models, such as [3], are trained with distributed asynchronous solutions due to its scalability and effectiveness in practice. However, asynchronous training has its limitations. Usually, it’s not as stable as synchronous training. A warm-up phase is sometimes needed. Learning rate is usually smaller compared with synchronous training and decay is also often needed. Normally, asynchronous training doesn’t scale beyond 100 trainers. In other words, when putting more trainers beyond that, the model cannot converge faster.

# Synchronous Training
Unlike asynchronous training, synchronous training requires step barriers. Parameter servers needs to wait for gradients from all trainers before they are applied to parameters and trainers will always pull the latest parameters.

An obvious advantage of synchronous training is that the behavior is more clearly defined. Usually, it's more stable than asynchronous training. Learning rate can be set larger and for some vision tasks, the final accuracy can be slightly higher. (In my practical experience, for some models, it can actually be worse).

Synchronous training usually faces scalability and performance issues, if not carefully implemented or deployed. In [2], native synchronous training can be 20%~40% slower than asynchronous training. A common trick to avoid slowness, discussed in [1] and [2], is to have backups. N+M replicas are scheduled while only the first N is needed for the training step the proceed.

Similar to asynchronous training, the benefit of synchronous training diminishes quickly. Depending on the models, increasing the number of trainers (effectively batch size) beyond a point won’t delivers faster converge time or better final model quality.

# Codistillation
Codistillation is a technique that tries to scale the training further. A few training instance (each training instance can be distributed) are performed during the same period. Each training instance has extra losses that comes from the prediction of other training instances. (likey teacher and student) The training process converges faster and usually converge to a better model quality. [4]


# Reference

[1] Jeffrey Dean, Greg Corrado, Rajat Monga, Kai Chen, Matthieu Devin, Mark Mao, Andrew Senior, Paul Tucker, Ke Yang, Quoc V Le, et al. Large scale distributed deep networks.

[2] Jianmin Chen, Rajat Monga, Samy Bengio, and Rafal Jozefowicz. Revisiting distributed synchronous SGD.

[3] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google’s neural machine translation system: Bridging the gap between human and machine translation.

[4] LARGE SCALE DISTRIBUTED NEURAL NETWORK TRAINING THROUGH ONLINE DISTILLATION
