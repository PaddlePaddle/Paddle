###############################
Cluster Training and Prediction
###############################

.. contents::

1. Network connection errors in the log during muliti-node cluster training
------------------------------------------------
The errors in the log belong to network connection during mulilti-node cluster training, for example, :code:`Connection reset by peer`.
This kind of error is usually caused by the abnormal exit of the training process in some node, and the others cannot connect with this node any longer. Steps to troubleshoot the problem as follows:

* Find the first error in the :code:`train.log`, :code:`server.log`, check whether other fault casued the problem, such as FPE, lacking of memory or disk.

* If network connection gave rise to the first error in the log, this may be caused by the port conflict of the non-exclusive execution. Connect with the operator to check if the current MPI cluster supports jobs submitted with parameter :code:`resource=full`. If so, change the port of job.

* If the currnet MPI cluster does not support exclusive pattern, ask the operator to replace or update the current cluster.
