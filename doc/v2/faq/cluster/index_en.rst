###############################
Cluster Training and Prediction
###############################

.. contents::

1. Network connection errors in the log during multi-node cluster training
------------------------------------------------
There are maybe some errors in the log belonging to network connection problem during multi-node cluster training, for example, :code:`Connection reset by peer`.
This kind of error is usually caused by the abnormal exit of a training process in some node, and the other nodes cannot connect with this node any longer. Steps to troubleshoot the problem are as follows:

* Find the first error in the :code:`train.log`, :code:`server.log`, check whether other fault casued the problem, such as FPE, lacking of memory or disk.

* If the first error in server.log says "Address already used", this may be caused by the port conflict of the non-exclusive execution. Connect the sys-admin to check if the current MPI cluster supports jobs submitted with parameter :code:`resource=full`. If the current MPI cluster does not support this parameter, change the server port and try agian.

* If the current MPI cluster does not support exclusive pattern which allows a process to occupy the whole node, ask the administrator to replace or update the this cluster.
