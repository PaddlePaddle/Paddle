/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

import "DataConfig.proto";
import "ModelConfig.proto";

package paddle;

message OptimizationConfig {
  required int32 batch_size = 3;
  required string algorithm = 4 [default = "async_sgd"];
  optional int32 num_batches_per_send_parameter = 5 [default = 1];
  optional int32 num_batches_per_get_parameter = 6 [default = 1];

  required real learning_rate = 7;
  optional real learning_rate_decay_a = 8 [default = 0];
  optional real learning_rate_decay_b = 9 [default = 0];
  optional string learning_rate_schedule = 27 [default = "constant"];
  // learning rate will be scaled according to learning_rate_schedule
  // 1), constant:
  // lr = learning_rate
  // 2), poly:
  // lr = learning_rate *
  //      pow(1 + learning_rate_decay_a * num_samples_processed,
  //          -learning_rate_decay_b)
  // 3), exp:
  // lr = learning_rate *
  //      pow(learning_rate_decay_a,
  //          num_samples_processed / learning_rate_decay_b)
  // 4), discexp:
  // lr = learning_rate *
  //      pow(learning_rate_decay_a,
  //          floor(num_samples_processed / learning_rate_decay_b))
  // 5), linear:
  // lr = max(learning_rate - learning_rate_decay_a * num_samples_processed,
  //          learning_rate_decay_b)

  // owlqn related
  // L1-regularization
  optional real l1weight = 10 [default = 0.1];
  // L2-regularization
  optional real l2weight = 11 [default = 0];
  // "c1" in wolfe condition: if (newobj <= oldobj + c1 * origDirDeriv * step)
  // then accept the step
  optional real c1 = 12 [default = 0.0001];
  // multiply the step with "backoff", when wolfe condition doesn't satisfy
  optional real backoff = 13 [default = 0.5];
  // how many "s"s and "y"s are kept in owlqn
  optional int32 owlqn_steps = 14 [default = 10];
  // accept the step if encountered "max_backoff" times of "reduce the step"
  optional int32 max_backoff = 15 [default = 5];
  // L2-regularization coefficient is reduced linearly from iteration 0 to
  // "l2weight_zero_iter", and set to 0 after "l2weight_zero_iter"
  // iterations. set "l2weight_zero_iter" to 0 to disable this strategy.
  optional int32 l2weight_zero_iter = 17 [default = 0];

  // averaged sgd
  // About average_window * numBatchProcessed parameter are used
  // for average. To be accurate, between average_window * numBatchProcessed
  // and 2 * average_window * numBatchProcessed parameters are used for
  // average.
  optional double average_window = 18 [default = 0];
  optional int64 max_average_window = 19 [default = 0x7fffffffffffffff];

  //////////////////////////
  // Options Adaptive SGD //
  //////////////////////////

  // learning method for sgd/asgd, such as "momentum", "adagrad", "adadelta", "rmsprop"
  // default learning method("momentum") use global decayed learning rate with momentum.
  // "adagrad", "adadelta" and "rmsprop" can set momentum too.
  optional string learning_method = 23 [default = "momentum"];
  optional real ada_epsilon = 24 [default = 1e-6];
  optional real ada_rou = 26 [default = 0.95];

  // Force to do average in cpu in order to save gpu memory usage
  optional bool do_average_in_cpu = 25 [default = false];

  // delta add rate in pserver, used while num_batches_per_send_parameter>1
  // will be divided by #machines automatically.
  optional real delta_add_rate = 28 [default = 1.0];

  // We split a large size into smaller mini-batches, whose sizes are
  // determined by mini_batch_size. It only takes effect when there is
  // an ExternalMachine.
  optional int32 mini_batch_size = 29 [default = 128];

  // automatically set if any one of parameters set sparse remote update flag
  optional bool use_sparse_remote_updater = 30 [default = false];

  // how to update center parameter and feedback to local parameter, 
  // when use local sgd update in cluster training.
  // A option is elastic_average, proposed by the paper: Deep learning with elastic averaging SGD.
  // If use elastic_average method, every trainer node should sample from whole data sets.
  optional string center_parameter_update_method = 31 [default = "average"];

  // shrink sparse parameter value
  // only works if parameter is remote sparse update and has L1 decay rate
  optional real shrink_parameter_value = 32 [default = 0];

  ////////////////////////////
  // Options Adam Optimizer //
  ////////////////////////////
  optional real adam_beta1 = 33 [default = 0.9];
  optional real adam_beta2 = 34 [default = 0.999];
  optional real adam_epsilon = 35 [default = 1e-8];

  // arguments for learning rate scheduler
  // Format: num1:rate1,num2:rate2,...,numK:rateK
  // For learning_rate_schedule="manual", num is the number of samples,
  // For learning_rate_schedule="pass_manual",
  //  num is the number of passes (starting from 0)
  optional string learning_rate_args = 36 [default = ""];
 
  // for async sgd gradient commit control.
  // when async_lagged_grad_discard_ratio * num_gradient_servers commit passed,
  // current async gradient will be discard silently.
  optional real async_lagged_grad_discard_ratio = 37 [default = 1.5];
};

message TrainerConfig {
  optional ModelConfig model_config = 1;
  optional DataConfig data_config = 2;
  required OptimizationConfig opt_config = 3;
  optional DataConfig test_data_config = 4;
  repeated string config_files = 5;

  // the directory to save/load model files for each training path
  optional string save_dir = 6 [default = "./output/model"];

  // Path of the initial model parameters.
  // If it was set, start_pass will be ignored.
  optional string init_model_path = 7;

  // Start training from this pass.
  // Will load parameter from the previous pass.
  optional int32 start_pass = 8 [default = 0];

  // file path to the trainer config file
  optional string config_file = 9;
}
