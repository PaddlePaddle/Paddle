# Argument Outline

It looks like there are a lot of arguments. However, most of them are for developers or alrealy set automatically in cluster submitting environment and users do not need to care about them. Here, we divide these arguments into serveral classes according to the scenario that they are used in. For example, the arguments in `common` can be used in all scenes. Some arguments can be only used in certain layers. Some are needed by multi machines training in cluster, etc.

<html>
<table border="2" frame="border">
<thead>
<tr>
<th scope="col" class="left"></th>
<th scope="col" class="left">args</th>
<th scope="col" class="left">local train</th>
<th scope="col" class="left">cluster train</th>
<th scope="col" class="left">local test</th>
<th scope="col" class="left">cluster test</th>
</tr>
</thead>

<tbody>
<tr>
<td class="left" rowspan="9">common</td>
<td class="left">job</td>
<td class="left">√</td><td class="left">√</td><td class="left">√</td><td class="left">√</td>
</tr>

<tr>
<td class="left">use_gpu</td>
<td class="left">√</td><td class="left">√</td><td class="left">√</td><td class="left">√</td>
</tr>

<tr>
<td class="left">local</td>
<td class="left">√</td><td class="left">√</td><td class="left">√</td><td class="left">√</td>
</tr>

<tr>
<td class="left">config</td>
<td class="left">√</td><td class="left">√</td><td class="left">√</td><td class="left">√</td>
</tr>

<tr>
<td class="left">config_args</td>
<td class="left">√</td><td class="left">√</td><td class="left">√</td><td class="left">√</td>
</tr>

<tr>
<td class="left">num_passes</td>
<td class="left">√</td><td class="left">√</td><td class="left">√</td><td class="left">√</td>
</tr>

<tr>
<td class="left">trainer_count</td>
<td class="left">√</td><td class="left">√</td><td class="left">√</td><td class="left">√</td>
</tr>

<tr>
<td class="left">version</td>
<td class="left">√</td><td class="left">√</td><td class="left">√</td><td class="left">√</td>
</tr>

<tr>
<td class="left">show_layer_stat</td>
<td class="left">√</td><td class="left">√</td><td class="left">√</td><td class="left">√</td>
</tr>

<tr>
<td class="left" rowspan="15">train</td><td class="left">dot_period</td>
<td class="left">√</td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">test_period</td>
<td class="left">√</td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">saving_period</td>
<td class="left">√</td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">show_parameter_stats_period</td>
<td class="left">√</td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">init_model_path</td>
<td class="left">√</td><td class="left">√</td><td class="left">√</td><td class="left"></td>
</tr>

<tr>
<td class="left">load_missing_parameter_strategy</td>
<td class="left">√</td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">saving_period_by_batches</td>
<td class="left">√</td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">use_old_updater</td>
<td class="left">√</td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">enable_grad_share</td>
<td class="left">√</td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">grad_share_block_num</td>
<td class="left">√</td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">log_error_clipping</td>
<td class="left">√</td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">log_clipping</td>
<td class="left">√</td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">save_only_one</td>
<td class="left">√</td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">start_pass</td>
<td class="left">√</td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">train/test</td><td class="left">save_dir</td>
<td class="left">√</td><td class="left">√</td><td class="left">√</td><td class="left">√</td>
</tr>

<tr>
<td class="left" rowspan = "2">testing during training</td><td class="left">test_period</td>
<td class="left">√</td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">average_test_period</td>
<td class="left">√</td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left" rowspan = "5">test</td><td class="left">model_list</td>
<td class="left"></td><td class="left"></td><td class="left">√</td><td class="left">√</td>
</tr>

<tr>
<td class="left">test_wait</td>
<td class="left"></td><td class="left"></td><td class="left">√</td><td class="left">√</td>
</tr>

<tr>
<td class="left">test_pass</td>
<td class="left"></td><td class="left"></td><td class="left">√</td><td class="left">√</td>
</tr>

<tr>
<td class="left">predict_output_dir</td>
<td class="left"></td><td class="left"></td><td class="left">√</td><td class="left">√</td>
</tr>

<tr>
<td class="left">distribute_test</td>
<td class="left"></td><td class="left"></td><td class="left">√</td><td class="left">√</td>
</tr>

<tr>
<td class="left">Auc/PnpairValidation</td><td class="left">predict_file</td>
<td class="left"></td><td class="left"></td><td class="left"></td>√<td class="left">√</td>
</tr>

<tr>
<td class="left" rowspan = "6">GPU</td><td class="left">gpu_id</td>
<td class="left">√</td><td class="left">√</td><td class="left">√</td><td class="left">√</td>
</tr>

<tr>
<td class="left">parallel_nn</td>
<td class="left">√</td><td class="left">√</td><td class="left">√</td><td class="left">√</td>
</tr>

<tr>
<td class="left">allow_only_one_model_on_one_gpu</td>
<td class="left">√</td><td class="left">√</td><td class="left">√</td><td class="left">√</td>
</tr>

<tr>
<td class="left">cudnn_dir</td>
<td class="left">√</td><td class="left">√</td><td class="left">√</td><td class="left">√</td>
</tr>

<tr>
<td class="left">cuda_dir</td>
<td class="left">√</td><td class="left">√</td><td class="left">√</td><td class="left">√</td>
</tr>

<tr>
<td class="left">cudnn_conv_workspace_limit_in_mb</td>
<td class="left">√</td><td class="left">√</td><td class="left">√</td><td class="left">√</td>
</tr>

<tr>
<td class="left" rowspan = "4">RNN</td>
<td class="left">beam_size</td>
<td class="left"></td><td class="left"></td><td class="left">√</td><td class="left">√</td>
</tr>

<tr>
<td class="left">rnn_use_batch</td>
<td class="left">√</td><td class="left">√</td><td class="left">√</td><td class="left">√</td>
</tr>

<tr>
<td class="left">prev_batch_state</td>
<td class="left">√</td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">diy_beam_search_prob_so</td>
<td class="left"></td><td class="left"></td><td class="left">√</td><td class="left">√</td>
</tr>

<tr>
<td class="left" rowspan = "16">PServer</td><td class="left">start_pserver</td>
<td class="left"></td><td class="left">√</td><td class="left"></td><td class="left">√</td>
</tr>

<tr>
<td class="left">pservers</td>
<td class="left"></td><td class="left">√</td><td class="left"></td><td class="left">√</td>
</tr>

<tr>
<td class="left">port</td>
<td class="left"></td><td class="left">√</td><td class="left"></td><td class="left">√</td>
</tr>

<tr>
<td class="left">port_num</td>
<td class="left"></td><td class="left">√</td><td class="left"></td><td class="left">√</td>
</tr>

<tr>
<td class="left">ports_num_for_sparse</td>
<td class="left"></td><td class="left">√</td><td class="left"></td><td class="left">√</td>
</tr>

<tr>
<td class="left">nics</td>
<td class="left"></td><td class="left">√</td><td class="left"></td><td class="left">√</td>
</tr>

<tr>
<td class="left">rdma_tcp</td>
<td class="left"></td><td class="left">√</td><td class="left"></td><td class="left">√</td>
</tr>

<tr>
<td class="left">small_messages</td>
<td class="left"></td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">loadsave_parameters_in_pserver</td>
<td class="left"></td><td class="left">√</td><td class="left"></td><td class="left">√</td>
</tr>

<tr>
<td class="left">log_period_server</td>
<td class="left"></td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">pserver_num_threads</td>
<td class="left"></td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">sock_send_buf_size</td>
<td class="left"></td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">sock_recv_buf_size</td>
<td class="left"></td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">num_gradient_servers</td>
<td class="left"></td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">parameter_block_size</td>
<td class="left"></td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">parameter_block_size_for_sparse</td>
<td class="left"></td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left" rowspan = "3">Async SGD</td><td class="left">async_count</td>
<td class="left"></td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">async_lagged_ratio_min</td>
<td class="left"></td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">async_lagged_ratio_default</td>
<td class="left"></td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left" rowspan = "8">Performance Tuning</td><td class="left">log_barrier_abstract</td>
<td class="left"></td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">log_barrier_lowest_nodes</td>
<td class="left"></td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">log_barrier_show_log</td>
<td class="left"></td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">check_sparse_distribution_batches</td>
<td class="left"></td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">check_sparse_distribution_ratio</td>
<td class="left"></td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">check_sparse_distribution_unbalance_degree</td>
<td class="left"></td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">check_sparse_distribution_in_pserver</td>
<td class="left"></td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">show_check_sparse_distribution_log</td>
<td class="left"></td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">Data Provider</td><td class="left">memory_threshold_on_load_data</td>
<td class="left">√</td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left" rowspan = "2">RandomNumber</td><td class="left">seed</td>
<td class="left">√</td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">thread_local_rand_use_global_seed</td>
<td class="left">√</td><td class="left">√</td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">UnitTest</td><td class="left">checkgrad_eps</td>
<td class="left"></td><td class="left"></td><td class="left"></td><td class="left"></td>
</tr>

<tr>
<td class="left">Matrix/Vector</td><td class="left">enable_parallel_vector</td>
<td class="left">√</td><td class="left">√</td><td class="left">√</td><td class="left">√</td>
</tr>

</tbody>

</table>
</html>
