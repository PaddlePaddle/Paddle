## Logic
All modules are decupled when we store data(program and some output) to etcd.

- Transpile program desc to sub program descs(graphs) and
- Store them to etcd
	- one graph has uniq ID. 
	- one graph has desired resource.
- Start workers(pods) to run one graph
   - worker store pod info to etcd, so 
   - the graph can commicate with each other by graph_ID.

## 


## Architect graph
<div style="align: center">
<img src="src/arch2.png" width="700" align=center/>
</div>

- foreground job: when client exits the jobs will be killed.
- background job: client's death doesn't affect the job.

## Peudo code of users
```
...
t_graphs,p_graphs = fluid.dist_transpiler()

job_name = "test_1"

# you can kill first
#if fluid.k8s.find(job_name):
#	fluid.k8s.kill(job_name)

job, err = fulid.k8s.init_job(job_name, run_type=foreground)
if err is not null:
   print "start job:", job_name, " errors:", err
   sys.exit(1)
   
trainers = job.add_workers(t_graphs,cpu=,gpu=,mem)
pservers = job.add_workers(p_graphs,cpu=,gpu=,mem)

pserver.start(mode=sync)
trainer.start(pass_num=10)

accs = trainers.get(acc)
for c in acc:
    print(" acc:" + str(c))

jobs.stop()
```


## Data base 
- etcd is a key-value storage, but we can convert table to key-value style easily by use combination key.
- We store info in multiple tables because some of them may be changed more frequtely than others.

### Table: graph_program_desc

| column name | description|
|----------|-------------|
| graph_ID |  ID of graph, key    |
| program_desc| program desc to be executed    |
| send_var_graph_map|map of var which will be send and graph_ID|
| recv_var_graph_map|map of var which will be received and graphID|
|resource|resource need by this graph|

### Table: graph_pod
| column name | description|
|----------|-------------|
|graph_ID|ID of graph|
|pod_name|pod name which execut graph, may be changed|
|pod_ip|pod ip which execut graph|
|pod_port|pod port which execut graph|

### Table: graph_output
| column name | description|
|----------|-------------|
|graph_ID|ID of graph|
|output|output of this graph,it's a list|
|checkpoint|last checkpoint of this graph|

## About Data-Operator
## About Checkpoint-Operator

## Fault tolerant
- Program desc contains check_point operator and
- Kubernets start new worker and worker executor normally.

## Auto scaling
Change graph_ID and send/recv variable map, so the  workers can communcation with others correctly.


## Discussion
- Is database like mysql is enough for us?
