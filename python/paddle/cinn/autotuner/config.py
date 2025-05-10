from typing import *
from dataclasses import dataclass
from paddle.cinn import common

@dataclass
class ModelInfo:
    """
    用于存储模型信息的类
    
    属性:
        name (str): 模型名称
        shape (list): 模型输入形状
        program_builder: 程序构建器实例
    """
    name: str
    shape: List[int|Tuple[int,int]]
    

@dataclass
class SearchOption:
    """
    用于配置自动调优搜索过程的选项类
    
    属性:
        num_measure_trials (int): 调优过程中的测试次数
        repeat (int): 每个配置重复测量的次数，用于获得更稳定的性能数据
        timeout (int): 每个任务的超时时间(秒)
        callbacks (list): 回调函数列表，用于记录或处理搜索过程中的数据
        early_stopping (int, optional): 提前停止条件，连续多少次没有改进则停止搜索
        use_parallel (bool, optional): 是否使用并行搜索
        num_parallel_jobs (int, optional): 并行搜索时的作业数量
        device_id (int, optional): 执行调优的设备ID
    """
    # num_measure_trials: int = 100
    # repeat: int = 5
    # timeout: int = 10
    # target = common.DefaultTarget()
    # # Below not use now 
    # callbacks: List = None
    # early_stopping=None
    # use_parallel=False
    # num_parallel_jobs=1
    # device_id=0

    def __init__(
        self,
        num_measure_trials: int = 100,
        repeat: int = 5,
        timeout: int = 10,
        target = common.DefaultTarget(),
        callbacks: Optional[List] = None,
        early_stopping: Optional[int] = None,
        use_parallel: bool = False,
        num_parallel_jobs: int = 1,
        device_id: int = 0
    ):
        self.num_measure_trials = num_measure_trials
        self.repeat = repeat
        self.timeout = timeout
        self.callbacks = callbacks if callbacks else []
        self.early_stopping = early_stopping
        self.use_parallel = use_parallel
        self.num_parallel_jobs = num_parallel_jobs
        self.device_id = device_id
        self.target = target

    
    def __str__(self):
        """返回SearchOption的字符串表示"""
        return f"SearchOption(trials={self.num_measure_trials}, repeat={self.repeat}, timeout={self.timeout})"