from typing import List, Optional, Any, Dict
import math
from accelerate import Accelerator
import torch
from torch.utils.tensorboard import SummaryWriter

class Metric:
    def __init__(self):
        pass

    def add(self, val):
        raise NotImplementedError

    def val(self) -> float:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def compute(self, val: Any):
        return val

    def __add__(self, other):
        raise NotImplementedError
    
    def __radd__(self, other):
        return self.__add__(other)


class MeanMetric(Metric):
    def __init__(self, num=0, denom=0):
        self.numerator = num
        self.denominator: int = denom

    def add(self, val: Any):
        self.numerator += self.compute(val)
        self.denominator += 1

    def many(self, vals: List[Any], denoms: Optional[List[int]] = None):
        if denoms is None:
            denoms = [1] * len(vals)
        assert len(vals) == len(denoms)

        for v, n in zip(vals, denoms):
            self.numerator += self.compute(v)
            self.denominator += n
    
    def val(self):
        if self.denominator == 0:
            return 0
        return self.numerator / self.denominator

    def reset(self):
        self.numerator = self.denominator = 0

    def __add__(self, other: 'MeanMetric'):
        return MeanMetric(self.numerator + other.numerator, self.denominator + other.denominator)

class SumMetric(Metric):
    def __init__(self, sum_=0):
        self.sum_ = sum_

    def add(self, val):
        self.sum_ += self.compute(val)

    def many(self, vals: List[Any]):
        self.sum_ += sum(self.compute(v) for v in vals)

    def val(self):
        return self.sum_

    def reset(self):
        self.sum_ = 0

    def __add__(self, other: 'SumMetric'):
        return SumMetric(self.sum_ + other.sum_)


class RealtimeMetric(Metric):
    def __init__(self, val=0):
        self.v = val

    def add(self, val):
        self.v = self.compute(val)
        
    def many(self, vals: List[Any]):
        self.add(vals[-1])
    
    def val(self):
        return self.v

    def reset(self):
        self.v = 0

    def __add__(self, other):
        return RealtimeMetric(self.v)

class PPLMetric(MeanMetric):
    def val(self):
        try:
            return math.exp(super().val())
        except OverflowError:
            return super().val()

    def __add__(self, other):
        return PPLMetric(self.numerator + other.numerator, self.denominator + other.denominator)


class Metrics():
    tb_writer = None
    def __init__(self, opt: Dict[str, Any], accelerator, mode='train'):
        self.metrics = {}
        self.mode = mode
        self.opt = opt
        self.accelerator = accelerator

        if Metrics.tb_writer is None and opt.logdir is not None and self.accelerator.is_main_process:
            Metrics.tb_writer = SummaryWriter(opt.logdir)

    def create_metric(self, metric_name: str, metric_obj: Metric):
        assert metric_name not in self.metrics
        self.metrics[metric_name] = metric_obj

    def record_metric(self, metric_name: str, val: Any):
        self.metrics[metric_name].add(val)

    def record_metric_many(self, metric_name: str, vals: List[Any], counts: Optional[List[int]] = None):
        if counts is None:
            self.metrics[metric_name].many(vals)
        else:
            self.metrics[metric_name].many(vals, counts)

    def reset(self, no_reset = ['global_exs']):
        for k, v in self.metrics.items():
            if k not in no_reset:
                v.reset()
                
    def all_gather_metrics(self):
        with torch.no_grad():
            metrics_tensor = {k: torch.tensor([v.val()], device=self.accelerator.device) for k, v in self.metrics.items()}
            
            if self.accelerator.use_distributed:
                gathered_metrics = self.accelerator.gather(metrics_tensor)
                for metric_name, gathered_tensor in gathered_metrics.items():
                    if metric_name == 'global_exs':
                        gathered_metrics[metric_name] = gathered_tensor.sum()
                    else:
                        gathered_metrics[metric_name] = gathered_tensor.float().mean()
            else:
                gathered_metrics = metrics_tensor
                                        
            gathered_metrics = {k: v.item() for k, v in gathered_metrics.items()}
        return gathered_metrics
    
    def write_tensorboard(self, global_step, gathered_metrics: Dict[str, float] = None):
        results = self.all_gather_metrics() if gathered_metrics is None else gathered_metrics
        if self.tb_writer is not None:
            for k, scalar in results.items():
                title = f"{k}/{'train' if 'train' == self.mode else 'eval'}"
                self.tb_writer.add_scalar(tag=title, scalar_value=scalar, global_step=global_step)
                
    def flush(self):
        if self.tb_writer is not None:
            self.tb_writer.flush()

    def display(self, global_step, data_size = None, gathered_metrics: Dict[str, float] = None):
        if not self.accelerator.is_main_process:
            return
        results = self.all_gather_metrics() if gathered_metrics is None else gathered_metrics
        log_str = ''
        if data_size is not None and 'global_exs' in results:
            print(f"=========== Step: {global_step}, Epoch: {(results['global_exs'] / data_size):.2f} ===========")
        else:
            print(f'=========== Step: {global_step} ===========')
        for k, value in results.items():
            if isinstance(value, float):
                if k == 'lr':
                    value = f'{value:.3e}'
                else:
                    value = f'{value:.4f}'
            log_str += f'{k}: {value}\t'
        print(log_str)        
        return results        

    
