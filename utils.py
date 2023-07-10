import torch
import torch.nn.functional as F
import logging
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from typing import Tuple

accelerator = None

def setup_accelerator():
    global accelerator
    if accelerator is None:
        accelerator = Accelerator(split_batches=True)
    return accelerator

def synchronize_if_distributed():
    if accelerator.use_distributed:
        accelerator.wait_for_everyone()

def to_cuda(batch):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(accelerator.device, non_blocking=True)

histroy_logs = set()
def print_rank_0(info, only_on_cuda0=False):
    if accelerator and not accelerator.is_main_process:
        return
    if only_on_cuda0 and info not in histroy_logs:
        histroy_logs.add(info)
        logging.info(info)
    return

def get_eval_ds_config(offload=None, stage=3):
    deepspeed_states = AcceleratorState().deepspeed_plugin

    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": device
        }
    }
    return {
        "train_micro_batch_size_per_gpu": deepspeed_states.deepspeed_config['train_micro_batch_size_per_gpu'],
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": True
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }

@torch.no_grad()
def get_global_statistics(accelerator, xs: torch.Tensor, mask=None, device='cpu') -> Tuple[float, float, int]:
    """
    Computes element-wise mean and variance of the tensor across processes
    https://github.com/microsoft/LMOps/blob/cde1fb1ef4608a7ac5bf00675fa3e94b1d960abb/minillm/minillm/utils.py#L108
    """
    xs = xs.to(accelerator.device)
    sum_and_count = torch.tensor([xs.sum(), (xs.numel() if mask is None else mask.sum())], device=xs.device)
    sum_and_count = accelerator.reduce(sum_and_count)
    global_sum, count = sum_and_count
    global_mean = global_sum / count

    sum_var = torch.sum(((xs - global_mean) ** 2).mul(1 if mask is None else mask))
    sum_var = accelerator.reduce(sum_var)
    global_var = sum_var / count
    
    return global_mean.to(device), global_var.to(device), count.to(device)

class RunningMoments:
    def __init__(self, accelerator):
        """
        Calculates the running mean and standard deviation of a data stream. Modified version of
        https://github.com/DLR-RM/stable-baselines3/blob/a6f5049a99a4c21a6f0bcce458ca3306cef310e0/stable_baselines3/common/running_mean_std.py
        """
        self.mean = 0
        self.std = 1
        self.var = 1
        self.count = 1e-24
        self.accelerator = accelerator

    @torch.no_grad()
    def update(self, xs: torch.Tensor) -> Tuple[float, float]:
        """
        Updates running moments from batch's moments computed across ranks
        """
        if self.accelerator.use_distributed:
            xs_mean, xs_var, xs_count = get_global_statistics(self.accelerator, xs)
        else:
            xs_count = xs.numel()
            xs_var, xs_mean = torch.var_mean(xs, unbiased=False)
        xs_mean, xs_var = xs_mean.float(), xs_var.float()

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += delta * xs_count / tot_count
        self.var = tot_sum / tot_count
        self.std = (self.var * tot_count / (tot_count - 1)).float().sqrt()
        self.count = tot_count

        return xs_mean.item(), (xs_var * xs_count / (xs_count - 1)).float().sqrt().item()

@torch.no_grad()
def whiten(xs: torch.Tensor, mask: torch.BoolTensor, shift_mean=True, accelerator=None) -> torch.Tensor:
    """
    Whitens values
    """
    if accelerator != None and accelerator.use_distributed:
        mean, var, _ = get_global_statistics(accelerator, xs, mask=mask, device=accelerator.device)
    else:
        mean = xs.sum() / mask.sum()
        var = torch.sum(((xs - mean) ** 2).mul(mask)) / mask.sum()

    whitened = (xs - mean) * torch.rsqrt(var + 1e-6)
    if not shift_mean:
        whitened += mean
    return whitened

def top_p_logits(logits, topp=0.9, filter_value=0, min_topk=1):
    """
    Filter a distribution of logits using nucleus (top-p) filtering
    https://github.com/OpenLMLab/MOSS/blob/e088f438d1a95d424c6dffef0d73134ebe62cb72/models_jittor/generation.py#L146
    """
    cum_logits = logits.clone()
    if topp > 0:
        logits_sorted, inds = torch.sort(logits, dim=-1, descending=True)
        mask = (logits_sorted.cumsum(dim=-1) - logits_sorted) >= topp
        mask[:, :min_topk] = False
        # Remove tokens with cumulative top_p above the threshold
        mask = torch.zeros_like(mask).to(torch.bool).scatter_(dim=-1, index=inds, src=mask)
        cum_logits[mask] = filter_value
        cum_logits.div_(cum_logits.sum(dim=-1, keepdim=True))
        
    return cum_logits

def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=-1)
    logpy = torch.gather(logp, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return logpy

def get_category_distribution_entropy(bsz, logits):
    """
    Compute category distribution entropy
    """
    logits_distribution = torch.distributions.categorical.Categorical(logits=logits.reshape(-1, logits.size(-1)))
    ent = logits_distribution.entropy().reshape(bsz, -1)
    return ent

def pad_sequences(seqs, pad_value, padding='right'):
    """
    Padding sequence to the same length
    """
    max_len = max(len(seq) for seq in seqs)
    if padding == 'right':
        padded_seqs = [seq + [pad_value] * (max_len - len(seq)) for seq in seqs]
    elif padding == 'left':
        padded_seqs = [[pad_value] * (max_len - len(seq)) + seq for seq in seqs]
    else:
        assert ValueError
    return padded_seqs