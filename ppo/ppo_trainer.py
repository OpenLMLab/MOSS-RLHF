import time, math, os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple, List
from torch.utils.data import DataLoader
from .ppo_datahelper import *
from utils import *
from metric import MeanMetric, PPLMetric, SumMetric, RealtimeMetric
from accelerate import Accelerator
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
from metric import Metrics


class TrainState:
    def __init__(self):
        self.total_steps = 0
        self.total_exps = 0
        self.best_score = -9999999
    
    def state_dict(self):
        return {
            'total_steps': self.total_steps,
            'total_exps': self.total_exps,
            'best_score': self.best_score,
        }

class RLHFTrainableModelWrapper(nn.Module):
    def __init__(self, policy_model, critic_model) -> None:
        super().__init__()
        self.policy_model = policy_model
        self.critic_model = critic_model
    
    def forward(self, inputs, **kwargs):
        return self.policy_model(decoder_input=inputs, **kwargs), self.critic_model(decoder_input=inputs, only_last=False, **kwargs)
    
    def train(self, mode=True):
        self.policy_model.train(mode)
        self.critic_model.train(mode)
        
    def eval(self):
        self.policy_model.eval()
        self.critic_model.eval()


class PPOTrainer():
    def __init__(self, opt, policy_model, ref_model, critic_model, reward_model, accelerator, **kwargs) -> None:
        self.opt = opt
        self.no_reset_metric_names = ['global_exs'] # metrics *NOT* be reset per save point
        self.print_interval = opt.num_rollouts // opt.batch_size

        self.num_rollouts: int = opt.num_rollouts
        self.reward_clip: float = opt.reward_clip
        self.pg_clip: float = opt.pg_clip
        self.value_clip: float = opt.value_clip
        self.entropy_clip: float = opt.entropy_clip
        self.advantage_clip: float = opt.advantage_clip
        self.kl_penalty_weight: float = opt.kl_penalty_weight
        self.vf_loss_weight: float = opt.vf_loss_weight
        self.entropy_loss_weight: float = opt.entropy_loss_weight

        self.ppo_pretrain_data_path: str = opt.ppo_pretrain_data_path
        self.ppo_pretrain_data_type: str = opt.ppo_pretrain_data_type
        self.ppo_pretrain_loss_weight: float = opt.ppo_pretrain_loss_weight

        self.use_entropy_loss: bool = opt.use_entropy_loss
        self.use_reward_clip: bool = opt.use_reward_clip
        self.use_reward_scaling: bool = opt.use_reward_scaling
        self.use_reward_norm: bool = opt.use_reward_norm
        self.use_advantage_norm: bool = opt.use_advantage_norm
        self.use_advantage_clip: bool = opt.use_advantage_clip
        self.use_critic_loss_clip: bool = opt.use_critic_loss_clip
        self.use_policy_loss_clip: bool = opt.use_policy_loss_clip
        self.use_ppo_pretrain_loss: bool = opt.use_ppo_pretrain_loss

        self.running = RunningMoments(accelerator)
        
        self.model = RLHFTrainableModelWrapper(policy_model=policy_model, critic_model=critic_model)
        self.accelerator = accelerator
        
        self.optimizer = self.build_optimizer()
        self.scheduler = optim.lr_scheduler.LambdaLR(
                                optimizer=self.optimizer, 
                                lr_lambda=self.invsqrt_scheduler(self.opt.warmup_steps)
                                )
        self.train_metrics = self.build_metrics('train')
        self.valid_metrics = self.build_metrics('eval')
        self.tokenizer = get_tokenizer(opt)
        
        self.train_state = TrainState()
        self.max_steps: int = opt.train_steps
        self.save_per_step = opt.save_per_step
        self.model_save_path = opt.model_save_path
        
        self.replay_buffer = []
        self.train_loader = None
        self.prompt_loader = DataLoader(
                                OnlyPromptDataset(self.opt, self.accelerator, mode='train'), 
                                batch_size=None, 
                                num_workers=self.opt.num_workers, 
                                prefetch_factor=self.opt.num_prefetch, 
                                pin_memory=True)
        self.pretrain_loader = None
        if self.use_ppo_pretrain_loss:
            self.pretrain_loader = iter(DataLoader(
                                    self.pretrain_dataset_class()(self.opt, self.accelerator), 
                                    batch_size=None, 
                                    num_workers=self.opt.num_workers, 
                                    prefetch_factor=self.opt.num_prefetch, 
                                    pin_memory=True))
        
        self.train_size = len(self.prompt_loader.dataset)
        self.prompt_loader = iter(self.prompt_loader)
        
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(self.model, self.optimizer, self.scheduler)
        
        # get unwrapped trainable model
        self.policy_model = self.accelerator.unwrap_model(self.model).policy_model
        self.critic_model = self.accelerator.unwrap_model(self.model).critic_model

        # get untrainable model
        eval_ds_config = get_eval_ds_config(offload=True)
        self.reward_model, *_ = deepspeed.initialize(model=reward_model, config=eval_ds_config)
        self.reward_model.eval()
        self.ref_model, *_ = deepspeed.initialize(model=ref_model, config=eval_ds_config)
        self.ref_model.eval()

        self.ppl_loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        self.PAD_TOKEN_LABEL_ID = self.ppl_loss_fct.ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction='none', label_smoothing=0.)
            
        synchronize_if_distributed()

    def build_metrics(self, mode='train'):
        metrics = Metrics(self.opt, mode=mode, accelerator=self.accelerator)

        metrics.create_metric('loss', MeanMetric())
        metrics.create_metric('rewards', MeanMetric())
        metrics.create_metric('res_len', MeanMetric())
        metrics.create_metric('ppl', PPLMetric())
        metrics.create_metric('ppl_policy0', PPLMetric())
        metrics.create_metric('ups', MeanMetric())
        metrics.create_metric('global_exs', SumMetric())
        metrics.create_metric('lr', RealtimeMetric())

        if mode == 'train':
            metrics.create_metric('reward_mean', MeanMetric())
            metrics.create_metric('reward_std', MeanMetric())
            
            metrics.create_metric('approx_kl', MeanMetric())
            metrics.create_metric('ref_kl', MeanMetric())
            metrics.create_metric('values', MeanMetric())
            metrics.create_metric('values_clipped', MeanMetric())
            metrics.create_metric('returns', MeanMetric())
            metrics.create_metric('advantages', MeanMetric())
            metrics.create_metric('ratio', MeanMetric())
            metrics.create_metric('pg_clip', MeanMetric())
            metrics.create_metric('vf_clip', MeanMetric())
            metrics.create_metric('pg_loss', MeanMetric())
            metrics.create_metric('vf_loss', MeanMetric())
            metrics.create_metric('entro_loss', MeanMetric())
            if self.use_ppo_pretrain_loss:
                metrics.create_metric('ppo_pretrain_loss', MeanMetric())
                metrics.create_metric('token_acc', MeanMetric())
        return metrics

    def invsqrt_scheduler(self, warmup_steps):
        def _invsqrt_lr(step):
            return math.sqrt(warmup_steps) / math.sqrt(max(warmup_steps, step))
        def _warmup_lr(step):
            return max(step / warmup_steps, 0.1)
        def _invsqrt_lr_with_warmup(step):
            return max(_warmup_lr(step) if step < warmup_steps else _invsqrt_lr(step), 1e-8)
        
        return _invsqrt_lr_with_warmup

    def get_parms(self, model, submodel_name, lr, eps):
        params = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if (not any(nd in n
                                for nd in ["bias", "LayerNorm.weight"]) and p.requires_grad and submodel_name in n)
                ],
                "weight_decay": 0.0,
                "lr": lr,
                "eps": eps,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if (any(nd in n
                            for nd in ["bias", "LayerNorm.weight"]) and p.requires_grad and submodel_name in n)
                ],
                "weight_decay": 0.0,
                "lr": lr,
                "eps": eps,
            },
        ]
        return params

    def build_optimizer(self):
        params = self.get_parms(self.model, 'policy_model.', self.opt.lr, self.opt.eps)
        params.extend(self.get_parms(self.model, 'critic_model.', self.opt.critic_lr, 1e-8))

        deepspeed_states = AcceleratorState().deepspeed_plugin
        if deepspeed_states.deepspeed_config['zero_optimization']['offload_optimizer']['device'] in ('none', None):
            return optim.AdamW(params, eps=self.opt.eps, betas=(self.opt.beta1, self.opt.beta2))
        return DeepSpeedCPUAdam(params, eps=self.opt.eps, betas=(self.opt.beta1, self.opt.beta2))

    def strip_pad_token_id(self, seq: List[int]):
        return [tok for tok in seq if tok != self.tokenizer.pad_token_id]

    def pretrain_dataset_class(self):
        if self.ppo_pretrain_data_type == 'sft':
            return PPOSFTDataset
        elif self.ppo_pretrain_data_type == 'pretrain':
            # TODO: pretrain loss for llama pretrain dataset.
            return PPOSFTDataset
        else:
            raise ValueError
    
    def reward_model_forward(self, inputs, **kwargs):
        return self.reward_model(decoder_input=inputs, **kwargs)
    
    def policy_model_forward(self, inputs, **kwargs):
        return self.policy_model(decoder_input=inputs, **kwargs)
    
    def ref_model_forward(self, inputs, **kwargs):
        return self.ref_model(decoder_input=inputs, **kwargs)
    
    def critic_model_forward(self, inputs, **kwargs):
        return self.critic_model(decoder_input=inputs, only_last=False, **kwargs)
    
    def RLHF_model_forward(self, batch: Dict[str, Any], **kwargs):
        return self.model(batch['text_vec'], **kwargs)
    
    def concat_context_and_response(self, context: List[List[int]], responses: List[List[Tuple[float, List[int]]]]):
        assert len(context) == len(responses), f'Size not match: {len(context)} and {len(responses)}'
        total_context, total_response = [], []
        for _context, _response in zip(context, responses):
            _context = self.strip_pad_token_id(_context)
            for _, resp in _response:
                resp = self.strip_pad_token_id(resp)
                if resp[-1] != self.tokenizer.eos_token_id:
                    logging.warn(f'Generated response is too long: {self.tokenizer.decode(_context + resp, skip_special_tokens=False)}')

                total_context.append(_context.copy())
                total_response.append(resp)

                # Debug
                # logging.info(f'===={self.tokenizer.decode(_context + resp, skip_special_tokens=False)}')
                
        total_gene_samples_vec = [c + r for c, r in zip(total_context, total_response)]
        return total_context, total_response, total_gene_samples_vec # total_context, total_response, total_gene_samples_vec
    
    def save_checkpoint(self, is_best: bool, total_steps: int):
        best_model_path = os.path.join(self.model_save_path, 'best_model')
        steps_model_path = os.path.join(self.model_save_path, 'Steps_{}'.format(total_steps))

        unwrapped_model = self.policy_model
        state_dict = self.accelerator.get_state_dict(unwrapped_model)

        if is_best:
            unwrapped_model.save_pretrained(
                best_model_path,
                is_main_process=self.accelerator.is_main_process,
                save_function=self.accelerator.save,
                state_dict=state_dict,
            )
            logging.info(f'Saved best model to {best_model_path}.')

        unwrapped_model.save_pretrained(
            steps_model_path,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
            state_dict=state_dict,
        )
        logging.info(f'Saved model of {total_steps} steps to {steps_model_path}.')
        
        synchronize_if_distributed()
        
    @torch.no_grad()
    def make_experiences(self):
        start_time = time.time()
        self.model.eval()
        synchronize_if_distributed()
        while len(self.replay_buffer) < self.num_rollouts:
            # get a batch from generator
            batch: Dict[str, Any] = next(self.prompt_loader)
            to_cuda(batch)
            context_vec = batch['text_vec'].tolist()
            
            # sample from env
            _, responses_vec = self.policy_model.generate(batch)
            assert len(context_vec) == len(responses_vec)
            
            context_vec_sampled, resp_vec_sampled, sampled_vec = self.concat_context_and_response(context_vec, responses_vec)
            sampled_vec = torch.tensor(pad_sequences(sampled_vec, pad_value=self.tokenizer.pad_token_id, padding='left'), 
                                       dtype=torch.long, device=self.accelerator.device)
            bsz = sampled_vec.size(0)
            
            rewards, *_ = self.reward_model_forward(sampled_vec)
            rewards = rewards.cpu()
            self.train_metrics.record_metric_many('rewards', rewards.tolist())
            
            if self.use_reward_scaling:
                # Reward scaling
                rewards_mean, rewards_std = self.running.update(rewards)
                if self.use_reward_norm:
                    rewards = (rewards - self.running.mean) / self.running.std
                else:
                    rewards /= self.running.std # do not -= mean since advantage will be normalized again
                logging.info(f"Running mean: {self.running.mean}, std: {self.running.std}")
                self.train_metrics.record_metric('reward_mean', rewards_mean)
                self.train_metrics.record_metric('reward_std', rewards_std)
                
            if self.use_reward_clip:
                # Reward clip
                rewards = torch.clip(rewards, -self.reward_clip, self.reward_clip)
                
            # Precompute logprobs, values
            ref_logits, *_ = self.ref_model_forward(sampled_vec)
            logits, *_ = self.policy_model_forward(sampled_vec)
            values, *_ = self.critic_model_forward(sampled_vec)
            torch.cuda.empty_cache()
            assert ref_logits.size(1) == logits.size(1) == values.size(1), f'{ref_logits.size()}, {logits.size()}, {values.size()}'
            
            ref_logprobs = logprobs_from_logits(ref_logits[:, :-1, :], sampled_vec[:, 1:])
            logprobs = logprobs_from_logits(logits[:, :-1, :], sampled_vec[:, 1:])
            values = values[:, :-1]
            
            kl_penalty = (-self.kl_penalty_weight * (logprobs - ref_logprobs)).cpu()

            # compute train ppl
            label = sampled_vec
            label[label == self.tokenizer.pad_token_id] = self.PAD_TOKEN_LABEL_ID
            shift_label = label[:, 1:].contiguous()
            valid_length = (shift_label != self.PAD_TOKEN_LABEL_ID).sum(dim=-1)
            
            # compute ppl
            shift_logits = logits[..., :-1, :].contiguous()
            ppl_value = self.ppl_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_label.view(-1))
            ppl_value = ppl_value.view(len(logits), -1)
            ppl_value = torch.sum(ppl_value, -1) / valid_length
            ppl_value = ppl_value.cpu().tolist()

            # compute ppl for policy0
            shift_ref_logits = ref_logits[..., :-1, :].contiguous()
            ppl0_value = self.ppl_loss_fct(shift_ref_logits.view(-1, shift_ref_logits.size(-1)), shift_label.view(-1))
            ppl0_value = ppl0_value.view(len(ref_logits), -1)
            ppl0_value = torch.sum(ppl0_value, -1) / valid_length
            ppl0_value = ppl0_value.cpu().tolist()
            
            logging.info(f'ppl_value: {ppl_value}')
            logging.info(f'ppl0_value: {ppl0_value}')
            
            # gather samples
            for i in range(bsz):
                resp_length = len(resp_vec_sampled[i])
                penalized_rewards = kl_penalty[i].clone()
                penalized_rewards[-1] += rewards[i]
                self.train_metrics.record_metric('ref_kl', (logprobs[i][-resp_length:] - ref_logprobs[i][-resp_length:]).mean().item())
                
                sample = {
                    'context_vec': context_vec_sampled[i],
                    'context': self.tokenizer.decode(context_vec_sampled[i], skip_special_tokens=False),
                    'resp_vec': resp_vec_sampled[i],
                    'resp': self.tokenizer.decode(resp_vec_sampled[i], skip_special_tokens=False),
                    'reward': penalized_rewards[-resp_length:].tolist(),
                    'values': values[i][-resp_length:].tolist(),
                    'ref_logprobs': ref_logprobs[i][-resp_length:].tolist(),
                    'logprobs': logprobs[i][-resp_length:].tolist(),
                    'ppl_value': ppl_value[i],
                    'ppl0_value': ppl0_value[i]
                }

                # get pretrain batch
                if self.use_ppo_pretrain_loss:
                    ppo_batch: Dict[str, Any] = next(self.pretrain_loader) # nums: opt.ppo_pretrain_batch_size_ratio
                    to_cuda(ppo_batch)
                    sample['ppo_context_vec'] = ppo_batch['text_vec'].tolist()
                    sample['ppo_loss_mask'] = ppo_batch['loss_mask'].tolist()

                self.replay_buffer.append(sample)
                
        logging.info(f'Sampled {len(self.replay_buffer)} samples in {(time.time() - start_time):.2f} seconds')
        self.model.train()
        
    def criterion(self, model_output: Tuple[torch.Tensor, ...], batch: Dict[str, Any], return_output=False, training=True):
        policy_output, critic_output = model_output
        policy_logits, *_ = policy_output
        values, *_ = critic_output
        values = values[:, :-1]
        
        loss_mask = batch['loss_mask']
        loss_mask = loss_mask[:, 1:]
        old_values = batch['values']
        old_logprobs = batch['logprobs']
        advantages = batch['advantages']
        returns = batch['returns']
        if self.use_advantage_norm:
            # advantage norm
            advantages = whiten(advantages, loss_mask, accelerator=self.accelerator)
        if self.use_advantage_clip:
            # advantage clip
            advantages = torch.clamp(advantages, -self.advantage_clip, self.advantage_clip)
        n = loss_mask.sum()

        
        logprobs = logprobs_from_logits(policy_logits[:, :-1, :], batch['text_vec'][:, 1:]) * loss_mask
        
        # vf loss
        values_clipped = torch.clamp(
            values,
            old_values - self.value_clip,
            old_values + self.value_clip,
        )
        vf_loss1 = (values - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2

        # critic model loss clip
        if self.use_critic_loss_clip:
            vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * loss_mask) / n
        else:
            vf_loss = 0.5 * torch.sum(vf_loss1 * loss_mask) / n

        vf_clipfrac = torch.sum((vf_loss2 > vf_loss1).float() * loss_mask) / n
        
        log_ratio = (logprobs - old_logprobs) * loss_mask
        ratio = torch.exp(log_ratio)
        with torch.no_grad():
            approx_kl = torch.sum((ratio - 1) - log_ratio) / n
            
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio,
            1.0 - self.pg_clip,
            1.0 + self.pg_clip,
        )
        # policy model loss clip
        if self.use_policy_loss_clip:
            pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * loss_mask) / n
        else:
            pg_loss = torch.sum(pg_loss1 * loss_mask) / n
        pg_clipfrac = torch.sum((pg_loss2 > pg_loss1).float() * loss_mask) / n

        # cal the entropy
        if self.use_entropy_loss:
            ent = get_category_distribution_entropy(len(policy_logits), policy_logits[:, :-1, :])
            entro_loss = torch.abs(torch.sum(ent * loss_mask) / n - self.entropy_clip)

        # cal pretrain loss
        if self.use_ppo_pretrain_loss: 
            pretrain_sampled_vec = batch['ppo_context_vec']

            scores, *_ = self.policy_model_forward(pretrain_sampled_vec)
            scores = scores[:, :-1, :]
            preds = scores.argmax(dim=-1)

            ppo_label_vec = batch['ppo_context_vec'][:, 1:].clone()
            ppo_loss_mask = batch['ppo_loss_mask'][:, 1:]
            ppo_label_vec[~ppo_loss_mask] = self.tokenizer.pad_token_id

            labels: torch.LongTensor = ppo_label_vec
            
            score_view = scores.reshape(-1, scores.size(-1)) # bs * num_tokens, vocab_size
            pretrain_loss = self.loss_fn(score_view, labels.reshape(-1)).sum()
            
            # calculate token acc
            notnull = labels.ne(self.tokenizer.pad_token_id)
            target_tokens = notnull.sum()
            correct = ((labels == preds) * notnull).sum()
            
            # average losses
            pretrain_loss = pretrain_loss / target_tokens

            if self.use_entropy_loss:
                loss1 = pg_loss + self.vf_loss_weight * vf_loss + self.entropy_loss_weight * entro_loss
            else:
                loss1 = pg_loss + self.vf_loss_weight * vf_loss
            loss2 = self.ppo_pretrain_loss_weight * pretrain_loss
            loss = loss1 + loss2
        else:
            if self.use_entropy_loss:
                loss = pg_loss + self.vf_loss_weight * vf_loss + self.entropy_loss_weight * entro_loss
            else:
                loss = pg_loss + self.vf_loss_weight * vf_loss
        
        with torch.no_grad():
            if training:
                obj_metrics = self.train_metrics
            else:
                obj_metrics = self.valid_metrics

            obj_metrics.record_metric('loss', loss.item())
            obj_metrics.record_metric('pg_loss', pg_loss.item())
            obj_metrics.record_metric('vf_loss', vf_loss.item())
            if self.use_entropy_loss:
                obj_metrics.record_metric('entro_loss', entro_loss.item())
            obj_metrics.record_metric('pg_clip', pg_clipfrac.item())
            obj_metrics.record_metric('vf_clip', vf_clipfrac.item())
            obj_metrics.record_metric('approx_kl', approx_kl.item())
            obj_metrics.record_metric('values', (values.mul(loss_mask).sum() / n).item())
            obj_metrics.record_metric('values_clipped', (values_clipped.mul(loss_mask).sum() / n).item())
            obj_metrics.record_metric('advantages', (advantages.mul(loss_mask).sum() / n).item())
            obj_metrics.record_metric('returns', (returns.mul(loss_mask).sum() / n).item())
            obj_metrics.record_metric('ratio', (ratio.mul(loss_mask).sum() / n).item())
            obj_metrics.record_metric('ppl', (batch['ppl_value'].sum() / n).item())
            obj_metrics.record_metric('ppl_policy0', (batch['ppl0_value'].sum() / n).item())
            if self.use_ppo_pretrain_loss:
                obj_metrics.record_metric('ppo_pretrain_loss', pretrain_loss.item())
                obj_metrics.record_metric('token_acc', (correct / target_tokens).item())

        if self.use_ppo_pretrain_loss:
            if return_output:
                return loss1, loss2, model_output
            else:
                return loss1, loss2

        if return_output:
            return loss, model_output

        return loss
    
    def train_step(self, batch: Dict[str, Any], **kwargs):
        self.optimizer.zero_grad()
        # forward
        assert self.model.training
        model_output = self.RLHF_model_forward(batch, **kwargs)

        # compute loss
        loss = self.criterion(model_output, batch)

        if self.use_ppo_pretrain_loss:
            self.accelerator.backward(loss[0])
            self.accelerator.backward(loss[1])
            loss = loss[0] + loss[1]
        else:
            self.accelerator.backward(loss)

        if torch.isnan(loss) or torch.isinf(loss) or loss.abs().gt(10000.):
            logging.warn(f'Strange loss {loss.item()} detected.')

        self.optimizer.step()
        if not self.accelerator.optimizer_step_was_skipped:
            self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, datatype='valid', **kwargs) -> Tuple[float, List]:
        assert datatype in ('valid', 'test')
        start_time = time.time()

        valid_dataloader = DataLoader(
            OnlyPromptDataset(self.opt, self.accelerator, mode=datatype), 
            batch_size=None, 
            num_workers=self.opt.num_workers, 
            prefetch_factor=self.opt.num_prefetch, 
            pin_memory=True)
        
        print_rank_0(f'Start evaluation on {datatype} data.')
        self.model.eval()
        
        for step, batch in enumerate(valid_dataloader):
            to_cuda(batch)
            _, responses = self.policy_model.generate(batch, **kwargs)
            _, _, output_vec = self.concat_context_and_response(batch['text_vec'].tolist(), responses)

            output_vec = torch.tensor(pad_sequences(output_vec, pad_value=self.tokenizer.pad_token_id, padding='left'), 
                                                     dtype=torch.long, device=self.accelerator.device)
            rewards = self.reward_model_forward(output_vec)[0].tolist()
            assert len(rewards) == output_vec.size(0), f"{rewards.size()}, {output_vec.size()}"
            self.valid_metrics.record_metric_many('rewards', rewards)

            # compute ppl
            ppl_logits, *_ = self.policy_model_forward(output_vec)
            ppl_ref_logits, *_ = self.ref_model_forward(output_vec)

            label = output_vec
            label[label == self.tokenizer.pad_token_id] = self.PAD_TOKEN_LABEL_ID
            shift_label = label[:, 1:].contiguous()
            valid_length = (shift_label != self.PAD_TOKEN_LABEL_ID).sum(dim=-1)

            # compute ppl
            shift_logits = ppl_logits[..., :-1, :].contiguous()
            ppl_value = self.ppl_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_label.view(-1))
            ppl_value = ppl_value.view(len(ppl_logits), -1)
            ppl_value = torch.sum(ppl_value, -1) / valid_length

            # compute ppl for policy 0
            shift_ref_logits = ppl_ref_logits[..., :-1, :].contiguous()
            ppl0_value = self.ppl_loss_fct(shift_ref_logits.view(-1, shift_ref_logits.size(-1)), shift_label.view(-1))
            ppl0_value = ppl0_value.view(len(ppl_ref_logits), -1)
            ppl0_value = torch.sum(ppl0_value, -1) / valid_length

            self.valid_metrics.record_metric_many('ppl', ppl_value.cpu().tolist())
            self.valid_metrics.record_metric_many('ppl_policy0', ppl0_value.cpu().tolist())
            
        # log info
        metrics = self.valid_metrics.all_gather_metrics()
        self.valid_metrics.display(self.train_state.total_steps, gathered_metrics=metrics)
        self.valid_metrics.write_tensorboard(self.train_state.total_steps, gathered_metrics=metrics)
        self.valid_metrics.flush()
        validation_score = metrics['rewards']
        self.valid_metrics.reset(no_reset=[])
        
        print_rank_0(f'Evaluation completed in {(time.time() - start_time):.2f} seconds.')
        self.model.train()
        torch.cuda.empty_cache()
        return validation_score, None

    def train(self):
        eval_score, _ = self.evaluate()
        self.train_state.best_score = eval_score

        synchronize_if_distributed()
        print_rank_0('Start training.')
        self.model.train()
        
        while self.train_state.total_steps < self.max_steps:
            self.make_experiences()
            self.train_loader = DataLoader(
                ExperienceDataset(self.replay_buffer, self.opt, self.accelerator), 
                batch_size=None, 
                num_workers=self.opt.num_workers, 
                prefetch_factor=self.opt.num_prefetch, 
                pin_memory=True)

            for batch in self.train_loader:
                if self.train_state.total_steps >= self.max_steps:
                    break
                
                start_time = time.time()

                with torch.no_grad():
                    batchsize = batch.get('n_exps', batch['text_vec'].size(0))
                    self.train_metrics.record_metric_many('res_len', batch['res_len'])
                    self.train_metrics.record_metric('global_exs', batchsize)
                    self.train_state.total_exps += batchsize

                to_cuda(batch)
                # perform a step of train
                self.train_step(batch)
                del batch
                
                # record
                cost_time = time.time() - start_time
                self.train_metrics.record_metric('ups', 1. / cost_time)
                if hasattr(self.scheduler, 'get_last_lr'):
                    lr = self.scheduler.get_last_lr()[0]
                else:
                    lr = self.optimizer.param_groups[0]['lr']
                self.train_metrics.record_metric('lr', lr)
                self.train_state.total_steps += 1
                
                # print metrics
                need_reset = False
                if self.train_state.total_steps % self.print_interval == 0:
                    metrics = self.train_metrics.all_gather_metrics()
                    self.train_metrics.write_tensorboard(self.train_state.total_steps, gathered_metrics=metrics)
                    self.train_metrics.display(self.train_state.total_steps, self.train_size, gathered_metrics=metrics)
                    need_reset = True
                    
                # do evaluation for every save_per_step steps
                if self.train_state.total_steps % self.save_per_step == 0:
                    eval_score, _ = self.evaluate()
                    self.model.train()
                        
                    # save checkpoint
                    is_best = eval_score > self.train_state.best_score
                    if is_best:
                        self.train_state.best_score = eval_score
                        print_rank_0(f'Greater than the best score {abs(eval_score)}.')
                    else:
                        print_rank_0(f'Did not beat the best score {abs(self.train_state.best_score)}.')
                        
                    self.save_checkpoint(is_best=is_best, total_steps=self.train_state.total_steps)
        
                if need_reset:
                    self.train_metrics.reset(no_reset=self.no_reset_metric_names)

            synchronize_if_distributed()
            self.train_loader = None
            self.replay_buffer.clear()
            torch.cuda.empty_cache()
