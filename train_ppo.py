import time
import math
import random
import logging
from typing import List
import numpy as np
import torch
import torch.nn as nn
from config_ppo import parse_args
from ppo.ppo_trainer import PPOTrainer
from ppo.ppo_datahelper import get_tokenizer
from utils import *
from transformers.models.llama.modeling_llama import LlamaForCausalLM

class Llama(LlamaForCausalLM):
    def __init__(self, config, opt, tokenizer):
        super().__init__(config)
        self.opt = opt
        self.tokenizer = tokenizer
        
    def forward(self, decoder_input, incr_state=None):

        attention_mask = decoder_input.ne(self.tokenizer.pad_token_id)
        if incr_state is not None:
            decoder_input = decoder_input[:, -1:]
            
        output = super().forward(
            input_ids=decoder_input,
            attention_mask=attention_mask,
            past_key_values=incr_state,
            return_dict=True,
            use_cache=not self.training
            )
        
        logits = output.logits
        new_incr_states = output.past_key_values
        
        return logits, new_incr_states

    @torch.no_grad()
    def generate(self, batch, **kwargs):
        """
        Generate response
        """        
        maxlen_res = kwargs.pop('maxlen_res', self.opt.maxlen_res)
        temperature = kwargs.pop('temperature', self.opt.temperature)
        repetition_penalty = kwargs.pop('repetition_penalty', self.opt.repetition_penalty)
        topp = kwargs.pop('topp', self.opt.topp)

        decoder_input: torch.LongTensor = batch['text_vec'] # (bsz, ...)
        assert decoder_input[:, -1].ne(self.tokenizer.pad_token_id).all(), 'Last token should not be a padding token (you can use left padding instead).'
            
        dev = decoder_input.device
        bsz = decoder_input.size(0)

        scores = torch.zeros((bsz,), device=dev, dtype=torch.float16)
        done = torch.zeros((bsz,), device=dev).to(torch.bool)

        inds = torch.arange(bsz).to(dev).unsqueeze(1).view(-1)
        decoder_input = torch.index_select(decoder_input, 0, inds)
        init_length = decoder_input.size(1)
            
        incr_state = None
        for _token in range(maxlen_res):
            if done.all():
                break
            score, incr_state, *_ = self.forward(decoder_input, incr_state)
            score = score.half()

            # now score is bs, len, vocab_size
            score = score[:, -1, :]
                
            # calculate repetition penalty
            if repetition_penalty > 1.:
                penalty_tokens = decoder_input[:, init_length:]
                penalty_scores = torch.gather(score, dim=1, index=penalty_tokens)
                penalty_scores = torch.where(penalty_scores < 0., penalty_scores * repetition_penalty, penalty_scores / repetition_penalty)
                score = score.scatter_(dim=1, index=penalty_tokens, src=penalty_scores)

            # nucleus sampling
            score = torch.softmax(score.div(temperature), dim=-1)
            probs = top_p_logits(score, topp=topp, filter_value=0)
            tok_ids = torch.multinomial(probs, 1)[:, 0]
            hyp_ids = torch.arange(probs.size(0), device=dev)
            scores = scores + probs[hyp_ids, tok_ids].log() * ~done

            tok_ids = torch.where(done, self.tokenizer.pad_token_id, tok_ids)
            decoder_input = torch.cat((decoder_input, tok_ids.unsqueeze(-1)), dim=-1)
            done = done | tok_ids.eq(self.tokenizer.eos_token_id)

            incr_state = self._reorder_cache(incr_state, hyp_ids)

        # get all finalized candidates for each sample
        decoder_input = decoder_input[:, init_length:]
        decoder_input = decoder_input.view(bsz, -1)
        scores = scores.view(bsz, )

        lengths = decoder_input.ne(self.tokenizer.pad_token_id).sum(dim=-1)

        length_penalty = torch.pow(lengths, 1.0)
        scores /= length_penalty

        preds_scores = []
        for i in range(bsz):
            seq: torch.LongTensor = decoder_input[i, :lengths[i, ]]
            res_scores = (float(scores[i, ]), seq.tolist())
            preds_scores.append([res_scores])

        best_preds_scores = [preds[0] for preds in preds_scores]
        return best_preds_scores, preds_scores


class LlamaRewardModel(LlamaForCausalLM):
    def __init__(self, config, opt, tokenizer):
        super().__init__(config)
        self.opt = opt
        self.tokenizer = tokenizer
        self.reward_head = torch.nn.Linear(config.hidden_size, 1, bias=False)
        
    def forward(self, decoder_input, only_last=True):
        attention_mask = decoder_input.ne(self.tokenizer.pad_token_id)
        output = self.model.forward(
            input_ids=decoder_input,
            attention_mask=attention_mask, 
            return_dict=True,
            use_cache=False
            )
        
        if only_last:
            logits = self.reward_head(output.last_hidden_state[:, -1, :]).squeeze(-1)
        else:
            logits = self.reward_head(output.last_hidden_state).squeeze(-1)
        
        return (logits,)
    

def main(opt):
    # setup accelerator
    accelerator = setup_accelerator()

    # setup deepspeed
    deepspeed_states = AcceleratorState().deepspeed_plugin
    deepspeed_states.deepspeed_config['train_micro_batch_size_per_gpu'] = opt.batch_size
    deepspeed_states.deepspeed_config['checkpoint'] = {'use_node_local_storage': True}

    # logging config
    logging.basicConfig(
            format='%(asctime)s - ' + f'Rank: {accelerator.process_index}' + ' - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level=logging.INFO
            )
    logger = logging.getLogger(__name__)

    # fix seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    
    # tokenizer
    tokenizer = get_tokenizer(opt)

    # load policy model
    logging.info(f"Loading policy model from: {opt.policy_model_path}...")
    policy_model = Llama.from_pretrained(opt.policy_model_path, opt, tokenizer)
    policy_model._set_gradient_checkpointing(policy_model.model, opt.gradient_checkpoint)

    # load critic model
    logging.info(f"Loading critic model from: {opt.critic_model_path}...")
    critic_model = LlamaRewardModel.from_pretrained(opt.critic_model_path, opt, tokenizer)
    critic_model._set_gradient_checkpointing(critic_model.model, opt.gradient_checkpoint)

    # load reference model
    logging.info(f"Loading reference model from: {opt.policy_model_path}...")
    ref_model = Llama.from_pretrained(opt.policy_model_path, opt, tokenizer)

    # load reward model
    logging.info(f"Loading reward model from: {opt.critic_model_path}...")
    reward_model = LlamaRewardModel.from_pretrained(opt.critic_model_path, opt, tokenizer)

    synchronize_if_distributed()
    trainer = PPOTrainer(opt, policy_model, ref_model, critic_model, reward_model, accelerator)
    trainer.train()

    logging.info('==================Congrats! Training completed, exit process...==================') 

if __name__ == '__main__':
    opt = parse_args()
    print_rank_0(opt)
    main(opt)