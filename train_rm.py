from transformers.models.llama.modeling_llama import LlamaForCausalLM

from typing import Dict, Any, Union, List, Tuple
import os
import random
import numpy as np

from config_rm import parse_args
from utils import *
from rm.reward_trainer import RewardTrainer
from rm.reward_datahelper import get_tokenizer

class LlamaRewardModel(LlamaForCausalLM):
    def __init__(self, config, opt: Dict[str, Any], tokenizer, **kwargs):
        super().__init__(config)
        self.opt = opt
        self.tokenizer = tokenizer
        
        self.end_idx = tokenizer.eos_token_id
        self.NULL_IDX = tokenizer.pad_token_id
        
        self.reward_head = torch.nn.Linear(config.hidden_size, 1, bias=False)
        self.calculate_lm_loss: bool = getattr(opt, 'reward_lm_loss_factor', 0.) > 0.
        self.post_init()
        
    def forward(self, decoder_input: torch.LongTensor, rank_all=False):
        if not (rank_all or decoder_input[:, -1].eq(self.end_idx).all()):
            logging.warn(f'Found sample that NOT ended with EOS token')
        
        attention_mask = decoder_input.ne(self.NULL_IDX)
        output = self.model.forward(input_ids=decoder_input, attention_mask=attention_mask, 
                                          return_dict=True, use_cache=False)
        if not rank_all:
            logits = output.last_hidden_state[:, -1, :]
            logits = self.reward_head(logits).squeeze(-1)
        else:
            logits = self.reward_head(output.last_hidden_state).squeeze(-1)
     
        if self.calculate_lm_loss:
            lm_logits = self.lm_head(output.last_hidden_state)
            return logits, lm_logits
        else:
            return (logits,)

def main(opt):
    # setup accelerator
    accelerator = setup_accelerator()

    # setup deepspeed
    deepspeed_states = AcceleratorState().deepspeed_plugin
    deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"] = opt.batch_size
    deepspeed_states.deepspeed_config["checkpoint"] = {"use_node_local_storage": True}

    # logging config
    if accelerator and accelerator.use_distributed:
        logging.basicConfig(
            format="%(asctime)s - "
            + f"Rank: {accelerator.process_index}"
            + " - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
        )
    else:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.DEBUG,
        )
    logger = logging.getLogger(__name__)

    # fix seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    
    # load initial reward model, if init_checkpoint_model is specified, load the specified model, otherwise load the pre-trained model
    if opt.init_checkpoint_model and os.path.isdir(opt.init_checkpoint_model):
        logging.info(f"Load existing model from {opt.init_checkpoint_model}")
        model = LlamaRewardModel.from_pretrained(
            opt.init_checkpoint_model, opt, get_tokenizer(opt)
        )
    else:
        logging.info(f"Load **init** model from {opt.hf_model_name_or_path}")
        model = LlamaRewardModel.from_pretrained(
            opt.hf_model_name_or_path, opt, get_tokenizer(opt)
        )

    # set gradient checkpointing
    model._set_gradient_checkpointing(model.model, opt.gradient_checkpoint)

    synchronize_if_distributed()
    
    # init reward trainer and start training
    trainer = RewardTrainer(opt, model, accelerator)
    trainer.train()
    
    logging.info('==================Congrats! Training completed, exit process...==================') 

if __name__ == "__main__":
    opt = parse_args()
    print_rank_0(opt)
    main(opt)
