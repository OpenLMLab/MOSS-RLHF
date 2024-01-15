from torch.utils.data import get_worker_info, IterableDataset
from transformers.models.llama.tokenization_llama import LlamaTokenizer

from typing import Dict, Any, List, Tuple, Union, Generator
import json, logging, torch, random
import os 

from utils import *

def get_human_prompt():
    return "Human:"


def get_assistant_prompt():
    return "Assistant:"

def get_separate_prompt(i: int):
    return get_human_prompt() if i % 2 == 0 else get_assistant_prompt()

def get_tokenizer(opt):
    tokenizer_name_or_path = opt.hf_model_name_or_path
    print_rank_0(f"Loading tokenizer from huggingface: {tokenizer_name_or_path}...", only_on_cuda0=True)
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    tokenizer.bos_token = '<s>'
    tokenizer.eos_token = '</s>'
    tokenizer.pad_token = '<unk>'
    tokenizer.pad_token_id = 0
    tokenizer.unk_token = tokenizer.pad_token
    tokenizer.unk_token_id = tokenizer.pad_token_id

    print_rank_0(f"Llama tokenizer size: {tokenizer.vocab_size}", only_on_cuda0=True)
    print_rank_0(f"Llama tokenizer pad token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}", only_on_cuda0=True)
    print_rank_0(f"Llama tokenizer. special token: {tokenizer.special_tokens_map}", only_on_cuda0=True)

    return tokenizer

class RMDialogDataset(IterableDataset):
    def __init__(self, opt, accelerator, mode: str = 'train', **kwargs) -> None:
        super().__init__()
        
        assert mode in ('train', 'valid')
        self.opt = opt
        self.accelerator = accelerator
        self.tokenizer = get_tokenizer(opt)
        self.mode = mode
        self.data: List[Tuple[List[str], str, str]] = [] # dataset list. (context, chosen, rejected)
        
        # init some params
        self.batch_size = opt.batch_size
        self.c_trunc: int = opt.context_truncate
        
        # load data from opt.data_path, support for suffix loading, format: opt.data_path/xxx{mode}.json
        file_names = sorted([file_name for file_name in os.listdir(opt.data_path) if file_name.endswith(f'{mode}.json')])
        for file_name in file_names:
            data_path = os.path.join(opt.data_path, file_name)
            
            try:
                file_data = self._load_data(data_path)
            except Exception as e:
                logging.warn(f"Load data from {data_path} failed. {str(e)}")
                
            self.data.extend(file_data)
            logging.info(f'Got {len(file_data)} samples from {data_path}')
            
        logging.info(f'Got {len(self.data)} samples totally from {file_names}')

        self.size = len(self.data)
        # get data for current rank on distributed train
        if accelerator and self.accelerator.use_distributed:
            self.data = self.data[self.accelerator.process_index::self.accelerator.num_processes]
        
    def __len__(self):
        return self.size
    
    def __iter__(self):
        data_generator = self.batch_generator()

        for batch_samples in data_generator:
            batch = self._batchify(batch_samples)
            yield batch    
            
    
    def _load_data(self, dpath: str):
        with open(dpath, 'r') as f:
            data: List[Dict[str, List[str]]] = json.load(f)
        output: List[Tuple[List[str], str, str]] = [] # context, chosen, rejected
        error_samples = []
        
        for sample in data:
            chosen, reject = sample['chosen'], sample['rejected']
            
            if len(chosen) != len(reject) or not all(chosen) or not all(reject) or chosen[-1] == reject[-1]:
                error_samples.append((chosen, reject))
                continue
            
            output.append((chosen[:-1], chosen[-1], reject[-1]))
            
        if error_samples:
            logging.warn(f'Detected {len(error_samples)} illegal samples')
            logging.warn(f'Examples: {error_samples[:5]}')
            
        del data, error_samples
        return output


    def _build_prompt(self, context: List[str], dialog_sep='\n'):
        human_prompt, assistant_prompt = get_human_prompt(), get_assistant_prompt()
        if context[-1].startswith(human_prompt):
            end_prompt = assistant_prompt
        elif context[-1].startswith(assistant_prompt):
            end_prompt = human_prompt
        else:
            logging.critical(context)
            raise ValueError
        
        context = dialog_sep.join(context)

        return f"{context}{dialog_sep}{end_prompt}"

    def _encode_sample(self, sample: Tuple[List[str], str, str]) -> Dict[str, Any]:
        context, chosen, rejected = sample
        
        # context's last utterance is from assistant
        # if len(content) % 2 == 0, means the history is truncated and the first utterance is from assistant
        context = [get_separate_prompt(i + (len(context) + 1) % 2) + s for i, s in enumerate(context)] 
        chosen_vec = self.tokenizer.encode(chosen, add_special_tokens=True) + [self.tokenizer.eos_token_id]
        rejected_vec = self.tokenizer.encode(rejected, add_special_tokens=True) + [self.tokenizer.eos_token_id]
        context_vec = self.tokenizer.encode(self._build_prompt(context, dialog_sep=self.opt.delimiter), 
                                            add_special_tokens=True) 
        
        
        label_len = max(len(chosen_vec), len(rejected_vec))
        text_len = len(context_vec) + label_len
        
        # truncate
        while len(context_vec) + label_len > self.c_trunc and len(context) > 1:
            context = context[1:]
            context_vec = self.tokenizer.encode(self._build_prompt(context, dialog_sep=self.opt.delimiter),
                                                add_special_tokens=True)
            
        chosen_sample_vec = context_vec + chosen_vec
        rejected_sample_vec = context_vec + rejected_vec
        
        # lm loss mask for chosen
        loss_mask = [0] * len(context_vec) + [1] * len(chosen_vec)

        output = {
            'text': sample,
            'text_len': text_len,
            'label_len': label_len,
            'loss_mask': loss_mask,
            'chosen_vec': chosen_sample_vec,
            'rejected_vec': rejected_sample_vec,
        }
        return output
    
    def _batchify(self, batch_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_text_vec = [sample['chosen_vec'] for sample in batch_samples] + [sample['rejected_vec'] for sample in batch_samples] 
        batch_text_vec = torch.tensor(pad_sequences(batch_text_vec, pad_value=self.tokenizer.pad_token_id, padding='left'), dtype=torch.long)
        loss_mask = torch.tensor(pad_sequences([sample['loss_mask'] for sample in batch_samples], pad_value=0, padding='left', pad_to=batch_text_vec.size(1)), dtype=torch.bool)
        assert batch_text_vec.size(1) == loss_mask.size(1) and batch_text_vec.size(0) == loss_mask.size(0) * 2, f'Illegal text and mask size of {batch_text_vec.size()} and {loss_mask.size()}'
        
        batch = {
            'text_vec': batch_text_vec,
            'loss_mask': loss_mask,
            'text_len': [sample['text_len'] for sample in batch_samples],
            'n_tokens': sum(len(sample['chosen_vec']) + len(sample['rejected_vec']) for sample in batch_samples),
            'text_trunc': [1 if sample['text_len'] > max(len(sample['chosen_vec']), len(sample['rejected_vec'])) else 0 for sample in batch_samples],
            'n_exps': len(batch_samples),
        }
        
        return batch

    def sample_generator(self):
        need_shuffle = self.mode == 'train'
        
        # if multiprocessing dataloader is used, split dataset for each worker
        worker_info = get_worker_info()
        if worker_info is not None:
            self.data = self.data[worker_info.id::worker_info.num_workers]
            logging.info(f'WORKER {worker_info.id} Got {len(self.data)} samples')
            
        if need_shuffle:
            random.shuffle(self.data)
        # yield samples
        for sample in self.data:
            yield self._encode_sample(sample)
            
    def batch_generator(self) -> Generator[List[Dict[str, Any]], None, None]:
        max_len = self.c_trunc
        min_len = 1
        batch: List[Dict[str, Any]] = []

        for sample in self.sample_generator():
            # skip too long or empty samples
            sample_len = max(len(sample['chosen_vec']), len(sample['rejected_vec']))
            if not (min_len <= sample_len <= max_len):
                logging.warn(f'Found sample with length of {sample_len} which > {max_len} or < {min_len}, skipped')
                continue
            
            # add legal samples to batch
            batch.append(sample)
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]
        if batch:
            yield batch
            

    