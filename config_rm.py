import argparse

def parse_args(*args):
    parser = argparse.ArgumentParser(description='MOSS-RLHF Reward Model @Fudan NLP Group')
    # training settings
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--lr', type=float, default=5e-6, help='learning rate of reward model')
    parser.add_argument('--batch_size', type=int, default=8, help='training batch size for single GPU')
    parser.add_argument('--gradient_checkpoint', action='store_true', help='deepspeed')
    parser.add_argument('--reward_lm_loss_factor', type=float, default=0., help='calculate lm loss on rm model')
    parser.add_argument('--warmup_steps', type=int, default=500, help='warmup steps')
    parser.add_argument('--train_steps', type=int, default=10000, help='train steps')
    parser.add_argument('--fp32_loss', action='store_true', help='use fp32 to calculate cross-entropy loss, enable when numeric stability problem occurs')
    parser.add_argument('--save_per_step', type=int, default=200, help='save ckpt and save validation tensorboard per steps')
    parser.add_argument('--print_interval', type=int, default=5, help='print training state and save training tensorboard per steps')
    parser.add_argument('--validation_metric', type=str, default='loss', help='metric to select the best model')
    
    # Optimizer , Scheduler and Dataloader
    parser.add_argument('--beta1', type=float, default=0.9, help='adam')
    parser.add_argument('--beta2', type=float, default=0.95, help='adam')
    parser.add_argument('--eps', type=float, default=1e-6, help='optimizer')
    parser.add_argument('--num_prefetch', type=int, default=32, help='dataloader')
    parser.add_argument('--num_workers', type=int, default=1, help='dataloader')
    parser.add_argument('--weight_decay', type=float, default=0., help='l2 weight decay')
    
    # Path
    parser.add_argument('--data_path', type=str, default='./data', help='dataset for training and validation')
    parser.add_argument('--init_checkpoint_model', type=str, default=None, help='checkpoint used to initialize the model, used for fine-tuning')
    parser.add_argument('--logdir', type=str, default=None, help='path to save tensorboard logs')
    parser.add_argument('--model_save_path', type=str, default='./outputs/', help='checkpoint path, used for save model and training')
    parser.add_argument('--hf_model_name_or_path', type=str, default='meta-llama/Llama-2-7b-hf', help='Hugging model name used to load tokenizer, configs and pretained models')
    
    # LLM settings
    parser.add_argument('--context_truncate', type=int, default=2048, help='max length for history')
    parser.add_argument('--delimiter', type=str, default='\n', help='delimiter to seperate dialog history')
   

    args = parser.parse_args()
    return args
    
    