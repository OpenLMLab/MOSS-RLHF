import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='MOSS-RLHF @Fudan NLP Group')

    # Path
    parser.add_argument('--model_save_path', type=str, default='', help='checkpoint path, used for save model and training')
    parser.add_argument('--policy_model_path', type=str, default='', help='policy model and reference model path')
    parser.add_argument('--critic_model_path', type=str, default='', help='critic model and reward model path')
    parser.add_argument('--tokenizer_name_or_path', type=str, default='/huggingface_models/open-chinese-llama-7b', help='tokenizer name or path')
    parser.add_argument('--data_path', type=str, default='./data', help='dataset for training and validation')
    parser.add_argument('--logdir', type=str, default=None, help='path to save tensorboard logs')

    # Training
    parser.add_argument('--lr', type=float, default=5e-7, help='learning rate of policy model')
    parser.add_argument('--critic_lr', type=float, default=15e-7, help='learning rate of critic model')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size, *NOT* for sampling from env')
    parser.add_argument('--train_steps', type=int, default=5000, help='train steps')
    parser.add_argument('--warmup_steps', type=int, default=500, help='warmup steps')
    parser.add_argument('--save_per_step', type=int, default=100, help='save ckpt per steps')
    parser.add_argument('--beta1', type=float, default=0.9, help='adam')
    parser.add_argument('--beta2', type=float, default=0.95, help='adam')
    parser.add_argument('--eps', type=float, default=1e-6, help='optimizer')
    parser.add_argument('--num_workers', type=int, default=1, help='dataloader')
    parser.add_argument('--num_prefetch', type=int, default=32, help='dataloader')
    parser.add_argument('--maxlen_prompt', type=int, default=2048, help='max len for training, including model prompt and response')
    parser.add_argument('--gradient_checkpoint', action='store_true', help='deepspeed')

    # PPO in LLMs
    parser.add_argument('--num_rollouts', type=int, default=128, help='nums of samples in current replay buffer')
    parser.add_argument('--rollout_batch_size', type=int, default=32, help='batch size of sampling from env')

    parser.add_argument('--ppo_pretrain_data_path', type=str, default='', help='dataset folder path for pertrain loss of step3: rlhf')
    parser.add_argument('--ppo_pretrain_data_type', type=str, default='sft', choices=['sft', 'pretrain'], help='dataset folder path for pertrain loss of step3: rlhf')
    parser.add_argument('--ppo_pretrain_batch_size_ratio', type=int, default=1, help='ppo batch size ratio')
    parser.add_argument('--ppo_pretrain_loss_weight', type=float, default=0., help='add pretrain loss in PPO training: ppo-rtx')
    parser.add_argument('--kl_penalty_weight', type=float, default=0.02, help='kl penalty')
    parser.add_argument('--advantage_clip', type=float, default=0.5, help='clip advantage')
    parser.add_argument('--vf_loss_weight', type=float, default=1., help='vf loss weight')
    parser.add_argument('--entropy_loss_weight', type=float, default=0., help='entropy loss weight')
    parser.add_argument('--reward_clip', type=float, default=10., help='reward clip')
    parser.add_argument('--entropy_clip', type=float, default=35., help='entropy loss clip')
    parser.add_argument('--pg_clip', type=float, default=0.2, help='pg loss clip')
    parser.add_argument('--value_clip', type=float, default=0.2, help='value clip for critic model')
    parser.add_argument('--gamma', type=float, default=1., help='GAE in PPO')
    parser.add_argument('--lam', type=float, default=0.95, help='GAE in PPO')

    # Trick and method options for PPO
    parser.add_argument('--use_reward_clip', action='store_true', help='use reward clip')
    parser.add_argument('--use_reward_scaling', action='store_true', help='use reward scaling')
    parser.add_argument('--use_reward_norm', action='store_true', help='user reward norm')
    parser.add_argument('--use_critic_loss_clip', action='store_true', help='use critic loss clip')
    parser.add_argument('--use_policy_loss_clip', action='store_true', help='use policy loss clip')
    parser.add_argument('--use_advantage_norm', action='store_true', help='use advantage norm')
    parser.add_argument('--use_advantage_clip', action='store_true', help='use advantage clip')
    parser.add_argument('--use_ppo_pretrain_loss', action='store_true', help='use ppo pretrain loss')
    parser.add_argument('--use_entropy_loss', action='store_true', help='use ppo entropy loss')

    # Sample from env
    parser.add_argument('--maxlen_res', type=int, default=128, help='max len for model response')
    parser.add_argument('--temperature', type=float, default=0.8, help='temperature')
    parser.add_argument('--repetition_penalty', type=float, default=1.1, help='repetition penalty')
    parser.add_argument('--topp', type=float, default=0.9, help='nucleus sampling')
    
    # Option for language
    parse_args.add_argument('--lang', type=str, choices=['zh', 'en'], help='language prompt for PPO-max-zh or PPO-max-en')

    opt = parser.parse_args()

    return opt
    
    