#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：whisper-main
@File    ：tuning_16.py
@IDE     ：PyCharm
@Author  ：Aisaka/Hao Ma @SDU
@Date    ：2023/7/16 下午10:54
'''

# expanded by Daniel Bohata @VUT

import argparse
import random
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import whisper
from tqdm import tqdm
from whisper.tokenizer import get_tokenizer
from data_utils.dataloader import get_dataloader
from model.prompting import Prompting
import os
import wandb  # Add wandb import
from whisper.model import ResidualAttentionBlock
from torch.optim.lr_scheduler import LinearLR
from lr_scheduler import REX_LR

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a Whisper model for ASR")
    # Dataloader-related arguments
    parser.add_argument(
        "--train-json",
        type=str,
        # required=True,
        default='data_utils/data/train-100.json',
        help="Path to a json file containing training data",
    )
    parser.add_argument(
        "--embed_path",
        type=str,
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--dev-batch-size", type=int, default=1, help="Batch size for validation")
    parser.add_argument(
        "--no-timestamps-training",
        default=True,
        help="Always use the no-timestamps training mode",
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to use for training",
    )
    parser.add_argument(
        "--model",
        default="medium",
        choices=whisper.available_models(),
        help="name of the Whisper model to use",
    )
    parser.add_argument(
        "--prompt_length",
        type=int,
        required=True,
        help="soft prompt length",
    )

    parser.add_argument(
        "--use_mlp",
        action='store_true',
        help="whether to reparameterize the prompt",
    )
    parser.add_argument(
        "--deep",
        action='store_true',
        help="deep prompting",
    )

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--exp_name", type=str, required=True, help="exp_name")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser


class System(nn.Module):
    def __init__(self, model, prompt_layer):
        super(System, self).__init__()
        self.prompt_layer = prompt_layer
        self.model = model
        self.hooks, self.prompt = self._install_hooks(prompt_layer.depth)
        i = 0
        # self.prompt[self.model.encoder.conv2] = None
        for layer in self.model.encoder.modules():
            if isinstance(layer, ResidualAttentionBlock):
                if i < self.prompt_layer.depth:
                    self.prompt[layer] = None
                i += 1
        self.prompt[self.model.decoder.token_embedding] = None
        i = 0
        for layer in self.model.decoder.modules():
            if isinstance(layer, ResidualAttentionBlock):
                if 0 < i < self.prompt_layer.depth:
                    self.prompt[layer] = None
                i += 1

    def forward(self, x, xvec, y_in):
        learned_prompt = self.prompt_layer(xvec)
        for layer, prompt in zip(self.prompt.keys(), learned_prompt):
            self.prompt[layer] = prompt
        logits = self.model.decoder(y_in, self.model.encoder(x))
        return logits

    def _install_hooks(self, depth):

        hooks = []
        prompt = {}

        def prompting_hook_fn_decoder_intermediate(module, args):
            modified_input = list(args)
            modified_input[0][:, 1:prompt[module].size(1)+1, :] = prompt[module]
            return tuple(modified_input)

        def prompting_hook_fn_decoder_input(module, _, fea_out):
            fea_out[:, 1:prompt[module].size(1)+1, :] = prompt[module]
            return fea_out

        def prompting_hook_fn_encoder_intermediate(module, args):
            modified_input = list(args)
            modified_input[0][:, 1:prompt[module].size(1)+1, :] = prompt[module]
            return tuple(modified_input)

        def prompting_hook_fn_encoder_input(module, args):
            modified_input = list(args)
            modified_input[0][:, 0:prompt[module].size(1), :] = prompt[module] + self.model.encoder.positional_embedding[0:prompt[module].size(1), :]
            return tuple(modified_input)

        i = 0
        for layer in self.model.encoder.modules():
            if isinstance(layer, ResidualAttentionBlock):
                if i == 0:
                    # encoder prompting encoder
                    hooks.append(layer.register_forward_pre_hook(prompting_hook_fn_encoder_input))
                if 0 < i < depth:
                    # encoder prompting intermediate
                    hooks.append(layer.register_forward_pre_hook(prompting_hook_fn_encoder_intermediate))
                i += 1
        # decoder prompting input
        hooks.append(self.model.decoder.token_embedding.register_forward_hook(prompting_hook_fn_decoder_input))
        i = 0
        for layer in self.model.decoder.modules():
            if isinstance(layer, ResidualAttentionBlock):
                # if i == 0:
                #     hooks.append(layer.register_forward_pre_hook(prompting_hook_fn_decoder_input))
                if 0 < i < depth:
                    # decoder prompting intermediate
                    hooks.append(layer.register_forward_pre_hook(prompting_hook_fn_decoder_intermediate))
                i += 1
        return hooks, prompt


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train(
        system,
        train_loader,
        epochs,
        optimizer,
        scheduler,
        exp_name,
        train_loss=None,
        init_epoch=-1,
):
    system.train()
    system.model.eval()
    system.cuda()
    train_loss = [] if train_loss is None else train_loss.tolist()
    
    for e in range(init_epoch + 1, epochs):
        pbar = tqdm(train_loader)
        epoch_loss = 0.0
        step_count = 0
        epoch_grad_norm = 0.0
        
        for i, (x, xvec, y_in, y_out) in enumerate(pbar):
            x, xvec, y_in, y_out = x.cuda(), xvec.cuda(), y_in.cuda(), y_out.cuda()
            logits = system(x, xvec, y_in)
            loss = F.cross_entropy(logits.transpose(1, 2), y_out)
            loss.backward()
            
            # Calculate gradient norm
            total_norm = 0.0
            for p in system.parameters():
                if p.requires_grad and p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            # Log per-batch metrics
            wandb.log({
                "batch_loss": loss.item(),
                "batch_grad_norm": total_norm,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "global_step": e * len(train_loader) + i
            })
            
            #if (i + 1) % 8 == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            loss_value = loss.detach().cpu().numpy()
            train_loss.append(loss_value)
            epoch_loss += loss_value
            epoch_grad_norm += total_norm
            step_count += 1
            
            pbar.set_postfix({
                "loss": loss_value, 
                'loss_mean': np.sum(train_loss)/len(train_loss),
                'grad_norm': total_norm
            })
        
        # Log epoch metrics
        wandb.log({
            "epoch": e,
            "epoch_avg_loss": epoch_loss / step_count,
            "epoch_avg_grad_norm": epoch_grad_norm / step_count,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        scheduler.step()
        torch.save(system.prompt_layer.state_dict(), './checkpoint/model_' + exp_name + '_{}'.format(e))


def main():
    args = get_parser().parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    
    scheduler_type = os.getenv('scheduler_type')

    wandb.init(
        project="ts-asr-whisper-experiment",
        name=args.exp_name,
        config={
            "model": args.model,
            "epochs": os.getenv('epoch'),
            "noisy_epochs": os.getenv('noisy_epoch'),
            "scheduler": scheduler_type,
            "prompt_length": args.prompt_length,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "use_mlp": args.use_mlp,
            "deep_prompting": args.deep,
        }
    )

    tokenizer = get_tokenizer(multilingual=".en" not in args.model, task="transcribe")
    model = whisper.load_model(args.model, args.device)
    #  -1 is for the special token `sot_prev` and the other half is for the transcribed tokens
    max_prompt_length = model.dims.n_text_ctx // 2 - 1

    fp16 = False
    #train_loader = get_dataloader(
    #    json=args.train_json,
    #    tokenizer=tokenizer,
    #    batch_size=args.batch_size,
    #    fp16=fp16,
    #    no_timestamps_training=args.no_timestamps_training,
    #    max_prompt_length=max_prompt_length,
    #    prompt_use_rate=1,
    #    no_timestamps_rate=0.0,
    #    context_len=args.prompt_length,
    #    embed_path=args.embed_path,
    #    shuffle=True,
    #    n_workers=8,
    #)

    prompt_layer = Prompting(
        dim=model.dims.n_text_state,
        prompt_length=args.prompt_length,
        depth=model.dims.n_audio_layer if args.deep else 1,
        use_mlp=args.use_mlp,
    )

    # freeze the whole whisper model
    for p in model.parameters():
        p.requires_grad = False
    system = System(model, prompt_layer)
    optimizer = torch.optim.AdamW([param for param in system.parameters() if param.requires_grad], lr=args.lr)

    epoch_cnt = os.getenv('epoch')
    epoch_cnt = int(epoch_cnt)

    """
    if scheduler_type is None:
        scheduler_type =  'step'  
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epoch_cnt // 2, gamma=0.1, verbose=True)
    elif scheduler_type == 'step':
        # Use StepLR scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epoch_cnt // 2, gamma=0.1, verbose=True)
    elif scheduler_type == 'linear':
        # Use LinearLR scheduler
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=epoch_cnt, verbose=True)
    elif scheduler_type == 'cosine':
        # Use CosineAnnealingLR scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_cnt, eta_min=0.0, verbose=True)
    elif scheduler_type == 'exponential':
        # Use ExponentialLR scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)
    elif scheduler_type == 'one_cycle':
        # Use OneCycleLR scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=epoch_cnt, verbose=True)
    elif scheduler_type == 'reflected_exponential':
        # Use REX_LR scheduler
        scheduler = REX_LR(optimizer, max_val=1.0, min_val=0.1, num_epochs=epoch_cnt)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")    
    """
            

    #train(
    #    system=system,
    #    train_loader=train_loader,
    #    epochs=epoch_cnt,
    #    optimizer=optimizer,
    #    scheduler=scheduler,
    #    exp_name=args.exp_name,
    #)

    # After training on clean data, log the transition
    #wandb.log({"training_phase": "clean_data_complete"})

    train_loader = get_dataloader(
        #json='./data_utils/data/train-100-noisy.json',
        json='./data_utils/data/train-100-mixed.json',
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        fp16=fp16,
        no_timestamps_training=args.no_timestamps_training,
        max_prompt_length=max_prompt_length,
        prompt_use_rate=1,
        no_timestamps_rate=0.0,
        context_len=args.prompt_length,
        embed_path=args.embed_path,
        shuffle=True,
        n_workers=8,
    )
    optimizer = torch.optim.AdamW([param for param in system.parameters() if param.requires_grad], lr=args.lr * 0.1)

    noisy_epoch_cnt = os.getenv('noisy_epoch')
    noisy_epoch_cnt = int(noisy_epoch_cnt)

    if scheduler_type is None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=noisy_epoch_cnt // 2, gamma=0.1, verbose=True)
    elif scheduler_type == 'step':
        # Use StepLR scheduler for noisy data training
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=noisy_epoch_cnt // 2, gamma=0.1, verbose=True)
    elif scheduler_type == 'linear':
        # Use LinearLR scheduler for noisy data training
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=noisy_epoch_cnt, verbose=True)
    elif scheduler_type == 'cosine':
        # Use CosineAnnealingLR scheduler for noisy data training
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=noisy_epoch_cnt, eta_min=0.0, verbose=True)
    elif scheduler_type == 'exponential':
        # Use ExponentialLR scheduler for noisy data training
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)
    elif scheduler_type == 'one_cycle':
        # Use OneCycleLR scheduler for noisy data training
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr * 0.1, total_steps=noisy_epoch_cnt, verbose=True)
    elif scheduler_type == 'reflected_exponential':
        # Use REX_LR scheduler for noisy data training
        scheduler = REX_LR(optimizer, max_val=1.0, min_val=0.1, num_epochs=noisy_epoch_cnt)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    
    # Log the transition to noisy data training
    wandb.log({"training_phase": "noisy_data_started"})
    
    train(
        system=system,
        train_loader=train_loader,
        epochs=noisy_epoch_cnt,
        optimizer=optimizer,
        scheduler=scheduler,
        exp_name='noisy' + args.exp_name,
    )
    
    # Finish the wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
