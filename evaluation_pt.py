#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：TS_Whisper 
@File    ：evaluation_pt.py
@IDE     ：PyCharm 
@Author  ：Aisaka/Hao Ma @SDU
@Date    ：2023/12/19 下午12:28 
'''

# expanded by Daniel Bohata @VUT

from whisper.decoding import DecodingTask
from model.prompting import Prompting, Prompting_len0
import argparse
import whisper
from whisper.tokenizer import get_tokenizer
import torch
from data_utils.dataloader import get_dataloader
import os
from jiwer import wer, cer
from tqdm import tqdm
from whisper.normalizers import EnglishTextNormalizer
from whisper.model import ResidualAttentionBlock
import wandb 


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test script")
    # Dataloader-related arguments
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to use for training",
    )
    parser.add_argument(
        "--embed_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model",
        default="large-v2",
        choices=whisper.available_models(),
        help="name of the Whisper model to use",
    )
    parser.add_argument(
        "--model_name",
        required=True,
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
    return parser


def test_(loader,
          model,
          prompt_layer,
          mytask,
          normal
          ):
    wer_scores = []
    cer_scores = [] 
    pbar = tqdm(loader)
    hook, prompts = install_hooks(model, depth=prompt_layer.depth)
    i = 0
    # prompts[model.encoder.conv2] = None
    for layer in model.encoder.modules():
        if isinstance(layer, ResidualAttentionBlock):
            if i < prompt_layer.depth:
                prompts[layer] = None
            i += 1
    i = 0
    prompts[model.decoder.token_embedding] = None
    for layer in model.decoder.modules():
        if isinstance(layer, ResidualAttentionBlock):
            if 0 < i < prompt_layer.depth:
                prompts[layer] = None
            i += 1

    sample_count = 0
    with torch.no_grad():
        for x, xvec, y_in, y_out, text in pbar:
            x, xvec = x.to(model.device), xvec.to(model.device)
            learned_prompt = prompt_layer(xvec)
            for layer, prompt in zip(prompts.keys(), learned_prompt):
                prompts[layer] = prompt

            results = mytask.run(x)
            for result, t in zip(results, text):
                normalized_reference = normal(t)
                normalized_hypothesis = normal(result.text)
                
                # Calculate WER
                wer_value = wer(normalized_reference, normalized_hypothesis)
                wer_scores.append(wer_value)
                
                # Calculate CER
                cer_value = cer(normalized_reference, normalized_hypothesis)
                cer_scores.append(cer_value)
                
                # Log individual sample metrics
                wandb.log({
                    "sample_wer": wer_value,
                    "sample_cer": cer_value,
                    "sample_id": sample_count,
                    "reference_text": normalized_reference,
                    "hypothesis_text": normalized_hypothesis
                })
                
                sample_count += 1
                
                pbar.set_postfix({
                    'wer': wer_value, 
                    'wer_mean': sum(wer_scores) / len(wer_scores),
                    'cer': cer_value,
                    'cer_mean': sum(cer_scores) / len(cer_scores),
                    'text': result.text[:30] + "..." if len(result.text) > 30 else result.text
                })
                print(f"Reference: {normalized_reference}")
                print(f"Hypothesis: {normalized_hypothesis}")
                print(f"WER: {wer_value:.4f}, CER: {cer_value:.4f}")
                print("-" * 50)
    
    # Calculate and log final metrics
    final_wer = sum(wer_scores) / len(wer_scores)
    final_cer = sum(cer_scores) / len(cer_scores)
    
    # Log summary metrics
    wandb.log({
        "final_wer": final_wer,
        "final_cer": final_cer,
        "total_samples": len(wer_scores)
    })
    
    print(f"Final WER: {final_wer:.4f}")
    print(f"Final CER: {final_cer:.4f}")
    
    return final_wer, final_cer


def install_hooks(model, depth):
    hooks = []
    prompt = {}

    def prompting_hook_fn_decoder_intermediate(module, args):
        modified_input = list(args)
        if args[0].size(1) > 1:
            modified_input[0][:, 1:prompt[module].size(1)+1, :] = prompt[module]
        return tuple(modified_input)

    def prompting_hook_fn_decoder_input(module, _, fea_out):
        if fea_out.size(1) > 1:
            fea_out[:, 1:prompt[module].size(1)+1, :] = prompt[module]
        return fea_out

    def prompting_hook_fn_encoder_intermediate(module, args):
        modified_input = list(args)
        modified_input[0][:, 1:prompt[module].size(1)+1, :] = prompt[module]
        return tuple(modified_input)

    # def prompting_hook_fn_encoder_input(module, _, fea_out):
    #     fea_out[:, :, 0:prompt[module].size(1)] = prompt[module].permute(0, 2, 1)
    #     return fea_out
    def prompting_hook_fn_encoder_input(module, args):
        modified_input = list(args)
        modified_input[0][:, 0:prompt[module].size(1), :] = prompt[module] + model.encoder.positional_embedding[0:prompt[module].size(1), :]
        return tuple(modified_input)

    # hooks.append(model.encoder.conv2.register_forward_hook(prompting_hook_fn_encoder_input))
    i = 0
    for layer in model.encoder.modules():
        if isinstance(layer, ResidualAttentionBlock):
            if i == 0:
                hooks.append(layer.register_forward_pre_hook(prompting_hook_fn_encoder_input))
            if 0 < i < depth:
                hooks.append(layer.register_forward_pre_hook(prompting_hook_fn_encoder_intermediate))
            i += 1
    hooks.append(model.decoder.token_embedding.register_forward_hook(prompting_hook_fn_decoder_input))
    i = 0
    for layer in model.decoder.modules():
        if isinstance(layer, ResidualAttentionBlock):
            if 0 < i < depth:
                hooks.append(layer.register_forward_pre_hook(prompting_hook_fn_decoder_intermediate))
            i += 1

    return hooks, prompt


if __name__ == '__main__':
    args = get_parser().parse_args()
    
    # Initialize wandb
    model_name_for_display = os.getenv('exp_name')
    wandb.init(
        project="ts-asr-whisper-eval-experiment",
        name=f"eval_{model_name_for_display}",
        config={
            "model": args.model,
            "checkpoint": args.model_name,
            "prompt_length": args.prompt_length,
            "use_mlp": args.use_mlp,
            "deep_prompting": args.deep,
            "scheduler": os.getenv('scheduler_type'),
            "epochs": os.getenv('epoch'),
            "noisy_epochs": os.getenv('noisy_epoch'),
            "evaluation_type": "clean"
        }
    )
    
    tokenizer = get_tokenizer(multilingual=".en" not in args.model, task="transcribe")
    model = whisper.load_model(args.model, args.device)

    max_prompt_length = model.dims.n_text_ctx // 2 - 1

    options = whisper.DecodingOptions(without_timestamps=True, fp16=False, prompt="@"*args.prompt_length)
    task = DecodingTask(model, options)
    normalizer = EnglishTextNormalizer()
    
    dataloader = get_dataloader(
        json='./data_utils/data/test.json',
        tokenizer=tokenizer,
        batch_size=16,
        fp16=False,
        no_timestamps_training=True,
        max_prompt_length=max_prompt_length,
        prompt_use_rate=1.0,
        no_timestamps_rate=0.0,
        context_len=args.prompt_length,
        shuffle=False,
        embed_path=args.embed_path,
        n_workers=0,
        dev=True,
    )
    prompt_layer = Prompting(
        dim=model.dims.n_text_state,
        prompt_length=args.prompt_length,
        depth=model.dims.n_audio_layer if args.deep else 1,
        use_mlp=args.use_mlp,
    )

    weight = torch.load("./checkpoint/" + args.model_name)
    prompt_layer.load_state_dict(weight, strict=True)
    prompt_layer.cuda()
    model.cuda()
    if args.use_mlp:
        prompt_layer.reparameterization().use_mlp = False
    model.eval()
    prompt_layer.eval()
    total = sum([param.nelement() for param in prompt_layer.parameters()])
    total += sum([param.nelement() for param in model.parameters()])

    print('Number of parameter: % .4fM' % (total / 1e6))
    wandb.log({"total_parameters_M": total / 1e6})
    
    wer_score, cer_score = test_(loader=dataloader, model=model, prompt_layer=prompt_layer, mytask=task, normal=normalizer)
    print(f"Final WER: {wer_score:.4f}, Final CER: {cer_score:.4f}")
    
    # Make sure to finish the wandb run
    wandb.finish()
