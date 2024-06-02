import os
import sys
import torch
import tqdm
import numpy as np
import pandas as pd
import time

from dataset.data_helper_moe import create_test_dataloader
from transformers import VisionEncoderDecoderConfig
from transformers import AutoTokenizer, TrOCRProcessor
from models.modeling_vision_encoder_decoder import VisionEncoderDecoderModel
from models.model_moe import BankOCRMoE
from models.modeling_ensemble import EnsembleGenerateModel
from config import parse_args

from post_process import get_post_result
from sentence_ensemble import run_sentence_ensemble


def inference(args, model, loader):
    model.eval()
    # processor = ViTImageProcessor.from_pretrained(args.pretrain_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_dir)
    with torch.no_grad():
        all_preds, all_labels = [], []
        all_probs = []
        with tqdm.tqdm(total=loader.__len__()) as t:
            for index, batch in enumerate(loader):
                pixel_values = batch['pixel_values'].to(args.device)
                texts = batch['texts']
                # priori = batch['priori'].to(args.device)
                priori = None
                outputs = model.generate(pixel_values,
                                         num_beams=args.infer_num_beams,
                                         max_length=args.infer_max_length,
                                         early_stopping=True,
                                         no_repeat_ngram_size=args.infer_no_repeat_ngram_size,
                                         min_length=args.infer_min_length,
                                         length_penalty=args.length_penalty,
                                         return_dict_in_generate=True,
                                         output_scores=True,
                                         # temperature=args.temperature,
                                         # top_k=args.top_k,
                                         # top_p=args.top_p,
                                         repetition_penalty=args.repetition_penalty,
                                         priori=priori
                                         )
                transition_scores = model.compute_transition_scores(
                    outputs.sequences,
                    outputs.scores,
                    outputs.beam_indices,
                    normalize_logits=True
                )
                generated_tokens = outputs.sequences[:, 1:]
                for generated_token, transition_score in zip(generated_tokens, transition_scores):
                    now_probs = []
                    now_pred = ''
                    for tok, score in zip(generated_token, transition_score):
                        if tok == 0:
                            continue
                        if tok == 2:
                            break
                        prob = round(np.exp(score.cpu().numpy()), 2)
                        # if prob < 0.1:
                        #     print(prob)
                        #     continue
                        now_pred = now_pred + tokenizer.decode(tok.cpu())
                        now_probs.append(prob)
                    all_preds.append(now_pred)
                    all_probs.append(now_probs)

                all_labels.extend(texts)
                t.update(1)

    return all_preds


def load_moe_model(pretrain_model_dir, ckpt_file, processor, load_half=False, device=None):
    config = VisionEncoderDecoderConfig.from_pretrained(pretrain_model_dir)
    model = BankOCRMoE(config=config)
    # model.load_moe_encoder_decoder_from_vision_pretrain(args.pretrain_model_dir)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    model.to(device)

    if load_half:
        model.half()
        model.load_state_dict(torch.load(ckpt_file, map_location='cpu'))
    else:
        model.load_state_dict(torch.load(ckpt_file, map_location='cpu'))
        model.half()

    return model


def load_model(pretrain_model_dir, ckpt_file, processor, load_half=False, device=None):
    config = VisionEncoderDecoderConfig.from_pretrained(pretrain_model_dir)
    model = VisionEncoderDecoderModel(config=config)
    # model.load_moe_encoder_decoder_from_vision_pretrain(args.pretrain_model_dir)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    model.to(device)

    if load_half:
        model.half()
        model.load_state_dict(torch.load(ckpt_file, map_location='cpu'))
    else:
        model.load_state_dict(torch.load(ckpt_file, map_location='cpu'))
        model.half()

    return model


def predict_text(args, text_path):
    processor = TrOCRProcessor.from_pretrained(args.pretrain_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_dir)

    model = load_moe_model(
        pretrain_model_dir=args.pretrain_model_dir, ckpt_file='./checkpoints/moe_seed2023_adda_addmerge_all.bin',
        processor=processor, load_half=True, device=args.device
    )
    model1 = load_model(
        pretrain_model_dir=args.pretrain_model_dir, ckpt_file='./checkpoints/text_seed2023_adda_addmergeA_all.bin',
        processor=processor, load_half=True, device=args.device
    )

    # model2 = load_model(
    #     pretrain_model_dir=args.pretrain_model_dir, ckpt_file='./checkpoints/text_seed1999_adda_addmergeA_all.bin',
    #     processor=processor, load_half=True, device=args.device
    # )

    model3 = load_model(
        pretrain_model_dir=args.pretrain_model_dir, ckpt_file='./checkpoints/text_seed3407_adda_addmergeA_all.bin',
        processor=processor, load_half=True, device=args.device
    )
    model4 = load_model(
        pretrain_model_dir=args.pretrain_model_dir, ckpt_file='./checkpoints/text_seed42_adda_addmergeA_all.bin',
        processor=processor, load_half=True, device=args.device
    )
    config = VisionEncoderDecoderConfig.from_pretrained(args.pretrain_model_dir)
    # ensemble_model = EnsembleGenerateModel(config=config, model_list=[model, model1, model2, model3, model4])

    transformer = lambda x: x  ##图像数据增强函数，可自定义
    # transformer = transforms.Grayscale(num_output_channels=3)

    # text
    test_dataloader, images_names = create_test_dataloader(
        args, processor, tokenizer, test_path=text_path, transformer=transformer, types='text'
    )

    # results = inference(args, ensemble_model, test_dataloader)
    results1 = inference(
        args,
        EnsembleGenerateModel(config=config, model_list=[model, model1, model3]),
        test_dataloader
    )
    # results1 = get_post_result(results1)

    results2 = inference(
        args,
        EnsembleGenerateModel(config=config, model_list=[model, model3, model4]),
        test_dataloader
    )
    # results2 = get_post_result(results2)

    results3 = inference(
        args,
        EnsembleGenerateModel(config=config, model_list=[model, model1, model4]),
        test_dataloader
    )
    # results3 = get_post_result(results3)

    # results4 = inference(
    #     args,
    #     EnsembleGenerateModel(config=config, model_list=[model, model1, model3, model4]),
    #     test_dataloader
    # )
    # results4 = get_post_result(results4)

    results = run_sentence_ensemble([results1, results2, results3])
    result_df = pd.DataFrame()
    result_df['file_name'] = images_names
    result_df['result'] = results
    return result_df


if __name__ == '__main__':
    args = parse_args()
    args.device = torch.device(args.device)

    result_df = predict_text(args)

    result_df[['file_name', 'result']].to_csv('./submit/num_sub.csv', encoding='utf8', index=False, )
