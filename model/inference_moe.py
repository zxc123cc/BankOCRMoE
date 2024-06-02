import torch
import tqdm
import numpy as np

from dataset.data_helper_moe import create_test_dataloader
from transformers import VisionEncoderDecoderConfig
from transformers import AutoTokenizer, TrOCRProcessor
from models.model_moe import BankOCRMoE
from config import parse_args
import pandas as pd


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


def predict_num(args, num_path):
    device = args.device

    processor = TrOCRProcessor.from_pretrained(args.pretrain_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_dir)

    config = VisionEncoderDecoderConfig.from_pretrained(args.pretrain_model_dir)
    model = BankOCRMoE(config=config)
    # model.load_moe_encoder_decoder_from_vision_pretrain(args.pretrain_model_dir)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    model.to(device)

    model.load_state_dict(torch.load('./checkpoints/moe_seed2023_adda_addmerge_all.bin', map_location='cpu'))

    model.half()

    transformer = lambda x: x  ##图像数据增强函数，可自定义
    # transformer = transforms.Grayscale(num_output_channels=3)
    test_dataloader, images_names = create_test_dataloader(
        args, processor, tokenizer, test_path=num_path, transformer=transformer, types='num'
    )

    results = inference(args, model, test_dataloader)

    result_df = pd.DataFrame()
    result_df['file_name'] = images_names
    result_df['result'] = results
    return result_df


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(args.device)

    num_results_df = predict_num(args, '')

    num_results_df[['file_name', 'result']].to_csv('./submit/num_sub.csv', encoding='utf8', index=False)
