import os
import torch
import argparse
import numpy as np
import pandas as pd
from time import time
from statistics import mean
from functools import partial

import datasets
from evaluate import load
from transformers import AutoTokenizer

import nltk
nltk.download('punkt')

def preprocessData(tokenizer, prefix, max_input_length, max_target_length, device, examples):
    """
    Preprocess CNNDM examples.
    Taken from preprocess_function in https://github.com/huggingface/notebooks/blob/main/examples/summarization.ipynb
    """
    
    inputs = [prefix + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, padding="longest", max_length=max_input_length, truncation=True, return_tensors="pt").to(device)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["highlights"], padding="longest", max_length=max_target_length, truncation=True, return_tensors="pt").input_ids
        labels = labels.to(device)

    model_inputs["labels"] = labels
    return model_inputs


def compute_metrics(metric, eval_pred):
    """
    Compute BERTScore scores for predicted and target summaries.
    Modified from preprocess_function in https://github.com/huggingface/notebooks/blob/main/examples/summarization.ipynb
    """

    # time evaluation
    start = time()

    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                      for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) 
                      for label in decoded_labels]
    
    # Compute ROUGE scores
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, lang="en")

    # Extract ROUGE f1 scores
    result = {key: mean(value) * 100 for key, value in result.items() if key != 'hashcode'}

    elapsed_time = time() - start
    
    # Add mean generated length to metrics
    predictions = [pred.to('cpu') for pred in predictions]
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    result['eval_time'] = elapsed_time
    
    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_model', dest='hf_model', required=False,
                        default=[],
                        choices = ["ainize/bart-base-cnn", "facebook/bart-large-cnn", "Chikashi/t5-small-finetuned-cnndm", "t5-base"],
                        help='The name of the finetuned model to load from Huggingface.',
                        nargs="+",
                        type=str)
    parser.add_argument('--checkpoint', dest='checkpoint', required=False,
                        default=[],
                        help='The filepath to the checkpoint of the finetuned model to load.',
                        nargs="+",
                        type=str)
    parser.add_argument('--pruning_block', dest='pruning_block', required=False,
                        default='decoder',
                        choices=['encoder', 'decoder'],
                        help='Whether to prune from the encoder or decoder.',
                        type=str)
    parser.add_argument('--pruning_strategy', dest='pruning_strategy', required=False,
                        default='ca-mag',
                        choices=['ca-mag', 'ca-var', 'top'],
                        help='The pruning strategy to use.',
                        type=str)
    parser.add_argument('--pruning_schedule', dest='pruning_schedule', required=False,
                        default=[0.0],
                        help='The pruning schedule; list of floats representing percentage of layers to prune and evaluate.',
                        nargs="+",
                        type=float)
    args = parser.parse_args()
    
    models = args.hf_model + args.checkpoint
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for finetuned_model in models:
        #  Load model and tokenizer
        if finetuned_model in args.hf_model:
            tokenizer = AutoTokenizer.from_pretrained(finetuned_model)
            if '/' in finetuned_model:
                is_t5 = 't5-' == finetuned_model.split("/")[1][:3] 
            else:
                is_t5 = 't5-' == finetuned_model[:3]

        else:
            architecture = finetuned_model.split('/')[0][:-6]
            tokenizer = AutoTokenizer.from_pretrained(architecture)
            is_t5 = 't5-' == finetuned_model[:3]

        # evalute each pruning model
        for remove in args.pruning_schedule:
            
            # create filename to store model results and predictions
            filename = finetuned_model.split('/')[1] if '/' in finetuned_model else finetuned_model
            if args.pruning_block == 'encoder':
                filename += f'-{args.pruning_block}-{args.pruning_strategy}-{remove}' if remove != 0 else f'-baseline'
            else:
                filename += f'-{args.pruning_strategy}-{remove}' if remove != 0 else f'-baseline'

            predfile = f'predictions/{filename}-predictions'
            assert os.path.isfile(predfile), f"{predfile} does not exist"

            # load raw data
            raw_data = datasets.load_dataset('cnn_dailymail', '3.0.0', split="test")

            # Pre-process data
            prefix = 'summarize: ' if is_t5 else ""
            max_input_length = 512
            max_target_length = 200
            test_data = raw_data.map(
                partial(preprocessData, tokenizer, prefix, max_input_length, max_target_length, device),
                batched=True
            )

            # prepare dataloader
            test_data.set_format(type='torch', columns=['input_ids', 'attention_mask'])

            # load predictions
            all_predictions = torch.load(open(predfile, "rb"))
                
            # flatten predictions
            all_predictions_flattened = [pred for preds in all_predictions for pred in preds]
            # tokenize and pad titles
            targets = test_data["labels"]

            # compute bertscore
            metric = load("bertscore")
            predictions_labels = [all_predictions_flattened, targets]
            results = compute_metrics(metric, predictions_labels)
            
            # store results
            if not os.path.isdir("results/bertscore"):
                os.makedirs("results/bertscore")
            pd.Series(results).to_csv(f'results/bertscore/{filename}.csv')
            
            print(f"Completed eval of {filename}. Results are...")
            for key, val in results.items():
                print(f"\t{key}: {val}")
            print(f"Results saved in results/bertscore/{filename}.csv")