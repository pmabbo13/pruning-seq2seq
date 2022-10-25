import argparse
import copy
import math
import numpy as np
import pandas as pd

from functools import partial
from tqdm import tqdm
from time import time

import torch
import torch.nn as nn
import nltk
nltk.download('punkt')

import datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def preprocessData(tokenizer, prefix, max_input_length, max_target_length, examples):
    inputs = [prefix + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, padding="longest", max_length=max_input_length, truncation=True, return_tensors="pt")

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["highlights"], padding="longest", max_length=max_target_length, truncation=True, return_tensors="pt").input_ids
        #labels[labels == tokenizer.pad_token_id] = -100

    model_inputs["labels"] = labels
    return model_inputs

def compute_metrics(metric, eval_pred):
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
    result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                            use_stemmer=True)

    # Extract ROUGE f1 scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length to metrics
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                      for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}


def getLayersForRemoval(model, is_t5, remove):
    
    layer_weights = []
    num_layers = model.config.num_decoder_layers if is_t5 else model.config.decoder_layers
    for i in range(num_layers):
        if is_t5:
            weight = model.state_dict()[f"decoder.block.{i}.layer.1.EncDecAttention.o.weight"]
        else:
            weight = model.state_dict()[f"model.decoder.layers.{i}.encoder_attn.out_proj.weight"]
        avg_abs_weight = torch.mean(torch.abs(weight)).item()
        layer_weights.append((i, avg_abs_weight))
    layer_weights.sort(key = lambda x: x[1])
    excl_layers = math.ceil(num_layers * remove)
    remove_layers = set([l for l, a in layer_weights[:excl_layers]])
    return remove_layers
    
def getPrunedModel(orig_model, is_t5, remove):
    
    remove_layers = getLayersForRemoval(orig_model, is_t5, remove)
    
    current_dec_layers = orig_model.decoder.block if is_t5 else orig_model.model.decoder.layers
    new_dec_layers = nn.ModuleList([current_dec_layers[i] for i in range(len(current_dec_layers)) if i not in remove_layers])
    
    new_model = copy.deepcopy(orig_model)
    if is_t5:
        new_model.decoder.block = new_dec_layers
        assert len(new_model.decoder.block) + len(remove_layers) == len(orig_model.decoder.block)
    else:
        new_model.model.decoder.layers = new_dec_layers
        assert len(new_model.model.decoder.layers) + len(remove_layers) == len(orig_model.model.decoder.layers)

    return new_model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def evaluate_model(model, test_data, batch_size, max_target_length):
    
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # prepare dataloader
    test_data.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    
    # get size of model
    num_parameters = count_parameters(model)
    
    # time evaluation
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
    else:
        start = time()
    
    # generate text for each batch
    all_predictions = []
    if torch.cuda.is_available():
        start.record()
    for i,batch in tqdm(enumerate(dataloader)):
        predictions = model.generate(**batch, max_length=max_target_length)
        all_predictions.append(predictions)
    
    if torch.cuda.is_available():
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
    else:
        elapsed_time = time() - start
    
    # flatten predictions
    all_predictions_flattened = [pred for preds in all_predictions for pred in preds]
    # tokenize and pad titles
    targets = test_data["labels"]

    # compute metrics
    metric = datasets.load_metric("rouge")
    predictions_labels = [all_predictions_flattened, targets]
    results = compute_metrics(metric, predictions_labels)
    
    results['eval_time'] = elapsed_time
    results['model_size'] = num_parameters
    
    return results

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', required=False,
                        default="all",
                        help='The name of the finetuned model to evaluate.',
                        type=list)
    parser.add_argument('--pruning_schedule', dest='pruning_schedule', required=False,
                        default=[0.0],
                        help='The pruning schedule; list of floats representing percentage of layers to prune and evaluate.',
                        type=list)
    parser.add_argument('--batch_size', dest='batch_size', required=False,
                        default=8, help='The batch size to be used for eval',
                        type=int)
    args = parser.parse_args()

    if args.model == "all":
        MODELS = ["ainize/bart-base-cnn", "facebook/bart-large-cnn", "t5-base", "t5-large"]
    else:
        MODELS = [args.model]
    
    for finetuned_model in MODELS:

        #  Load Model and Tokenize
        tokenizer = AutoTokenizer.from_pretrained(finetuned_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(finetuned_model)
        is_t5 = 't5-' == finetuned_model.split("/")[1][:3]

        # Load raw data
        raw_data = datasets.load_dataset('cnn_dailymail', '3.0.0', split="test")

        # Pre-process data
        prefix = 'summarize: ' if is_t5 else ""
        max_input_length = 512
        max_target_length = 128
        test_data = raw_data.map(
            partial(preprocessData, tokenizer, prefix, max_input_length, max_target_length),
            batched=True
        )
        test_data = test_data

        for remove in args.pruning_schedule:

            if remove == 0:
                pruned_model = model
            else:
                pruned_model = getPrunedModel(model, is_t5, remove)

            results = evaluate_model(pruned_model, test_data, args.batch_size, max_target_length)

            filename = finetuned_model.split('/')[1] if '/' in finetuned_model else finetuned_model
            filename += '_' + str(remove) if remove != 0 else '_baseline'
            pd.Series(results).to_csv(f'{filename}.csv')

            print(f"Completed eval of {filename}. Results are...\n\t{results}")