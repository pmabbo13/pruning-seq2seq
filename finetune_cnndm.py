import argparse

import datasets
import numpy as np
import torch
import nltk
nltk.download('punkt')

from functools import partial
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

# ENTIRE SCRIPT IS MODIFIED FROM HUGGINGFACE TUTORIAL FOR FINETUNING T5 MODEL FOR SUMMARIZATION:
# https://github.com/huggingface/notebooks/blob/main/examples/summarization.ipynb

def preprocessData(tokenizer, prefix, max_input_length, max_target_length, device, examples):
    """
    Preprocess CNNDM examples
    """

    inputs = [prefix + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, padding="longest", max_length=max_input_length, truncation=True, return_tensors="pt").to(device)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["highlights"], padding="longest", max_length=max_target_length, truncation=True, return_tensors="pt").input_ids
        labels = labels.to(device)

    model_inputs["labels"] = labels
    return model_inputs

def compute_metrics(eval_pred):
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--orig_model', dest='pretrained_model', required=True,
                        choices=['t5-base', 't5-small', 'bart-base', 'bart-large'],
                        help='The name of the pre-trained model to fine-tune.',
                        type=str)
    parser.add_argument('--batch_size', dest='batch_size', required=False,
                        default=8, help='The batch size to be used for training',
                        type=int)
    parser.add_argument('--learning_rate', dest='learning_rate', required=False,
                        default=2e-5, help='The learning rate to be used for training',
                        type=float)
    parser.add_argument('--save_steps', dest='save_steps', required=False,
                        default=3589, help='Number of steps of training until checkpoint if saved',
                        type=int)
    parser.add_argument('--epochs', dest='epochs', required=False,
                        default=1, help='Number of training epochs',
                        type=int)
    args = parser.parse_args()

    is_t5 = args.pretrained_model[:3] == 't5-'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Training on {device}")
    
    # Load raw data
    raw_data = datasets.load_dataset('cnn_dailymail', '3.0.0')

    # Pre-process data
    prefix = 'summarize: ' if is_t5 else ""
    max_input_length = 512
    max_target_length = 200
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    processed_data = raw_data.map(
        partial(preprocessData, tokenizer, prefix, max_input_length, max_target_length, device),
        batched=True
    )
    train_sample = processed_data["train"]
    val_sample = processed_data["validation"]

    # set up training arguments
    ft_model_name = f"{args.pretrained_model}-cnndm"
    ft_model_dir = f"./{ft_model_name}"
    train_args = Seq2SeqTrainingArguments(
        ft_model_dir,
        evaluation_strategy="steps",
        eval_steps=args.save_steps,
        logging_strategy="steps",
        logging_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_total_limit=3,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        report_to="tensorboard"
    )

    # get original model and train/val samples
    model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model)
    model = model.to(device)
    model = model.train()

    # train model
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    metric = datasets.load_metric("rouge")
    trainer = Seq2SeqTrainer(
        model,
        train_args,
        train_dataset=train_sample,
        eval_dataset=val_sample,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print(f"Training args are:\n\tmodel:{args.pretrained_model}\n\tbatch_size:{args.batch_size}\n\tlearning_rate:{args.learning_rate}\n\tsave_steps:{args.save_steps}")
    print(f"New model will be named: {ft_model_name}")

    trainer.train()
