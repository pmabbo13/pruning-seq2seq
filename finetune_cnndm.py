import argparse

import datasets
import numpy as np
import nltk
nltk.download('punkt')

from functools import partial
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer


def preprocessData(tokenizer, prefix, max_input_length, max_target_length, examples):
    inputs = [prefix + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["highlights"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(tokenizer, metric, eval_pred):
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
                        choices=['t5-base', 't5-large', 'bart', 'bart-large'],
                        help='The name of the pre-trained model to fine-tune.')
    parser.add_argument('--batch_size', dest='batch_size', required=False,
                        default=128, help='The batch size to be used for training')
    parser.add_argument('--learning_rate', dest='learning_rate', required=False,
                        default=.001, help='The learning rate to be used for training')
    parser.add_argument('--save_steps', dest='save_steps', required=False,
                        default=5000, help='Number of steps of training until checkpoint if saved')
    args = parser.parse_args()

    is_t5 = args.pretrained_model[:3] == 't5-'
    
    # Load raw data
    raw_data = datasets.load_dataset('cnn_dailymail', '3.0.0')

    # Pre-process data
    prefix = 'summarize' if is_t5 else ""
    max_input_length = 512
    max_target_length = 64
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    clean_data = raw_data.filter(
        lambda example: len(example['article']) >= max_input_length and len(example['highlights']) >= max_target_length
    )
    processed_data = clean_data.map(
        partial(preprocessData, tokenizer, prefix, max_input_length, max_target_length),
        batched=True
    )

    print(f"Training args are:\n\tmodel:{args.pretrained_model}\n\tbatch_size:{args.batch_size}\n\tlearning_rate:{args.learning_rate}\n\tsave_steps:{args.save_steps}")
    print(f"New model will be named: {ft_model_name}")
    # set up training arguments
    ft_model_name = f"{args.pretrained_model}-cnndm"
    ft_model_dir = f"./{ft_model_name}"
    args = Seq2SeqTrainingArguments(
        ft_model_dir,
        evaluation_strategy="steps",
        eval_steps=args.save_steps,
        logging_strategy="steps",
        logging_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=args.batch_size/4,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.0,
        save_total_limit=2,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=True,
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        report_to="tensorboard"
    )

    # get original model and train/val samples
    orig_model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model)
    train_sample = processed_data["train"]
    val_sample = processed_data["validation"]

    # train model
    data_collator = DataCollatorForSeq2Seq(tokenizer)
    metric = datasets.load_metric("rouge")
    trainer = Seq2SeqTrainer(
        model_init=orig_model,
        args=args,
        train_dataset=train_sample,
        eval_dataset=val_sample,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    #print(result)
