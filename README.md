# pruning-seq2seq

This library explores layer pruning strategies on transformer-based sequence-to-sequence models to identify whether we can achieve comparable task performance on abstractive summarization using smaller derivatives of BART and T5 models. The pruning stragies considered are:

1. Removing layers with the lowest average magnitude of weights applied to their cross-attention outputs
2. Remove layers from the top downards on the model's decoder
3. Remove layer from the top downards on the model's encoder

## Data
We use the CNNDM dataset (sourced from Hugging Face's `datasets` library) to train and evaluate our models for abstractive text summarization. We process the input articles by tokenizing and truncating them to a maximum length of 512. The target summaries are also tokenized and truncated to a maximum length of 200.

## Models
We use [BART-base](https://huggingface.co/ainize/bart-base-cnn), [BART-large](https://huggingface.co/facebook/bart-large-cnn), and [T5-small](https://huggingface.co/Chikashi/t5-small-finetuned-cnndm) models that have already been fine-tuned on the task and made publicly available through Hugging Face's `transformers` library. We also use a pre-trained [T5-base](https://huggingface.co/t5-base) model and fine-tune it ourselves using our `finetune_cnndm.py` script.
We take advantage of Hugging Face's `transformers.Seq2SeqTrainingArguments` and `transformers.Seq2SeqTrainer` methods to train these models.

## Package Dependencies
    datasets
    evaluate
    nltk
    numpy
    pandas
    pytorch
    tensorboard
    transformers
    

## Scripts
`finetune_cnndm.py` is used to fine-tune an input pre-trained model on the CNNDM dataset for the task of abstractive summarization. Our training implementation is slightly modified from the training instructions provided in [this Hugging Face tutorial](https://github.com/huggingface/notebooks/blob/main/examples/summarization.ipynb).

We describe the script's input parameters below:

    --orig_model: The name of the pre-trained model to fine-tune. (ex. bart-base, bart-large, t5-small, t5-base)
    --batch_size: The batch size used during training
    --learning_rate: The learning rate used during training
    --epochs: The number of training epochs
    --save_steps: The number of optimization steps to occur before the model is evaluated against the validation set and the checkpoint is saved if it is the best performing checkpoint thus far in the training process

We used the following command to fine-tune the pre-trained t5-base model:
  
  `finetune_cnndm.py --orig_model t5-base --batch_size 8 --learning_rate 2e-05 --epochs 1 --save_steps 3489`

----

`evaluate_cnndm.py` is used to apply a given layer pruning technique to the its input model and evaluate it against the test set of the CNNDM dataset. It computes the model's ROUGE scores, average length of each summary, the total time taken to generate the summaries, and the number of parameters in the model. A report of these results is saved in the `results ` directory. It also saves the predicted sequences in the `predictions` summary so that we can perform manual inspections of the sequences and also evaluate them against other metrics if we wish.

We describe the script's input parameters below:

    --hf_model: The name of the finetuned model to load from Huggingface. (ex. 'facebook/bart-large-cnn')
    --checkpoint: The filepath to a model checkpoint we want to load (ex. filepath to checkpoing for fine-tuned t5-base model)
    --pruning_block: Whether to prune from the encoder or decoder
    --pruning_strategy: What pruning strategy to use (optionas are 'ca-mag' to use magnitude of cross-attention weights or 'top' to use top-down method)
    --pruning_schedule: List of pruning percentages to evaluate against. Value of 0.0 will evalute model without any pruning, and is used to generate baseline values.
    --batch_size: The batch size to be used for evaluation
    --metric: The metric to be used for evaluation (current implementation only supports ROUGE)

We used the following command to evaluate the fine-tuned bart-large model using a top-down strategy on the decoder at pruning percentages of 0 (i.e. baseline), 10%, 25%, and 50%.
  
  `evaluate_cnndm.py --hf_model facebook/bart-large-cnn --pruning_block decoder --pruning_strategy top --pruning_schedule 0.0 0.1 0.25 0.5 --batch_size 8 --metric rouge`
  
  
----

`bertscore.py` is used to evaluate system generated predictions using the BERTScore metric. The results are saved in a report in the `results/bertscore` directory.

It takes the following input parameters to locate the corresponding predictions in the `predictions` directory:

    --hf_model: The name of the finetuned model to load from Huggingface. (ex. 'facebook/bart-large-cnn')
    --checkpoint: The filepath to a model checkpoint we want to load (ex. filepath to checkpoing for fine-tuned t5-base model)
    --pruning_block: Whether to prune from the encoder or decoder
    --pruning_strategy: What pruning strategy to use (optionas are 'ca-mag' to use magnitude of cross-attention weights or 'top' to use top-down method)
    --pruning_schedule: List of pruning percentages to evaluate against. Value of 0.0 will evalute model without any pruning, and is used to generate baseline values.
    
We used the follwoing command to compute the BERTScore values for the fine-tuned bart-large model using a top-down strategy on the decoder at pruning percentages of 0 (i.e. baseline), 10%, 25%, and 50%.

  `bertscore.py --hf_model facebook/bart-large-cnn --pruning_block decoder --pruning_strategy top --pruning_schedule 0.0 0.1 0.25 0.5`
  
