import argparse
import glob
import json
import logging
import os
import sys

import numpy as np
from transformers import GPT2Tokenizer, Trainer, TrainingArguments, GPT2LMHeadModel
from transformers import TrainerCallback
from torch.utils.data import DataLoader

from mydataset import TrainDataset, EvalDataset


def setup_logger(name, log_file, level=logging.DEBUG):
    """Sets up a logger to output to both terminal and file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class StringOutputEvaluator(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer, logger, output_dir="./"):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.logger = logger
        self.Tdataloader = DataLoader(eval_dataset, batch_size=1,
                                      shuffle=False)  # Multiple Batch Size requires paddings
        self.output_dir = output_dir

    def on_log(self, args, state, control, **kwargs):
        # model = kwargs.get('model')
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model.to(device)
        # model.eval()

        epoch = state.log_history[-1]['epoch']
        loss = state.log_history[-1].get('loss', np.inf)
        step = state.log_history[-1]['step']

        history_len = len(state.log_history)
        history_avg_1 = 9999
        history_avg_2 = 0
        if history_len > 20:
            history_len_2 = 20
            history_len_4 = 10
            history_avg_1 = np.average([x.get('loss', 9999) for x in state.log_history[-history_len_2:-history_len_4]])
            history_avg_2 = np.average([x.get('loss', 9999) for x in state.log_history[-history_len_4:]])

        diff = abs(history_avg_1 - history_avg_2)
        log_str = json.dumps({'step': step,
                              'epoch': epoch,
                              'loss': loss,
                              "history": {
                                  "history_avg_1": history_avg_1,
                                  "history_avg_2": history_avg_2,
                                  "diff": diff
                              }}
                             )

        self.logger.info(log_str)

        if loss <= 0.001 or diff < 0.0000001:
            sys.exit()
        elif os.path.exists(os.path.join(self.output_dir, "stop.txt")):
            sys.exit()


def main():
    parser = argparse.ArgumentParser(description='Train a GPT-2 model.')
    parser.add_argument('--model_name', type=str, default='gpt2', help='Pre-trained model name or path')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Path to the checkpoint file.')
    parser.add_argument('--dataset_train', type=str, default='./data/bindataRL_4_A.csv',
                        help='Path to the training dataset')
    parser.add_argument('--dataset_eval', type=str, default='./data/bindata_4_A.csv',
                        help='Path to the evaluation dataset')
    parser.add_argument('--output_dir', type=str, default='./results_bintrainRL1', help='Path to output directory')
    parser.add_argument('--epoch', type=int, default=15000, help='Path to output directory')
    parser.add_argument('--batch_size', type=int, default=256, help='Path to output directory')
    parser.add_argument('--logging_steps', type=int, default=250, help='Path to output directory')

    args = parser.parse_args()
    print(f"args:{args}")

    model = None
    if args.ckpt_path:
        checkpoint_paths = sorted(glob.glob(os.path.join(args.ckpt_path, "checkpoint-*")))
        if len(checkpoint_paths) > 0:
            model = GPT2LMHeadModel.from_pretrained(checkpoint_paths[-1])
            print(f"load checkpoint from {checkpoint_paths[-1]}")

    if model is None:
        model = GPT2LMHeadModel.from_pretrained(args.model_name)

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)

    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'eos_token': '[EOS]'})
    tokens = json.load(open(args.dataset_train.replace(".csv", "_token.json"), "r"))
    special_tokens_dict = {'additional_special_tokens': tokens}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    dataset_train = TrainDataset(args.dataset_train, tokenizer)
    dataset_eval = EvalDataset(args.dataset_eval, tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.batch_size,
        save_steps=args.logging_steps,  # Save the model every 50 steps
        save_total_limit=1,  # Keep a maximum of 3 checkpoints
        logging_steps=args.logging_steps,  # Log(output) after every 10 steps
        learning_rate=3e-5,  # Initial learning rate
        weight_decay=0.01  # L2 weight decay (regularization)
    )

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger("my_logger", os.path.join(args.output_dir, "trainer.log"))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        tokenizer=tokenizer,
        callbacks=[StringOutputEvaluator(dataset_eval, tokenizer, logger, args.output_dir)]
    )

    # Train the model
    trainer.train()


# Initialize Trainer
if __name__ == '__main__':
    main()
