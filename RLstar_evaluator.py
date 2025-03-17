import argparse
from collections import defaultdict
import glob
import json
import os
import shutil

import csv
import numpy as np
from transformers import LogitsProcessor, LogitsProcessorList, GPT2Tokenizer, GPT2LMHeadModel
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from mydataset import make_input_data_generate, make_input_data, EvalDataset

RNG = np.random.RandomState(0)
MAX_LENGTH = 30


class CustomLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, valid_output, step):
        self.tokenizer = tokenizer
        self.valid_output = valid_output
        self.step = step
        self.pos = 0

    def __call__(self, input_ids, scores):
        if self.pos in self.valid_output[self.step].keys():
            valid_logits = self.valid_output[self.step][self.pos]
            # Suppose scores is a tensor that you want to modify
            # Create a mask initialized to True
            mask = torch.full(scores.shape, False).to(scores.device)
            # Set mask values to True at valid indices
            mask[:, valid_logits] = True
            # Set scores to -inf where mask is False
            scores = scores.masked_fill(~mask, -np.inf)
            # Custom logic to modify the scores.
            # For demonstration, let's simply return the scores unmodified.
            self.pos += 1
        return scores


# Example usage
class Evaluator(object):
    def __init__(self, model, dataloader, dataloader_full, tokenizer, batch_size=128, temperature=1):
        self.model = model
        self.dataloader = dataloader
        self.dataloader_full = dataloader_full
        self.batch_size = batch_size
        self.temperature = temperature
        self.tokenizer = tokenizer
        self.valid_output = self.generate_valid_output()

    def softmax_with_temperature(self, weights, temperature=1.0):
        if len(weights) == 1:
            return np.array([1.])
        # Convert weights to a numpy array
        weights = np.array(weights)

        # Adjust weights by the temperature

        # print(f"temperature={temperature}")
        adjusted_weights = weights / temperature

        # Subtract the maximum value from adjusted weights for numerical stability
        max_adjusted_weights = np.max(adjusted_weights)
        stabilized_weights = adjusted_weights - max_adjusted_weights

        # Calculate the exponential values for each stabilized weight
        exp_values = np.exp(stabilized_weights)

        exp_values = exp_values.astype(np.float64)

        # Sum of all the exponential values
        sum_exp_values = np.sum(exp_values)

        # Compute the softmax by dividing each exponential value by the sum
        probabilities = exp_values / sum_exp_values

        return probabilities

    def generate_all(self, dataloader, next_step):
        S = []
        for j, batch in enumerate(dataloader):
            for i in batch[f's{next_step}']:
                S.append(i)
        # print(type(S))
        return S

    def generate_valid_output(self):
        valid_output = {}
        for batch in self.dataloader_full:
            for j, (jj, sjs) in enumerate(batch.items()):
                if j not in valid_output.keys():
                    valid_output[j] = {}
                for sj in sjs:
                    sj_encoding = self.tokenizer.encode(sj)
                    for k, sk in enumerate(sj_encoding):
                        if k not in valid_output[j]:
                            valid_output[j][k] = []
                        if sk not in valid_output[j][k]:
                            valid_output[j][k].append(sk)

        for batch in self.dataloader_full:
            for j, (jj, sjs) in enumerate(batch.items()):
                for sj in sjs:
                    sj_encoding = self.tokenizer.encode(sj)
                    for k, sk in enumerate(sj_encoding):
                        valid_output[j][k] = sorted(valid_output[j][k])
                    valid_output[j][len(sj_encoding)] = [self.tokenizer.eos_token_id]

        return valid_output

    def process_logits(self, lm_logits, next_step):
        next_text = ''
        for ikey, valid_indices in self.valid_output[next_step].items():
            if ikey < lm_logits.shape[0]:
                valid_logits = lm_logits[ikey, valid_indices]
                probs = self.softmax_with_temperature(valid_logits.cpu().numpy(), temperature=self.temperature).tolist()
                # print(f"ikey={ikey}, valid_indices={tokenizer.decode(valid_indices)}, probs={probs}, valid_logits={valid_logits}")
                next_ids = RNG.choice(valid_indices, p=probs)
                next_text += self.tokenizer.decode(next_ids)
        return next_text

    def generate_one_step(self, input_text, target_text, next_step):
        # print(f"input_text={input_text}")

        input_items_new = {}
        for ikey in ['input_ids', 'attention_mask', 'position_ids']:
            input_items_new[ikey] = []
        for t1, t2 in zip(input_text, target_text):
            input_item = make_input_data(t1, t2, self.tokenizer)
            for ikey in input_items_new.keys():
                input_items_new[ikey].append(input_item[ikey].cpu().numpy())
        for ikey in input_items_new.keys():
            input_items_new[ikey] = torch.tensor(input_items_new[ikey]).to(self.model.device)

        output_for_logits = self.model.forward(**input_items_new, return_dict=True)

        lm_logits = output_for_logits.logits
        position_start = len(self.tokenizer.encode(input_text[0]))
        position_end = position_start + len(self.tokenizer.encode(target_text[0]))

        output_text = []
        for i in range(lm_logits.shape[0]):
            next_text = self.process_logits(lm_logits[i, position_start:position_end, :], next_step=next_step)
            output_text.append(next_text)

        return output_text

    def generate_one_step_generate(self, input_text, target_text, next_step):

        # print(f"input_text={input_text}")
        max_length = 40
        input_items = {}
        for ikey in ['input_ids', 'attention_mask', 'position_ids']:
            input_items[ikey] = []
        for t in input_text:
            input_item = make_input_data_generate(t, self.tokenizer)
            for ikey in input_items.keys():
                input_items[ikey].append(input_item[ikey].cpu().numpy())
        for ikey in input_items.keys():
            input_items[ikey] = torch.tensor(input_items[ikey]).to(self.model.device)

        logit_processor = LogitsProcessorList([
            CustomLogitsProcessor(tokenizer=self.tokenizer, valid_output=self.valid_output, step=next_step)
            # Add any additional custom processors
        ])

        output = self.model.generate(**input_items,
                                     max_length=max_length,
                                     logits_processor=logit_processor,
                                     eos_token_id=self.tokenizer.eos_token_id,
                                     do_sample=True,
                                     temperature=float(self.temperature)
                                     )

        output_text = []
        for o in output:
            output_text_i = self.tokenizer.decode(o[input_items['input_ids'].shape[1]:-1]).replace(" ", "")
            output_text.append(output_text_i)

        return output_text

    def generate_one_step_enumerate(self, input_text, next_step):
        global RNG
        # print(f"input_text={input_text}")
        possible_output = self.generate_all(self.dataloader_full, next_step)
        # RNG.shuffle(possible_output)
        # print("P", possible_output)
        min_loss = np.inf  # infinite
        best_text = None
        best_idx = -1

        input_data_buffer = defaultdict(list)
        next_text_buffer = []

        probability_dist = []
        for i, next_text in enumerate(possible_output):
            input_data = make_input_data(input_text, next_text, self.tokenizer, self.dataloader.dataset.max_length)
            next_text_buffer.append(next_text)
            for ikey, ival in input_data.items():
                ival = torch.reshape(ival, (1, ival.shape[0]))
                input_data_buffer[ikey].append(ival)
            # inputs = tokenizer.encode(combined_text, return_tensors='pt').to(model.device)
            if (i + 1) % self.batch_size == 0 or (i + 1) == len(possible_output):
                input_data_batch = {}
                for ikey, ival in input_data_buffer.items():
                    input_data_batch[ikey] = torch.concatenate(ival, dim=0).to(self.model.device)

                outputs = self.model.forward(**input_data_batch, return_dict=True)
                lm_logits = outputs.logits
                labels = input_data_batch['labels']
                for i in range(lm_logits.shape[0]):
                    for j in range(lm_logits.shape[1]):
                        difference = np.sum((lm_logits[i, j] != lm_logits[0, j]).cpu().numpy())
                        # print(f"j={j}, difference={difference}")

                labels = labels.to(lm_logits.device)
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()

                for k, next_text_k in enumerate(next_text_buffer):
                    cur_shift_logits = shift_logits[k:k + 1]
                    cur_shift_labels = shift_labels[k:k + 1]
                    cur_loss = loss_fct(cur_shift_logits.view(-1, cur_shift_logits.size(-1)), cur_shift_labels.view(-1))
                    probability_dist.append((next_text_k, float(1 / cur_loss)))
                    # probability_dist.append((next_text_k, float(torch.exp(-cur_loss))))

                    if cur_loss < min_loss:
                        min_loss = cur_loss
                        best_text = next_text_k
                        best_idx = i
                input_data_buffer.clear()
                next_text_buffer.clear()
        weights = [x[1] for x in probability_dist]
        next_items = [x[0] for x in probability_dist]
        normalized_weights = self.softmax_with_temperature(weights, temperature=self.temperature)
        best_text_stochastic = RNG.choice(next_items, p=normalized_weights)
        best_idx_stochastic = np.argmax(normalized_weights)
        return best_text_stochastic

    # Example usage
    def evaluate_per_step(self, etype="logit_prob"):
        self.model.eval()

        with torch.no_grad():
            # possible_output =  generate_all(dataloader)
            # np.random.shuffle(possible_output)
            # print("P",possible_output)
            result = []
            for current_step in range(self.dataloader.dataset.start_step,
                                      self.dataloader.dataset.n):  # dataloader.dataset.n
                total_correct = 0
                total = 0
                # for _ in range(10):
                # print("----------------------------------------")
                for j, input_raw in enumerate(self.dataloader):
                    # print(f"input_raw = {input_raw}")
                    input_text = input_raw[f's{current_step}']
                    target_text = input_raw[f's{current_step + 1}']
                    if etype == 'enumerate':
                        best_text = []
                        for i in range(len(input_text)):
                            input_text_i = input_text[i]
                            best_text_i = self.generate_one_step_enumerate(input_text_i, next_step=current_step + 1)
                            best_text.append(best_text_i)

                    elif etype == 'generate':
                        best_text = self.generate_one_step_generate(input_text, target_text, next_step=current_step + 1)

                    elif etype == "logit_prob":
                        best_text = self.generate_one_step(input_text, target_text, next_step=current_step + 1)
                    else:
                        best_text = []
                        assert 0
                    # if best_text == target_text:
                    #    total_correct += 1
                    for i, t in enumerate(best_text):
                        total += 1
                        if t == target_text[i]:
                            total_correct += 1
                        else:
                            pass
                            # print( f"input_text={input_text[i]}, target_text={target_text[i]}, best_text={best_text[i]}, total_correct={total_correct}")
                accuracy = total_correct / total
                print(f"step={current_step + 1}, accuracy={accuracy:.4f}")
                result.append({'step': current_step + 1, 'accuracy': accuracy})
            return result

    def evaluate_cot(self, repeat=1, etype="logit_prob"):
        print("------------------------------------")
        print(f"etype={etype}")
        self.model.eval()
        # total_correct = 0
        # total = 0

        correct_path = []
        full_correct_path = []
        all_path = []
        with torch.no_grad():
            # possible_output =  generate_all(dataloader)
            # np.random.shuffle(possible_output)
            # print("P",possible_output)
            for k in range(repeat):
                print(f"iteration:{k}")
                result = []
                for j, input_raw in enumerate(self.dataloader):
                    input_text = input_raw[f's0']
                    target_text = input_raw[f's{self.dataloader.dataset.n}']
                    trajectory = [[t] for t in input_text]
                    correct_all = [True for _ in input_text]
                    for current_step in range(self.dataloader.dataset.n):
                        next_target_text = input_raw[f's{current_step + 1}']
                        # input_text = generate_one_step(input_text, model, tokenizer, dataloader, dataloader_full,
                        #                               next_step=current_step+1, batch_size=batch_size, temperature=temperature)

                        if etype == 'enumerate':
                            best_text = []
                            for i in range(len(input_text)):
                                input_text_i = input_text[i]
                                best_text_i = self.generate_one_step_enumerate(input_text_i, next_step=current_step + 1)
                                best_text.append(best_text_i)

                        elif etype == 'generate':
                            input_text = self.generate_one_step_generate(input_text, next_target_text,
                                                                         next_step=current_step + 1)

                        elif etype == "logit_prob":
                            input_text = self.generate_one_step(input_text, next_target_text,
                                                                next_step=current_step + 1)

                        for ti in range(len(trajectory)):
                            traj = trajectory[ti]
                            t = input_text[ti]
                            t2 = next_target_text[ti]
                            traj.append(t)
                            if t != t2:
                                correct_all[ti] = False

                    for ti in range(len(trajectory)):
                        traj = trajectory[ti]
                        t = target_text[ti]
                        c = correct_all[ti]
                        if c:
                            full_correct_path.append(traj)
                        if traj[-1] == t:
                            # total_correct += 1
                            correct_path.append(traj)
                        all_path.append(traj)
                        # total += 1
            accuracy = len(correct_path) / len(all_path)
            print(f"accuracy={accuracy:.4f}")
            return accuracy, full_correct_path, correct_path, all_path


def main():
    parser = argparse.ArgumentParser(description='Evaluate model with checkpoint and dataset paths.')
    parser.add_argument('--model_name', type=str, default='./models/gpt2', help='Pre-trained model name or path')
    parser.add_argument('--ckpt_path', type=str,
                        default="./results/iter_result_0_01_4096",
                        help='Path to the checkpoint file.')
    parser.add_argument('--dataset_path', type=str, default="./data/zip2_3_gt.csv",
                        help='Path to the dataset file.')

    parser.add_argument('--dataset_path_full', type=str, default="./data/zip2_3_gt.csv",
                        help='Path to the dataset file.')

    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to the dataset file.')
    parser.add_argument('--repeat', type=int, default=1,
                        help='Path to the dataset file.')
    parser.add_argument('--temperature', type=float, default=0.01,
                        help='Path to the dataset file.')

    args = parser.parse_args()
    print(f"args:{args}")

    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Assuming EvalDataset is definedgg elsewhere in your code

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)

    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'eos_token': '[EOS]'})
    tokens = json.load(open(args.dataset_path_full.replace(".csv", "_token.json"), "r"))
    special_tokens_dict = {'additional_special_tokens': tokens}
    tokenizer.add_special_tokens(special_tokens_dict)

    # tokenizer.pad_token = tokenizer.eos_token
    checkpoint_path = sorted(glob.glob(os.path.join(args.ckpt_path, "checkpoint-*")))[-1]
    model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
    model.to(device)
    model.debug = True
    # Load the dataset and create the dataloader
    dataloader = DataLoader(EvalDataset(args.dataset_path, tokenizer), batch_size=24,
                            shuffle=False)  # Multiple Batch Size requires paddings
    dataloader_full = DataLoader(EvalDataset(args.dataset_path_full, tokenizer), batch_size=24,
                                 shuffle=False)  # Multiple Batch Size requires paddings

    temperature = args.temperature
    evaluator = Evaluator(model, dataloader, dataloader_full, tokenizer, temperature=temperature)

    accuracy_per_step_logit_prob = evaluator.evaluate_per_step(etype='logit_prob')

    accuracy_logit_prob, full_correct_path, correct_path, all_path = evaluator.evaluate_cot(repeat=args.repeat,
                                                                                            etype="logit_prob")
    # accuracy_enumerate, correct = evaluate(model, dataloader, dataloader_full, batch_size=128, repeat=1, temperature=temperature, etype="enumerate")

    print("--------------------------------------------------")
    print(f"accuracy_per_step:{accuracy_per_step_logit_prob}")
    # if args.enumerate:
    #    print(f"accuracy_per_step_enumerate:{accuracy_per_step_enumerate}")

    print(f"accuracy:{accuracy_logit_prob}")
    if args.output_path is not None:
        if accuracy_logit_prob >= 1.0:
            f = open(os.path.join(os.path.dirname(args.output_path), "complete.txt"), "w")
        else:
            f = open(os.path.join(os.path.dirname(args.output_path), "incomplete.txt"), "w")
    # print(f"accuracy_enumerate:{accuracy_enumerate}")

    if args.output_path is not None:
        original_row_num = dataloader.dataset.data.shape[0]
        new_rows_num = len(correct_path)
        new_rows_uniq_num = len(set(str(correct_path)))
        print(f"original_row_num={original_row_num}")
        print(f"new_rows_num={new_rows_num}")
        print(f"new_rows_uniq_num={new_rows_uniq_num}")
        add_correct(correct_path, args.output_path, dataloader)
        add_correct(all_path, args.output_path.replace(".csv", "_all.csv"), dataloader)
        add_correct(full_correct_path, args.output_path.replace(".csv", "_full_correct.csv"), dataloader)
        shutil.copy(args.dataset_path_full.replace(".csv", "_token.json"),
                    args.output_path.replace(".csv", "_token.json"))


def add_correct(C, file_path, dataloader):
    dataset = [list(dataloader.dataset.data.keys())]
    # for j, input_raw in enumerate(dataloader):
    #    #print(f'j={j},input_raw={input_raw}')
    #    dataset.append(list([x[0] for x in input_raw.values()]))
    for i in range(len(C)):
        dataset.append(C[i])
    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(dataset)
    print(f"Dataset saved to {file_path}")


if __name__ == '__main__':
    main()
