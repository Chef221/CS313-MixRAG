import json, os, re
import sys

# from span_selection_utils import *
# from program_generation_utils import *
import math 
"""MathQA utils.
"""
import argparse
import collections
import json
import numpy as np
import os
import re
import string
import sys
import math
import tqdm
from sympy import simplify
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union, Optional
import json
import argparse
import string
import re
from scipy.optimize import linear_sum_assignment

all_ops = ["add", "subtract", "multiply", "divide", "exp"]

sys.path.insert(0, '../utils/')
max_seq_length = 512
max_program_length = 30

class MathQAExample(
        collections.namedtuple(
            "MathQAExample",
            "id original_question question_tokens answer"
        )):

    def convert_single_example(self, *args, **kwargs):
        return convert_single_mathqa_example(self, *args, **kwargs)


def tokenize(tokenizer, text, apply_basic_tokenization=False):
    """Tokenizes text, optionally looking up special tokens separately.

    Args:
      tokenizer: a tokenizer from bert.tokenization.FullTokenizer
      text: text to tokenize
      apply_basic_tokenization: If True, apply the basic tokenization. If False,
        apply the full tokenization (basic + wordpiece).

    Returns:
      tokenized text.

    A special token is any text with no spaces enclosed in square brackets with no
    space, so we separate those out and look them up in the dictionary before
    doing actual tokenization.
    """

    _SPECIAL_TOKENS_RE = re.compile(r"^<[^ ]*>$", re.UNICODE)

    tokenize_fn = tokenizer.tokenize
    if apply_basic_tokenization:
        tokenize_fn = tokenizer.basic_tokenizer.tokenize

    tokens = []
    for token in text.split(" "):
        if _SPECIAL_TOKENS_RE.match(token):
            if token in tokenizer.get_vocab():
                tokens.append(token)
            else:
                tokens.append(tokenizer.unk_token)
        else:
            tokens.extend(tokenize_fn(token))

    return tokens



def convert_single_mathqa_example(example, tokenizer, max_seq_length):
    """Converts a single MathQAExample into an InputFeature."""
    # input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_text_encoded = tokenizer.encode_plus(example.original_question,
                                    max_length=max_seq_length,
                                    pad_to_max_length=True)
    input_ids = input_text_encoded["input_ids"]
    input_mask = input_text_encoded["attention_mask"]
    
    label_encoded = tokenizer.encode_plus(str(example.answer),
                                    max_length=16,
                                    pad_to_max_length=True)
    label_ids = label_encoded["input_ids"]
    
    this_input_feature = {
        "uid": example.id,
        "tokens": example.question_tokens,
        "question": example.original_question,
        "input_ids": input_ids,
        "input_mask": input_mask,
        "label_ids": label_ids,
        "label": str(example.answer)   
    }

    return this_input_feature


def read_mathqa_entry(entry, tokenizer, entity_name):
    if entry["qa"][entity_name] != "span_selection":
        return None
    
    
    context = ""
    for idx in entry["model_input"]:
        if type(idx) == int:
            context += entry["paragraphs"][idx][:-1]
            context += " "

        else:
            context += entry["table_description"][idx][:-1]
            context += " "
    
    question = entry["qa"]["question"]
    this_id = entry["uid"]
    
    original_question = f"Question: {question} Context: {context.strip()}"
    if "answer" in entry["qa"]:
        answer = entry["qa"]["answer"]
    else:
        answer = ""
    if type(answer) != str:
        answer = str(int(answer))

    original_question_tokens = original_question.split(' ')


    return MathQAExample(
        id=this_id,
        original_question=original_question,
        question_tokens=original_question_tokens,
        answer=answer)


def read_examples(input_path, tokenizer):
    """Read a json file into a list of examples."""
    with open(input_path) as input_file:
        input_data = json.load(input_file)

    examples = []
    for entry in tqdm(input_data):
        examples.append(read_mathqa_entry(entry, tokenizer))
    return input_data, examples

def convert_examples_to_features(examples,
                                 tokenizer,
                                 max_seq_length,
                                 verbose=True):
    """Converts a list of DropExamples into InputFeatures."""
    res = []
    for (example_index, example) in enumerate(examples):
        feature = example.convert_single_example(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length
            )     
        res.append(feature)
    return res


def write_predictions(all_predictions, output_prediction_file):
    """Writes final predictions in json format."""

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")



# From here through get_metric was originally copied from:
# https://github.com/allenai/allennlp-reading-comprehension/blob/master/allennlp_rc/eval/drop_eval.py
def _remove_articles(text: str) -> str:
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)


def _white_space_fix(text: str) -> str:
    return " ".join(text.split())


EXCLUDE = set(string.punctuation)


def _remove_punc(text: str) -> str:
    if not _is_number(text):
        return "".join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text


def _lower(text: str) -> str:
    return text.lower()


def _tokenize(text: str) -> List[str]:
    #增加了中文符号–
    return re.split(" |-", text)


def _normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    parts = [
        _white_space_fix(_remove_articles(_normalize_number(_remove_punc(_lower(token)))))
        for token in _tokenize(text)
    ]
    parts = [part for part in parts if part.strip()]
    normalized = " ".join(parts).strip()
    return normalized


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(float(text))
    else:
        return text


def _answer_to_bags(
    answer: Union[str, List[str], Tuple[str, ...]]
) -> Tuple[List[str], List[Set[str]]]:
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: List[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> List[float]:
    """
    Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
    between them and gets maximum metric values over all the answers.
    """
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, pred_item):
                scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 =(
        (2 * precision * recall) / (precision + recall)
        if not (precision == 0.0 and recall == 0.0)
        else 0.0
    )
    return f1


def _match_numbers_if_present(gold_bag: Set[str], predicted_bag: Set[str]) -> bool:
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if _is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if _is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def get_span_selection_metrics(predicted, gold):
    predicted_spans, predicted_token_bags = _answer_to_bags(predicted)
    gold_spans, gold_token_bags = _answer_to_bags(gold)
    """
    Takes a predicted answer and a gold answer (that are both either a string or a list of
    strings), and returns exact match and the DROP F1 metric for the prediction.  If you are
    writing a script for evaluating objects in memory (say, the output of predictions during
    validation, or while training), this is the function you want to call, after using
    :func:`answer_json_to_strings` when reading the gold answer from the released data file.
    """
    # 检查是否包含数值
    if len(predicted_spans) > 0 and len(gold_spans) > 0:
        if _is_number(predicted_spans[0]) and _is_number(gold_spans[0]):
            try:
                pred_num = float(predicted_spans[0])
                gold_num = float(gold_spans[0])
                # 检查数字是否是小数且小数点后有6位
                if isinstance(gold_num, float) and len(str(gold_num).split('.')[-1]) >= 6:
                    # 将数字四舍五入到小数点后5位
                    gold_num_round5 = round(gold_num, 5)
                else:
                    gold_num_round5=gold_num
                
                if abs(abs(pred_num) - abs(gold_num_round5)) == 0.0:
                    exact_match = 1.0
                    f1 = 1.0
                    return exact_match, f1
                elif (abs(abs(pred_num) - abs(gold_num_round5)) > 0.0) and (abs(abs(pred_num)/100 - abs(gold_num_round5)) < 0.000005):
                    exact_match = 1.0
                    f1 = 1.0
                    return exact_match, f1
                elif (abs(abs(pred_num) - abs(gold_num_round5)) > 0.0) and (abs(abs(pred_num)*100 - abs(gold_num_round5)) < 0.0005):
                    exact_match = 1.0
                    f1 = 1.
                    return exact_match, f1
                elif (abs(abs(pred_num) - abs(gold_num_round5)) > 0.0) and (abs(abs(pred_num)/100 - abs(gold_num_round5)) > 0.000005) and (abs(abs(pred_num)*100 - abs(gold_num_round5)) > 0.0005) and (abs(abs(pred_num)*1000000 - abs(gold_num_round5)) == 0.0):
                    exact_match = 1.0
                    f1 = 1.0
                    return exact_match, f1
                elif (abs(abs(pred_num) - abs(gold_num_round5)) > 0.0) and (abs(abs(pred_num)/100 - abs(gold_num_round5)) > 0.000005) and (abs(abs(pred_num)*100 - abs(gold_num_round5)) > 0.0005) and (abs(abs(pred_num)/1000000 - abs(gold_num_round5)) == 0.0):
                    exact_match = 1.0
                    f1 = 1.0
                    return exact_match, f1
            except ValueError:
                pass 

    if set(predicted_token_bags[0]) == set(gold_token_bags[0]) and len(predicted_token_bags[0]) == len(gold_token_bags[0]):
        exact_match = 1.0
    else:
        exact_match = 0.0

    f1_per_bag = _align_bags(predicted_token_bags, gold_token_bags)
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return exact_match, f1

def span_selection_evaluate(all_preds, all_filename_id, test_file):
    '''
    Exact Match
    F1
    '''
    results = []
    exact_match, f1 = 0, 0
    with open(test_file) as f_in:
        data_ori = json.load(f_in)

    data_dict = {}
    for each_data in data_ori:
        assert each_data["uid"] not in data_dict
        data_dict[each_data["uid"]] = each_data["qa"]["answer"]

    for pred, uid in zip(all_preds, all_filename_id):
        gold = data_dict[uid]
        if type(gold) != str:
            gold = str(int(gold))

        cur_exact_match, cur_f1 = get_span_selection_metrics(pred, gold)
        
        result = {"uid": uid, "answer": gold, "predicted_answer": pred, "exact_match": exact_match, "f1": f1}
        results.append(result)

        exact_match += cur_exact_match
        f1 += cur_f1

    exact_match = exact_match / len(all_preds)
    f1 = f1 / len(all_preds)
    return exact_match, f1


def combine_predictions(span_selection_json_in, program_generation_json_in, test_file_json_in, output_dir):
    span_selection_data = json.load(open(span_selection_json_in))
    program_generation_data = json.load(open(program_generation_json_in))
    orig_data = json.load(open(test_file_json_in))

    prediction_dict = {}
    for example in span_selection_data + program_generation_data:
        uid = example["uid"]
        pred_ans = example["predicted_ans"]
        pred_program = example["predicted_program"]

        if uid in prediction_dict:
            print(f"uid {uid} already in prediction_dict")
        else:
            prediction_dict[uid] = {
                "uid": uid,
                "predicted_ans": pred_ans,
                "predicted_program": pred_program
            }
    
    output_data = []
    for example in orig_data:
        output_data.append(prediction_dict[example["uid"]])

    mode = "dev" if "dev" in test_file_json_in else "test"
    output_file = os.path.join(output_dir, f"{mode}_predictions.json")
    json.dump(output_data, open(output_file, "w"), indent=4)

    print(f"{mode}: Combine {len(span_selection_data)} examples from span selection output, {len(program_generation_data)} examples from program generation output. The prediction are generated in {output_file}")

    return prediction_dict


def evaluation_prediction_result(pred_ans,gold_ans):
    exact_match_total, f1_total = 0, 0
    true_or_flase_list=[]
    for i in range(len(pred_ans)):
        # predict和true_ans对应 RECAP 预测值和真实值
        predict=pred_ans[i]
        true_ans=gold_ans[i]

        exact_acc, f1_acc = get_span_selection_metrics(predict, true_ans)
        if exact_acc==1:
            true_or_flase_list.append('True')
        else:
            true_or_flase_list.append('False')
        exact_match_total += exact_acc
        f1_total += f1_acc
    num_examples=len(pred_ans)
    exact_match_score, f1_score = exact_match_total / num_examples, f1_total / num_examples
    print(f"Exact Match Score: {exact_match_score}, F1 Score: {f1_score}")

    return exact_match_score, f1_score, true_or_flase_list