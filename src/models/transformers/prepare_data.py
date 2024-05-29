import sys

from tqdm import tqdm
from transformers import AutoTokenizer

from models.tbcc.tree import trans_to_sequences

sys.setrecursionlimit(1000000)


def prepared_data(records, max_seq_length=1024, test_records=None, model_name_or_path="google-bert/bert-base-cased"):
    """
    Main function to read files and prepare the data.
    """
    past_new_tokens = [
        'ForStatement', 'WhileStatement', 'DoStatement', 'SwitchStatement', 'IfStatement',
        'End', 'MethodDeclaration', 'ConstructorDeclaration', 'BlockStatement', 'block'
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, max_length=max_seq_length)
    tokenizer.model_max_length = max_seq_length
    tokenizer.add_tokens(past_new_tokens)

    data, test_data = [], []
    for row in tqdm(records, total=len(records)):
        code = row["code"]
        past = " ".join(trans_to_sequences(row["tree"]))

        code_tokenized = tokenizer(code)
        past_tokenized = tokenizer(past)

        row["code_input_ids"] = code_tokenized["input_ids"]
        row["code_attention_mask"] = code_tokenized["attention_mask"]

        row["past_input_ids"] = past_tokenized["input_ids"]
        row["past_attention_mask"] = past_tokenized["attention_mask"]

        data.append(row)

    if test_records and isinstance(test_records, list):
        for row in tqdm(test_records, total=len(test_records)):
            code = row["code"]
            past = " ".join(trans_to_sequences(row["tree"]))

            code_tokenized = tokenizer(code)
            past_tokenized = tokenizer(past)

            row["code_input_ids"] = code_tokenized["input_ids"]
            row["code_attention_mask"] = code_tokenized["attention_mask"]

            row["past_input_ids"] = past_tokenized["input_ids"]
            row["past_attention_mask"] = past_tokenized["attention_mask"]

            test_data.append(row)

    return data, test_data, tokenizer
