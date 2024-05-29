import sys

from models.tbcc.tree import trans_to_sequences
from tqdm import tqdm

sys.setrecursionlimit(1000000)


def prepared_data(records, max_seq_length=510, test_records=None):
    """
    Main function to read files and prepare the data.
    """
    vocabulary = {"<pad>": 0, "<unk>": 1, "<cls>": 2, "<sep>": 3}
    inverse_vocabulary = ['<pad>', '<unk>', '<cls>', '<sep>']

    data, test_data = [], []
    for row in tqdm(records, total=len(records)):
        q2n = []

        sequences = trans_to_sequences(row["tree"])
        sequences = sequences[:max_seq_length-2]

        tokens_for_sents = ['<cls>'] + sequences + ['<sep>']
        for token in tokens_for_sents:
            if token not in vocabulary:
                vocabulary[token] = len(inverse_vocabulary)
                inverse_vocabulary.append(token)
            
            q2n.append(vocabulary[token])

        row["q2n"] = q2n
        row["sequences"] = sequences
        row["tokens_for_sents"] = tokens_for_sents
        data.append(row)

    if test_records and isinstance(test_records, list):
        for row in tqdm(test_records, total=len(test_records)):
            q2n = []

            sequences = trans_to_sequences(row["tree"])
            sequences = sequences[:max_seq_length-2]

            tokens_for_sents = ['<cls>'] + sequences + ['<sep>']
            for token in tokens_for_sents:
                if token not in vocabulary:
                    vocabulary[token] = len(inverse_vocabulary)
                    inverse_vocabulary.append(token)
            
                q2n.append(vocabulary[token])

            row["q2n"] = q2n
            row["sequences"] = sequences
            row["tokens_for_sents"] = tokens_for_sents
            test_data.append(row)

    return data, test_data, vocabulary, inverse_vocabulary
