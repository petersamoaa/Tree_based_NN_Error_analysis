import re

import javalang





def parse(code):
    tokens = list(javalang.tokenizer.tokenize(code))
    tree = javalang.parse.parse(remove_comments(code))
    return tokens, tree
