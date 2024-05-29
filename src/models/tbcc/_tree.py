from javalang.ast import Node


def get_token(node):
    token = ''
    primitivetype = ['int', 'byte', 'short', 'long', 'float', 'double', 'boolean', 'char']
    if isinstance(node, str):
        if node in primitivetype:
            token = 'PrimitiveType'
        elif node.isnumeric():
            token = 'IntegerLiteralExpr'
        else:
            token = 'StringLiteralExpr'
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__
    # print(node.__class__.__name__,str(node))
    # print(node.__class__.__name__, node)
    return token


def get_child(root):
    # print(root)
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    # print(sub_item)
                    yield sub_item
            elif item:
                # print(item)
                yield item

    return list(expand(children))  # list of tree objects


def get_sequences(node, sequence):
    token, children = get_token(node), get_child(node)
    sequence.append(token)
    # print(len(sequence), token)
    for child in children:
        get_sequences(child, sequence)


def trans_to_sequences(ast):
    sequence = []
    get_sequences(ast, sequence)
    return sequence


def check_length(arr, limited_size):
    if len(arr) > limited_size:
        return True
    return False
