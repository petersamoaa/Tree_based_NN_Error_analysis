import sys
from collections import defaultdict

import javalang
import javalang.ast

sys.setrecursionlimit(1000000)


def _create_samples(node):
    """Create samples based on the node's children.

    Args:
        node (javalang.ast.Node): The node for which to create samples.

    Returns:
        list: A list of samples created.
    """

    samples = []
    for child in _iter_child_nodes(node):
        sample = {
            "node": _name(child),
            "parent": _name(node),
            "children": [_name(x) for x in _iter_child_nodes(child)]
        }
        samples.append(sample)

    return samples


def _traverse_tree_bfs(tree, callback):
    """Traverse the tree using breadth-first search and apply a callback.

    Args:
        tree (javalang.ast.Node): The root node of the tree.
        callback (function): The callback function to apply.
    """

    queue = [tree]
    while queue:
        current_node = queue.pop(0)
        children = list(_iter_child_nodes(current_node))
        queue.extend(children)
        callback(current_node)


def _traverse_tree(root):
    """
    Traverse a tree to produce a JSON-like structure.

    Args:
        root: The root node of the tree.

    Returns:
        A tuple containing the JSON-like structure and the number of nodes.
    """
    num_nodes = 1
    queue = [root]
    root_json = {
        "node": _name(root),
        "children": []
    }
    queue_json = [root_json]

    while queue:
        current_node = queue.pop(0)
        num_nodes += 1
        current_node_json = queue_json.pop(0)

        children = list(_iter_child_nodes(current_node))
        queue.extend(children)

        for child in children:
            child_json = {
                "node": _name(child),
                "children": []
            }
            current_node_json['children'].append(child_json)
            queue_json.append(child_json)

    return root_json, num_nodes


def _iter_child_nodes(node):
    """Generate child nodes for a given node.

    Args:
        node (javalang.ast.Node): The node for which to generate child nodes.

    Yields:
        javalang.ast.Node: A child node.
    """

    if not isinstance(node, javalang.ast.Node):
        return

    for attr, value in node.__dict__.items():
        if isinstance(value, javalang.ast.Node):
            yield value
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, javalang.ast.Node):
                    yield item


def _name(node):
    """Retrieve the name of a node's class.

    Args:
        node (javalang.ast.Node): The node for which to retrieve the class name.

    Returns:
        str: The name of the node's class.
    """

    return type(node).__name__


def prepare_nodes(data, per_node=-1, limit=-1):
    node_counts = defaultdict(int)
    node_counts[None] += 1
    samples = []

    has_capacity = lambda x: per_node < 0 or node_counts[x] < per_node
    can_add_more = lambda: limit < 0 or len(samples) < limit

    for root in data:
        new_samples = [
            {
                'node': _name(root),
                'parent': None,
                'children': [_name(x) for x in _iter_child_nodes(root)]
            }
        ]
        gen_samples = lambda x: new_samples.extend(_create_samples(x))
        _traverse_tree_bfs(root, gen_samples)
        for sample in new_samples:
            if has_capacity(sample['node']):
                samples.append(sample)
                node_counts[sample['node']] += 1
            if not can_add_more():
                break
        if not can_add_more():
            break

    return node_counts, samples


def prepare_trees(data, minsize=-1, maxsize=-1):
    samples = []

    for row in data:
        root = row["tree"]
        sample, size = _traverse_tree(root)
        if minsize > 0 and maxsize > 0:
            if size > maxsize or size < minsize:
                continue
        elif minsize > 0 and not maxsize > 0:
            if size < minsize:
                continue
        elif not minsize > 0 and maxsize > 0:
            if size > maxsize:
                continue

        row["_tree"] = sample
        samples.append(row)

    return samples
