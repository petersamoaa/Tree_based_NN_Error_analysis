import os

import javalang


def count_words_in_file(file_path):
    """
    Count the number of words in a file.

    :param file_path: Path to the file.
    :return: The word count of the file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        contents = file.read()
        words = contents.split()
        return len(words)


def find_java_files(directory, word_count_threshold=350):
    """
    Walk through a directory and retrieve all .java files with their names and full paths.

    :param directory: The root directory to search in.
    :param word_count_threshold: Threshold for categorizing files as 'long' or 'short'.
    :return: A list of tuples, each containing the file name and its full path.
    """
    java_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".java"):
                full_path = os.path.join(root, file)
                word_count = count_words_in_file(full_path)
                file_type = 'long' if word_count >= word_count_threshold else 'short'
                java_files.append((file, full_path, file_type))

    return java_files


def parse_java_code_to_ast(java_code, logger=None, jid=None):
    """
    Parses Java code into an AST and converts it to numerical format.

    :param java_code: The Java code as a string.
    :param logger:
    :return: AST tree representing the Java code.
    """
    try:
        tree = javalang.parse.parse(java_code)
    except Exception as e:
        if logger:
            logger.info(f"Syntax error when parsing Java code {jid}: {e}")
        else:
            print(f"Syntax error when parsing Java code {jid}: {e}")

        tree = None

    return tree
