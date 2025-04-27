from tree_sitter import Language, Parser
import tree_sitter_python
import tree_sitter_c_sharp
import os
from loguru import logger

logger.add("bot.log", rotation="1 MB", level="INFO")


def parse_code(file_path: str):
    if file_path.endswith(".py"):
        lang = Language(tree_sitter_python.language())
    elif file_path.endswith(".cs"):
        lang = Language(tree_sitter_c_sharp.language())
    else:
        return []

    parser = Parser()
    parser.set_language(lang)

    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()

    tree = parser.parse(code.encode("utf-8"))
    root = tree.root_node

    functions = []
    query = lang.query("(function_definition name: (identifier) @name) @func")
    for capture, _ in query.captures(root):
        if capture.type == "identifier":
            functions.append(capture.text.decode("utf-8"))

    return functions


async def list_functions(directory_path: str):
    results = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith((".py", ".cs")):
                file_path = os.path.join(root, file)
                functions = parse_code(file_path)
                if functions:
                    results.append(f"{file}: {', '.join(functions)}")
    return results
