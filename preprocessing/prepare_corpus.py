from pathlib import Path
import random
import ast
import json


RAW_DIR = Path("data/raw")
TRAIN_DIR = Path("data/processed/train")
EVAL_DIR = Path("data/processed/generation_eval")
STATS_DIR = Path("data/stats")

TRAIN_RATIO = 0.8
SEED = 42


def should_skip(path: Path):

    skip_dirs = {
        "tests",
        "__pycache__",
        ".github",
        ".devcontainer",
    }

    return any(
        part.lower() in skip_dirs
        for part in path.parts
    )

def clean_code(code: str, strip_non_ascii=False):

    if strip_non_ascii:
        code = code.encode(
            "ascii",
            errors="ignore"
        ).decode("ascii")

    return code

def get_python_files():
    """
    Find all .py files inside data/raw.
    """

    files = []

    for path in RAW_DIR.rglob("*.py"):

        if should_skip(path):
            continue

        files.append(path)

    return files


def extract_functions(code: str):
    """
    Extract complete function definitions using Python AST.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    functions = []

    for node in ast.walk(tree):

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):

            lines = code.splitlines(keepends=True)

            function_code = "".join(
                lines[node.lineno - 1: node.end_lineno]
            )

            functions.append(function_code)

    return functions


def main():

    random.seed(SEED)

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    files = get_python_files()

    print(f"Found {len(files)} Python files")

    random.shuffle(files)

    split_index = int(len(files) * TRAIN_RATIO)

    train_files = files[:split_index]
    eval_files = files[split_index:]

    print(f"Training files: {len(train_files)}")
    print(f"Evaluation files: {len(eval_files)}")

    total_train_chars = 0
    total_eval_functions = 0

    # -------------------------
    # TRAINING FILES
    # -------------------------

    for path in train_files:

        code = path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        code = clean_code(code, strip_non_ascii=True)

        output_path = TRAIN_DIR / path.name

        output_path.write_text(
            code,
            encoding="utf-8"
        )

        total_train_chars += len(code)

    # -------------------------
    # EVALUATION FUNCTIONS
    # -------------------------

    eval_id = 0

    for path in eval_files:

        code = path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        code = clean_code(code)

        functions = extract_functions(code)

        for function in functions:

            if not function.strip():
                continue

            eval_id += 1

            output_path = EVAL_DIR / f"eval_{eval_id:04d}.py"

            output_path.write_text(
                function,
                encoding="utf-8"
            )

            total_eval_functions += 1

    # -------------------------
    # STATS
    # -------------------------

    stats = {
        "total_source_files": len(files),
        "training_files": len(train_files),
        "heldout_files": len(eval_files),
        "training_characters": total_train_chars,
        "evaluation_functions": total_eval_functions,
    }

    stats_path = STATS_DIR / "corpus_stats.json"

    stats_path.write_text(
        json.dumps(stats, indent=4),
        encoding="utf-8"
    )

    print("\nCorpus preparation complete!")
    print(json.dumps(stats, indent=4))


if __name__ == "__main__":
    main()