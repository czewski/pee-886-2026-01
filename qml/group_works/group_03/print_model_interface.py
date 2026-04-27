import ast
from pathlib import Path


def _get_layer_lines(class_node: ast.ClassDef):
    layer_lines = []
    init_node = None

    for item in class_node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
            init_node = item
            break

    if init_node is None:
        return layer_lines

    for stmt in init_node.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if len(stmt.targets) != 1:
            continue

        target = stmt.targets[0]
        if not (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
            continue

        layer_name = target.attr
        layer_value = ast.unparse(stmt.value).strip()
        layer_lines.append(f"{layer_name}: {layer_value}")

    return layer_lines


def _print_model_layers(file_path: Path):
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    printed_any_class = False

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        printed_any_class = True
        print(f"\n- {node.name}")
        layer_lines = _get_layer_lines(node)
        if not layer_lines:
            print("  (no layers found in __init__)")
            continue
        for layer_line in layer_lines:
            print(f"  {layer_line}")

    if not printed_any_class:
        print("\n(no model classes found)")


def main():
    models_dir = Path(__file__).parent / "models"
    model_files = sorted(
        p for p in models_dir.glob("*.py") if p.name != "__init__.py"
    )

    print("group_03 model layers")
    print("=====================")
    for model_file in model_files:
        print(f"\n[{model_file.name}]")
        _print_model_layers(model_file)


if __name__ == "__main__":
    main()
