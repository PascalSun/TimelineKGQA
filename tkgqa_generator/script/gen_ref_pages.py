"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files

root = Path(__file__).parent.parent
src = root / "tkgqa_generator"

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    full_doc_path = "Code" / module_path.with_suffix(".md")

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        # parts = parts[:-1]
        # get this to be the index html under this folder
        # replace the __init__ with index
        identifier = ".".join(parts[:-1])
        parts = parts[:-1] + ("index",)
        full_doc_path = "Code" / module_path.parent / "index.md"
        if identifier == "":
            print(
                "::: tkgqa_generator",
                file=mkdocs_gen_files.open(full_doc_path, "w"),
            )
        else:
            print(
                f"::: tkgqa_generator.{identifier}",
                file=mkdocs_gen_files.open(full_doc_path, "w"),
            )
        continue
    elif parts[-1] == "__main__":
        continue

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        print(f"::: tkgqa_generator.{identifier}", file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))
