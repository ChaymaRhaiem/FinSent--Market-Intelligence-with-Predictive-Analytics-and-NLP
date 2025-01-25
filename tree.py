import os

def generate_markdown_tree(root_dir, indent=""):
    """
    Generates a directory tree in markdown format without depth annotations.
    """
    tree = f"{indent}- **{os.path.basename(root_dir)}/**\n"

    try:
        items = sorted(os.listdir(root_dir))
    except PermissionError:
        return f"{indent}  - *[Access Denied]*\n"

    for item in items:
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            # Add subdirectory recursively
            tree += generate_markdown_tree(item_path, indent + "  ")
        else:
            # Add file entry
            tree += f"{indent}  - {item}\n"

    return tree

# Replace '.' with your target directory
root_directory = "."
tree_output = generate_markdown_tree(root_directory)

# Save to README.md or print
output_file = "tree_structure.md"  # Change to README.md if required
with open(output_file, "w") as f:
    f.write("# Project Directory Structure\n\n")
    f.write(tree_output)

print(f"Directory tree saved to {output_file}")
