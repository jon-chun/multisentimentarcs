import os
from pathlib import Path
from typing import List, Set
from collections import defaultdict

# Global constant variables
ROOT_DIR = os.path.join('..', 'data')
SUBDIR_KEEP_LS = [
    "Captain_Kidd_1945",
    "Father's_Little_Dividend_1951",
    "Detour_1945",
    "Kansas_City_Confidential_1952",
    "The_Stranger_1946",
    "Royal_Wedding_1951",
    "The_Amazing_Mr_X_1948",
    "Too_Late_for_Tears_1949",
    "Vengence_Valley_1951"
]
GITIGNORE_OUTFILE = '../gitignore_pubic_domain_films.txt'
TEST_FLAG = True  # Set to False to actually write the .gitignore file

def get_all_subdirs(root_dir: str) -> List[str]:
    """
    Crawl the full subtree and generate a list of all full paths to each subdir.
    """
    root_path = Path(root_dir)
    return [str(path.relative_to(root_path)) for path in root_path.rglob('*') if path.is_dir()]

def is_path_to_keep(path: str, paths_to_keep: Set[str]) -> bool:
    """
    Check if a path or any of its parent directories should be kept.
    """
    path_parts = Path(path).parts
    for i in range(len(path_parts)):
        if os.path.join(*path_parts[:i+1]) in paths_to_keep:
            return True
    return False

def remove_kept_paths(all_paths: List[str], paths_to_keep: Set[str]) -> List[str]:
    """
    Remove paths that need to be kept from the complete list.
    """
    return [path for path in all_paths if not is_path_to_keep(path, paths_to_keep)]

def group_paths_by_depth(paths: List[str]) -> dict:
    """
    Group paths by their depth (number of directories in the path).
    """
    depth_groups = defaultdict(list)
    for path in paths:
        depth = len(Path(path).parts)
        depth_groups[depth].append(path)
    return depth_groups

def consolidate_paths(paths_to_delete: List[str], paths_to_keep: Set[str]) -> List[str]:
    """
    Consolidate paths by recursively checking and combining paths in a breadth-first manner.
    """
    depth_groups = group_paths_by_depth(paths_to_delete)
    max_depth = max(depth_groups.keys())

    consolidated = set(paths_to_delete)

    for depth in range(max_depth, 0, -1):
        for path in depth_groups[depth]:
            parent = str(Path(path).parent)
            if parent and parent != '.' and not is_path_to_keep(parent, paths_to_keep):
                siblings = [p for p in consolidated if str(Path(p).parent) == parent]
                if all(p.startswith(parent) for p in siblings):
                    for sibling in siblings:
                        consolidated.remove(sibling)
                    consolidated.add(parent)

    return sorted(consolidated)

def generate_gitignore_paths(root_dir: str, paths_to_keep: List[str]) -> List[str]:
    """
    Generate a list of paths to include in .gitignore file.
    """
    all_subdirs = get_all_subdirs(root_dir)
    paths_to_keep_set = set(os.path.join(*Path(path).parts) for path in paths_to_keep)
    paths_to_delete = remove_kept_paths(all_subdirs, paths_to_keep_set)
    consolidated_paths = consolidate_paths(paths_to_delete, paths_to_keep_set)
    return consolidated_paths

def write_gitignore_file(paths: List[str], outfile: str, test_flag: bool):
    """
    Write the generated paths to the .gitignore file or print them if in test mode.
    """
    if test_flag:
        print("\nPaths to include in .gitignore:")
        for path in paths:
            print(path)
    else:
        with open(outfile, 'w') as f:
            for path in paths:
                f.write(f"{path}\n")
        print(f"Gitignore file written to {outfile}")

if __name__ == "__main__":
    gitignore_paths = generate_gitignore_paths(ROOT_DIR, SUBDIR_KEEP_LS)
    write_gitignore_file(gitignore_paths, GITIGNORE_OUTFILE, TEST_FLAG)