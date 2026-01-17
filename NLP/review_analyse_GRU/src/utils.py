from pathlib import Path


def safe_path_join(base_path, *paths):
    """安全路径拼接，防止路径遍历攻击"""
    base_path = Path(base_path).resolve()
    full_path = base_path.joinpath(*paths).resolve()
    if not str(full_path).startswith(str(base_path)):
        raise ValueError("Path traversal detected")
    return full_path