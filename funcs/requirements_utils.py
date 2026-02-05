def is_package_importable(requirement: str) -> bool:
    """
    Check if the package specified in the requirement string is importable.
    """
    name = _extract_module_name(requirement)
    if not name:
        return False
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def _iter_requirements_lines(requirements_path):
    """
    Generator that yields non-empty, non-comment lines from the requirements file.
    """
    with open(requirements_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                yield line


def install_missing(requirements_path, quiet=False):
    """
    Install missing packages listed in the requirements file using pip.
    """
    missing = []
    for requirement in _iter_requirements_lines(requirements_path):
        if not is_package_importable(requirement):
            missing.append(requirement)

    if missing:
        import subprocess
        import sys
        command = [sys.executable, '-m', 'pip', 'install'] + missing
        if quiet:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(command)


# Existing functionality for backward compatibility

def install_missing_packages(requirements_path):
    install_missing(requirements_path)
