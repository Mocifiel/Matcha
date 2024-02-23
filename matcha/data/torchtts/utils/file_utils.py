import contextlib
import random
import string
import filecmp
import shutil
from pathlib import Path


def _get_incomplete_path(filename):
    """Returns a temporary filename based on filename."""
    random_suffix = "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    return filename + ".incomplete" + random_suffix


@contextlib.contextmanager
def incomplete_dir(dirname):
    """Create temporary dir for dirname and rename on exit."""
    dirname = Path(dirname)
    tmp_dir = dirname.parent.joinpath(_get_incomplete_path(dirname.name))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield str(tmp_dir)
        tmp_dir.rename(dirname)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


class dircmp(filecmp.dircmp):  # noqa: N801
    """
    Compare the content of dir1 and dir2. In contrast with filecmp.dircmp, this
    subclass compares the content of files with the same path.
    """

    def phase3(self):
        """
        Find out differences between common files.
        Ensure we are using content comparison with shallow=False.
        """
        f_comp = filecmp.cmpfiles(self.left, self.right, self.common_files, shallow=False)
        self.same_files, self.diff_files, self.funny_files = f_comp


def check_folder_equality(folder1, folder2, structure_only=False):
    dcmp = dircmp(folder1, folder2)
    return _is_same_folder(dcmp, structure_only)


def _is_same_folder(dcmp: dircmp, structure_only):
    if dcmp.left_only or dcmp.right_only:
        return False
    if not structure_only and dcmp.diff_files:
        return False
    for sub_dcmp in dcmp.subdirs.values():
        if not _is_same_folder(sub_dcmp, structure_only):
            return False
    return True
