import json
import os
import pickle
import tempfile


def atomic_write_bytes(path, data):
    """Write bytes via same-directory temp file, then atomically replace target."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def atomic_pickle_dump(obj, path):
    atomic_write_bytes(path, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def atomic_json_dump(obj, path, ensure_ascii=False, indent=2):
    text = json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent)
    atomic_write_bytes(path, text.encode("utf-8"))
