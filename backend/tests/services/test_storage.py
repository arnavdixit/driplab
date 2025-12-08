from backend.app.services.storage import LocalStorage


def test_save_writes_file_and_returns_relative(tmp_path) -> None:
    storage = LocalStorage(root=tmp_path)
    data = b"sample-bytes"

    path = storage.save(data, filename="photo.png")

    assert path.startswith("uploads/")
    absolute = storage.root / path
    assert absolute.exists()
    assert absolute.read_bytes() == data


def test_save_generates_unique_names(tmp_path) -> None:
    storage = LocalStorage(root=tmp_path)

    first = storage.save(b"first", filename="photo.png")
    second = storage.save(b"second", filename="photo.png")

    assert first != second


def test_get_url_and_delete(tmp_path) -> None:
    storage = LocalStorage(root=tmp_path)
    path = storage.save(b"to-delete")

    url = storage.get_url(path)

    assert url.startswith("file://")
    assert path in url

    storage.delete(path)
    assert not (storage.root / path).exists()
