import pathlib
import zstandard
import urllib


def gather(domain: str, file: str, name: str):
    if not pathlib.Path(name).exists():
        path = __download(domain + "/" + file, file)
        __unpack(path, name)
        __remove(file)


def __download(url: str, name: str) -> str:
    path, _ = urllib.request.urlretrieve(url, name)
    return path


def __unpack(path: str, name: str):
    input_file = pathlib.Path(path)
    with open(input_file, 'rb') as compressed:
        decomp = zstandard.ZstdDecompressor()
        output_path = name
        with open(output_path, 'wb') as destination:
            decomp.copy_stream(compressed, destination)
            destination.close()
        compressed.close()


def __remove(path: str):
    pathlib.Path.unlink(pathlib.Path(path))
