from pathlib import Path


def test_non_ascii_espeak_data_is_copied_to_ascii_cache(tmp_path):
    from core.guidance.tts import espeak_data_path
    source = tmp_path/"данные"/"espeak-ng-data"
    source.mkdir(parents=True)
    (source/"phontab").write_bytes(b"phonemes")
    target = espeak_data_path(source, tmp_path/"cache")
    assert str(target).isascii()
    assert (target/"phontab").read_bytes() == b"phonemes"
    assert (source/"phontab").exists()


def test_ascii_espeak_install_needs_no_copy(tmp_path):
    from core.guidance.tts import espeak_data_path
    source = tmp_path/"espeak-ng-data"
    source.mkdir()
    (source/"phontab").write_bytes(b"phonemes")
    assert espeak_data_path(source, tmp_path/"cache") == source
    assert not (tmp_path/"cache").exists()
