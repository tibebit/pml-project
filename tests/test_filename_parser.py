import pytest

from src.dataset.workbook import parse_filename


def test_parse_valid_sample_filename():
    parsed = parse_filename("S1-w035-Uebu-A-a1-1")

    assert parsed["raw_voice_code"] == "S1"
    assert parsed["raw_person_code"] == "w035"
    assert parsed["raw_singer_code"] == "S1-w035"
    assert parsed["singer_id"] == "soprano_dramatic_w035"
    assert parsed["voice_type"] == "soprano"
    assert parsed["class_label"] == "dramatic"
    assert parsed["piece_or_context_code"] == "Uebu"
    assert parsed["vowel_code"] == "A"
    assert parsed["pitch_code"] == "a1"
    assert parsed["take"] == 1


def test_parse_valid_sample_filename_with_take_suffix():
    parsed = parse_filename("S1-w005-Isol-A-g2-1-is1")

    assert parsed["raw_voice_code"] == "S1"
    assert parsed["raw_person_code"] == "w005"
    assert parsed["pitch_code"] == "g2"
    assert parsed["take"] == 1


@pytest.mark.parametrize(
    "filename",
    [
        "",
        "   ",
        "Durchschnitt aus Matlab",
        "Durchschnittaus Matlab",
        "SD",
        "SD PHE in Hz",
        "Test-Durchschnitt",
        "Test Mittelwert",
    ],
)
def test_parse_rejects_empty_and_statistic_rows(filename):
    with pytest.raises(ValueError):
        parse_filename(filename)


def test_parse_rejects_malformed_filename():
    with pytest.raises(ValueError):
        parse_filename("S1-w035-Uebu-A-a1")
