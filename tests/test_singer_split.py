import pandas as pd

from src.splits.singer_split import (
    check_no_singer_leakage,
    make_final_holdout_split,
    make_stratum_column,
)


def test_make_final_holdout_split_has_no_singer_overlap():
    df = _example_singer_df()

    train_df, test_df = make_final_holdout_split(df, test_size=0.25, seed=7)

    check_no_singer_leakage(train_df, test_df)
    assert len(train_df) + len(test_df) == len(df)
    assert not test_df.empty


def test_make_final_holdout_split_same_seed_is_reproducible():
    df = _example_singer_df()

    first = make_final_holdout_split(df, test_size=0.25, seed=7)
    second = make_final_holdout_split(df, test_size=0.25, seed=7)

    assert first[0]["singer_id"].tolist() == second[0]["singer_id"].tolist()
    assert first[1]["singer_id"].tolist() == second[1]["singer_id"].tolist()


def test_make_stratum_column_combines_voice_type_and_class():
    df = pd.DataFrame({"voice_type": ["soprano"], "class_label": ["dramatic"]})

    assert make_stratum_column(df).tolist() == ["soprano_dramatic"]


def _example_singer_df():
    records = []
    for voice_type in ["soprano", "tenor"]:
        for class_label, class_id in [("lyric", 0), ("dramatic", 1)]:
            for index in range(4):
                records.append(
                    {
                        "singer_id": f"{voice_type}_{class_label}_{index}",
                        "voice_type": voice_type,
                        "class_label": class_label,
                        "class_id": class_id,
                    }
                )
    return pd.DataFrame.from_records(records)
