import arviz as az
import numpy as np

from src.models.diagnostics import plot_prior_predictive_check, summarize_mcmc_diagnostics_summary


def test_mcmc_diagnostics_summary_is_one_compact_row():
    idata = _fake_idata()

    summary = summarize_mcmc_diagnostics_summary(
        idata,
        draws=5,
        tune=5,
        chains=2,
        target_accept=0.9,
    )

    assert len(summary) == 1
    assert list(summary.columns) == [
        "chains",
        "draws",
        "tune",
        "target_accept",
        "n_parameters_reported",
        "n_divergences",
        "max_rhat",
        "min_bulk_ess",
        "min_tail_ess",
        "min_bfmi",
        "sampler_warnings",
    ]
    assert summary.loc[0, "n_divergences"] == 0


def test_prior_predictive_check_plot_is_written(tmp_path):
    idata = _fake_idata()
    output_path = tmp_path / "prior_predictive_check.png"

    result = plot_prior_predictive_check(idata, output_path, ["PHE", "SC"])

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def _fake_idata():
    rng = np.random.default_rng(1)
    coords = {
        "voice_type": ["soprano"],
        "class_label": ["lyric", "dramatic"],
        "feature": ["PHE", "SC"],
        "singer_id": ["s1", "s2", "s3"],
    }
    dims = {
        "pi": ["voice_type"],
        "mu": ["voice_type", "class_label", "feature"],
        "tau": ["voice_type", "class_label", "feature"],
        "sigma": ["voice_type", "feature"],
        "z": ["singer_id", "feature"],
    }
    posterior = {
        "pi": rng.uniform(0.2, 0.8, size=(2, 5, 1)),
        "mu": rng.normal(size=(2, 5, 1, 2, 2)),
        "tau": rng.uniform(0.2, 1.0, size=(2, 5, 1, 2, 2)),
        "sigma": rng.uniform(0.2, 1.0, size=(2, 5, 1, 2)),
        "z": rng.normal(size=(2, 5, 3, 2)),
    }
    sample_stats = {
        "diverging": np.zeros((2, 5), dtype=bool),
    }
    prior = {
        "mu": rng.normal(size=(1, 7, 1, 2, 2)),
        "z": rng.normal(size=(1, 7, 3, 2)),
        "pi": rng.uniform(0.2, 0.8, size=(1, 7, 1)),
        "tau": rng.uniform(0.2, 1.0, size=(1, 7, 1, 2, 2)),
        "sigma": rng.uniform(0.2, 1.0, size=(1, 7, 1, 2)),
    }
    prior_predictive = {
        "c_obs": rng.integers(0, 2, size=(1, 7, 3)),
        "x": rng.normal(size=(1, 7, 9, 2)),
    }
    observed_data = {
        "c_obs": np.array([0, 1, 0]),
        "x": rng.normal(size=(9, 2)),
    }
    return az.from_dict(
        posterior=posterior,
        sample_stats=sample_stats,
        prior=prior,
        prior_predictive=prior_predictive,
        observed_data=observed_data,
        coords=coords,
        dims={
            **dims,
            "c_obs": ["singer_id"],
            "x": ["vocalization", "feature"],
        },
    )
