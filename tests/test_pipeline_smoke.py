from __future__ import annotations

from app.features.build import build_features
from app.models.train import run_training
from app.nlp.finbert_scorer import LexiconScorer


def test_end_to_end_training_smoke(make_panel):
    panel = make_panel(tickers=("AAPL", "MSFT"), n=400, start="2022-01-03")
    feats = build_features(panel)

    result = run_training(
        feats,
        model_name="logistic",
        n_folds=3,
        embargo_days=7,
    )
    assert "roc_auc_mean" in result["summary"]
    assert result["baseline_summary"] is not None
    preds = result["predictions"]
    splits = set(preds["split"])
    val_splits = [s for s in splits if s.startswith("val_fold")]
    assert len(val_splits) == 3 and "test" in splits
    last_val_date = preds[preds["split"].isin(val_splits)]["date"].max()
    first_test_date = preds[preds["split"] == "test"]["date"].min()
    assert last_val_date < first_test_date
    assert len(preds[preds["split"] == "test"]) > 0


def test_lexicon_scorer_signs():
    scorer = LexiconScorer()
    assert scorer.score("Company beats earnings, stock rallies on record growth") > 0
    assert scorer.score("Shares plunge as losses mount and fraud probe widens") < 0
    assert scorer.score("The meeting is scheduled for Tuesday") == 0.0
