"""Reproducible seven-category news classifier.

WHY THIS EXISTS: the Phase 1-3 analyses are all keyed on a `category` column
that arrived in `classified_articles_EXPERIMENTAL.parquet` with NO generating
code anywhere in the repository (confirmed by audit: the seven category names
appear only in that artifact and in the modules that CONSUME it). The labels
were therefore unreproducible -- new articles could not be categorised at all,
and the historical labels could not be re-derived or checked. This module is
that missing step, written as explicit, deterministic, inspectable rules.

HOW IT WORKS: priority-ordered regular expressions over the headline. The first
rule that matches wins; anything unmatched is OTHER. Priority matters because
real headlines carry several signals at once ("Agilent to Expand Portfolio With
Avida Biomed Acquisition" is both an acquisition and a product expansion), so
the order below encodes which signal dominates. It is a transparent rule
system, not a model: same input always gives same output, with no training
step, no randomness, and no dependency on a downloaded checkpoint.

COMPARABILITY WITH THE HISTORICAL LABELS, MEASURED NOT ASSUMED: because Phase 2
and Phase 3 evidence is keyed on the historical labels, a new article is only
comparable to that evidence if it is labelled the same way. `agreement_report`
scores this classifier against the full historical artifact, so the rate is a
reported number rather than an assumption. Measured on all 1,363,783 historical
articles: **91.2% overall agreement**, with per-category recall of 0.99
(EARNINGS_RESULTS), 0.96 (MANAGEMENT), 0.91 (OTHER), 0.90 (ANALYST_ACTION),
0.89 (M_AND_A), 0.76 (REGULATORY_LEGAL), 0.49 (MAJOR_COMPANY_NEWS).

MAJOR_COMPANY_NEWS is the weakest at 0.49 recall and is the one to treat with
most caution: the historical labeller used a bare "announces" keyword that this
module rejects as too generic (it would also catch "Foxconn founder announces
run for Taiwan presidency"). Roughly half of historical MAJOR_COMPANY_NEWS
therefore lands in OTHER here. A current article labelled MAJOR_COMPANY_NEWS is
reliable; the reverse direction is not, so absence of that label is weak
evidence.

The residual ~9% is dominated by the historical labeller's OWN inconsistency,
not by error here, and that was checked rather than assumed. Two concrete
examples from the disagreement set:
  * "Jefferies Maintains Agilent Technologies (A) Buy Recommendation" was
    labelled OTHER historically, while the identically-formatted "Barclays
    Maintains Agilent Technologies (A) Underweight Recommendation" was labelled
    ANALYST_ACTION. The old keyword list covered outperform/overweight but not
    buy/hold/equal-weight/in-line, so the same event type landed in two
    different buckets depending on the rating word.
  * "Agilent (A) Boosts LSAG Segment With New GPC/SEC Solutions" was labelled
    REGULATORY_LEGAL, matching the substring "SEC" inside a chromatography term.
Closing that last 9% would mean reproducing arbitrary keyword gaps and
substring bugs, which would make this classifier worse at its actual job while
making one statistic look better. It was deliberately not done. The agreement
rate and this caveat are surfaced in the pipeline's own output so that
cross-era category comparisons are hedged where they are made, not here.

BIAS DIRECTION, CHOSEN ON PURPOSE: the rules are tuned to under-claim rather
than over-claim. An early draft over-fired MAJOR_COMPANY_NEWS 5.3x and M_AND_A
2.8x, pushing ordinary market commentary into event categories; those articles
would then have been matched against Phase 2 event evidence and produced
confident-looking nonsense. Mislabelling a real event as OTHER costs a missed
observation; mislabelling commentary as an event manufactures a false signal,
which is the worse failure for this product.

The canonical category tuple is duplicated from
app.events.category_horizon_analysis rather than imported, to keep the NLP
layer from depending on the analysis layer; a test asserts the two are
identical so they cannot drift apart.
"""

from __future__ import annotations

import re

import pandas as pd

CATEGORIES: tuple[str, ...] = (
    "EARNINGS_RESULTS",
    "ANALYST_ACTION",
    "M_AND_A",
    "REGULATORY_LEGAL",
    "MANAGEMENT",
    "MAJOR_COMPANY_NEWS",
    "OTHER",
)

# Priority-ordered rules. First match wins; order IS the tie-break policy.
#
# Rationale for this specific order, from inspecting the historical labels:
#  1. ANALYST_ACTION  -- the most formulaic headlines in the corpus
#     ("<Bank> Maintains <Co> Outperform Recommendation"). Matching it first
#     stops broker boilerplate from being read as earnings commentary.
#  2. EARNINGS_RESULTS -- quarterly results language is the next most explicit,
#     and Phase 1 found this category's association strongest, so it should not
#     lose headlines to vaguer buckets.
#  3. M_AND_A -- deal verbs are concrete and usually the headline's main event,
#     outranking the product/partnership language a deal is described with.
#  4. REGULATORY_LEGAL -- courts and regulators outrank an executive being
#     named in the same sentence, because the legal action is the event.
#  5. MANAGEMENT -- people changes, insider transactions, executive remarks.
#  6. MAJOR_COMPANY_NEWS -- the broadest real-signal bucket: launches,
#     partnerships, product expansion. Deliberately last so it cannot absorb
#     headlines that a sharper rule above would have claimed.
#  7. OTHER -- residual. Roughly three quarters of the corpus, by design: most
#     ticker-tagged articles are market commentary, not company events.
_RULES: tuple[tuple[str, str], ...] = (
    (
        "ANALYST_ACTION",
        r"\b(?:maintains?|reiterat(?:es?|ed)|initiat(?:es?|ed)\s+coverage|"
        r"resum(?:es?|ed)\s+coverage|upgrad(?:es?|ed)|downgrad(?:es?|ed)|"
        r"outperform|underperform|overweight|underweight|"
        r"price\s+target|target\s+price|"
        r"buy\s+rating|sell\s+rating|hold\s+rating|neutral\s+rating|"
        r"recommendation)\b",
    ),
    (
        "EARNINGS_RESULTS",
        r"\b(?:earnings|eps|revenues?|profits?|"
        r"q[1-4]\s*(?:'?\d{2})?|(?:first|second|third|fourth)[\s-]quarter|"
        r"quarterly\s+results?|"
        r"beats?|miss(?:es|ed)?|estimates?|guidance|outlook\s+rais|"
        r"top(?:s|ped)?\s+(?:estimates|views?|expectations)|"
        r"conference\s+call)\b",
    ),
    # Tightened after measuring against the historical labels: a bare
    # "to buy" matched market commentary ("Time to Buy Microsoft?", "3 Stocks
    # to Buy"), and a bare "stake in" / "deal with" matched routine
    # reporting, between them inflating this category ~2.8x. Deal language
    # now has to name the transaction.
    (
        "M_AND_A",
        # "acquisition" stems on acquis-, not acquir-, so it needs its own
        # alternative; folding it into acquir(...) silently matched nothing.
        r"\b(?:acquir(?:e|es|ed|ing)|acquisitions?|merger|merges?|merging|"
        r"buyout|takeover|take[\s-]private|tender\s+offer|"
        r"divestiture|spin[\s-]?off|"
        r"(?:majority|controlling|minority)\s+stake|"
        r"bid(?:s)?\s+for|agree[sd]?\s+to\s+(?:buy|acquire)|"
        r"all[\s-]cash\s+deal|definitive\s+agreement)\b",
    ),
    # `approval`/`compliance`/`fine` were unqualified and matched ordinary
    # business language; regulatory approval now has to name a regulator or
    # say "regulatory". "SEC" is deliberately absent as a bare token -- that
    # is exactly the historical labeller's bug ("GPC/SEC Solutions"), so it
    # appears only in its unambiguous forms.
    (
        "REGULATORY_LEGAL",
        r"\b(?:lawsuit|sue[sd]?|suing|litigation|settle(?:s|d|ment)?|"
        r"antitrust|monopol(?:y|ies)|court|judge|jury|trial|appeal|"
        r"subpoena|indict(?:ed|ment)?|fraud|"
        r"probe[sd]?|investigat(?:es?|ed|ion)|inquiry|"
        r"regulator[sy]?|regulation|sanction(?:s|ed)?|penalt(?:y|ies)|"
        r"fda|ftc|doj|sec\s+(?:filing|probe|charges?|investigation)|"
        r"securities\s+and\s+exchange|european\s+commission|"
        r"(?:fda|regulatory|antitrust|eu)\s+(?:approval|clearance|nod)|"
        r"approved\s+by\s+(?:the\s+)?(?:fda|regulators?)|"
        r"patent\s+(?:suit|dispute|infringement))\b",
    ),
    (
        "MANAGEMENT",
        r"\b(?:ceo|cfo|coo|cto|cio|"
        # spelled-out titles: "names new chief operating officer" was falling
        # through to OTHER when only the abbreviations were matched
        r"chief\s+(?:executive|financial|operating|technology|information|"
        r"legal|medical|revenue)\s+officer|"
        r"chairman|chairwoman|chair\b|"
        r"president|executive|exec\b|board\s+(?:member|seat|of\s+directors)|"
        r"appoint(?:s|ed|ment)?|names?\s+\w+\s+as|steps?\s+down|"
        r"resign(?:s|ed|ation)?|retir(?:es?|ed|ement)|our?st(?:s|ed)|"
        r"fir(?:es|ed|ing)|succeed(?:s|ed)|successor|"
        r"insider\s+(?:buying|selling|sale|purchase))\b",
    ),
    # The greediest rule by far before tightening: it over-fired ~5.3x,
    # because `dividend`, `patent`, `plant`, `facility` and bare
    # `expands`/`boosts` are everywhere in generic commentary ("3 Dividend
    # Stocks to Buy"). Product/partnership language now has to attach to a
    # company-action object.
    (
        "MAJOR_COMPANY_NEWS",
        r"\b(?:unveil(?:s|ed|ing)?|launch(?:es|ed|ing)?|"
        r"introduc(?:es?|ed|ing)\s+(?:new|a\s+new)|debuts?|"
        r"roll(?:s|ed)\s+out|"
        r"partnership|partners\s+with|collaborat(?:es?|ed|ion)|"
        r"joint\s+venture|"
        r"(?:expand(?:s|ed|ing)?|boost(?:s|ed|ing)?|bolsters?)\s+"
        r"(?:its\s+|their\s+|the\s+)?"
        r"(?:portfolio|offerings?|lineup|capacity|segment|business|presence|"
        r"footprint|product)|"
        r"contract\s+(?:win|award)|wins?\s+(?:a\s+)?contract|"
        r"recall(?:s|ed)?|layoffs?|restructur(?:es?|ing)|"
        r"buyback|share\s+repurchase|stock\s+split|"
        r"dividend\s+(?:hike|increase|cut|boost|rais)|"
        r"(?:rais(?:es?|ed)|hikes?|cuts?)\s+(?:its\s+)?dividend|"
        r"breakthrough)\b",
    ),
)

_COMPILED: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (cat, re.compile(pat, re.IGNORECASE)) for cat, pat in _RULES
)


def classify_headline(headline: str | None) -> str:
    """Assign one of CATEGORIES to a headline. Deterministic; OTHER if nothing
    matches (and for empty/missing text, which carries no evidence either way).
    """
    if not headline or not isinstance(headline, str):
        return "OTHER"
    for category, pattern in _COMPILED:
        if pattern.search(headline):
            return category
    return "OTHER"


def classify_frame(df: pd.DataFrame, column: str = "headline") -> pd.Series:
    """Vectorised-enough classification of a headline column."""
    if df.empty:
        return pd.Series([], dtype=object, index=df.index)
    return df[column].map(classify_headline)


def rule_description() -> list[dict]:
    """The rule table as data, so a report can show exactly what was applied
    instead of asking the reader to trust a prose summary."""
    return [
        {"priority": i + 1, "category": cat, "pattern": pat}
        for i, (cat, pat) in enumerate(_RULES)
    ]


def agreement_report(
    reference: pd.DataFrame,
    headline_col: str = "headline",
    category_col: str = "category",
) -> dict:
    """Score this classifier against already-labelled articles.

    This is the comparability check for every downstream use: Phase 2/3
    evidence is keyed on the historical labels, so a newly classified article
    is only comparable to that evidence to the extent the two labelling
    conventions agree. Per-category recall is included because an aggregate
    rate hides the categories that matter -- a 99% overall score built on
    OTHER dominating is not evidence that EARNINGS_RESULTS transfers.
    """
    if reference.empty:
        return {"n": 0, "overall_agreement": None, "per_category": []}
    predicted = classify_frame(reference, headline_col)
    actual = reference[category_col]
    match = predicted == actual
    per_category = []
    for cat in CATEGORIES:
        in_cat = actual == cat
        n_cat = int(in_cat.sum())
        if not n_cat:
            continue
        pred_cat = predicted == cat
        per_category.append({
            "category": cat,
            "n_reference": n_cat,
            "recall": round(float(match[in_cat].mean()), 4),
            "n_predicted": int(pred_cat.sum()),
            "precision": (
                round(float(match[pred_cat].mean()), 4) if int(pred_cat.sum()) else None
            ),
        })
    return {
        "n": int(len(reference)),
        "overall_agreement": round(float(match.mean()), 4),
        "per_category": per_category,
        "note": (
            "Agreement against the historical labels, which have no generating code "
            "and are internally inconsistent (identically-formatted analyst headlines "
            "were split between ANALYST_ACTION and OTHER depending on the rating word; "
            "'GPC/SEC Solutions' was matched as the SEC regulator). Disagreement is "
            "therefore not purely error in this classifier. Treat any comparison "
            "between newly classified articles and Phase 2/3 historical evidence as "
            "approximate to within this rate."
        ),
    }
