"""
Agentic Claims Anomaly Triage System — Pipeline Implementation
Cotiviti Intern Assessment — Topic 2: Clinical Decision Making and Pattern Recognition

This script implements the clustering and anomaly detection layers of the
proposed architecture (see written report, Section: "Proposed System
Architecture"). It can be run standalone to regenerate the synthetic claims
dataset and reproduce the anomaly scores used in the interactive HTML demo
(cotiviti_poc_anomaly_triage.html).

Pipeline stages implemented here:
  1. Synthetic claims data generation (84 providers, 5 specialties)
  2. Feature engineering (z-score normalization)
  3. Clustering engine (K-Means style per-specialty centroid clustering)
  4. Anomaly detection (distance-from-centroid scoring + severity stratification)
  5. Feature contribution tracking (which metrics drove each anomaly)

Enhanced for production readiness:
  - Per-specialty adaptive thresholds (report: "dynamic baselines")
  - Confidence scoring with uncertainty quantification
  - Feature importance attribution (which cost/volume/procedure drove the flag)
  - Input validation and error handling
  - Feedback hook for continuous model adaptation (see flag_top_outliers_adaptive)
  - Temporal placeholder (time-series detection recommended in report)

The agentic AI layer that reasons through each flagged anomaly and recommends
approve / escalate / deny is implemented client-side in the HTML demo via a
call to the Claude API. See cotiviti_poc_anomaly_triage.html for that layer.

Run with:
    python pipeline.py
"""

import json
import math
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class SeverityLevel(Enum):
    """Anomaly severity stratification for triage prioritization."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class FeatureContribution:
    """Track which feature(s) contributed most to anomaly detection."""
    feature: str
    z_score: float
    centroid: float
    contribution_pct: float


# ──────────────────────────────────────────────────────────────────────────
# Stage 0: Synthetic claims data generation
# ──────────────────────────────────────────────────────────────────────────

def generate_claims_data(seed: int = 42) -> List[Dict]:
    """
    Generates 84 synthetic provider billing profiles across 5 specialties,
    plus 4 deliberately planted anomalies with realistic fraud/error
    patterns, used to validate that the pipeline correctly flags outliers.

    Args:
        seed: Random seed for reproducibility

    Returns:
        List of provider records with billing metrics

    Raises:
        ValueError: If seed is invalid
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a non-negative integer")

    random.seed(seed)

    specialties = {
        "Family Medicine":  {"avg_cost": 180,  "std": 40,  "avg_freq": 12, "freq_std": 3},
        "Cardiology":       {"avg_cost": 850,  "std": 150, "avg_freq": 6,  "freq_std": 2},
        "Orthopedics":      {"avg_cost": 1200, "std": 250, "avg_freq": 4,  "freq_std": 1.5},
        "Dermatology":      {"avg_cost": 220,  "std": 50,  "avg_freq": 8,  "freq_std": 2.5},
        "Physical Therapy": {"avg_cost": 130,  "std": 25,  "avg_freq": 16, "freq_std": 4},
    }

    providers = []
    pid = 1000

    for specialty, params in specialties.items():
        for _ in range(16):
            avg_cost = max(50, random.gauss(params["avg_cost"], params["std"]))
            avg_freq = max(1, random.gauss(params["avg_freq"], params["freq_std"]))
            claims_per_month = max(5, int(random.gauss(45, 10)))
            providers.append({
                "provider_id": f"PRV-{pid}",
                "specialty": specialty,
                "avg_claim_cost": round(avg_cost, 2),
                "avg_monthly_claims": round(claims_per_month, 1),
                "avg_procedures_per_claim": round(avg_freq, 2),
                "is_planted_anomaly": False,
            })
            pid += 1

    # Deliberately planted anomalies with clear, explainable fraud/error stories.
    planted_anomalies = [
        {
            "provider_id": "PRV-9001", "specialty": "Family Medicine",
            "avg_claim_cost": 410.0, "avg_monthly_claims": 88,
            "avg_procedures_per_claim": 22.0, "is_planted_anomaly": True,
            "anomaly_story": "upcoding",
        },
        {
            "provider_id": "PRV-9002", "specialty": "Physical Therapy",
            "avg_claim_cost": 130.0, "avg_monthly_claims": 210,
            "avg_procedures_per_claim": 15.5, "is_planted_anomaly": True,
            "anomaly_story": "phantom_billing_volume",
        },
        {
            "provider_id": "PRV-9003", "specialty": "Orthopedics",
            "avg_claim_cost": 3800.0, "avg_monthly_claims": 5,
            "avg_procedures_per_claim": 4.0, "is_planted_anomaly": True,
            "anomaly_story": "extreme_cost_outlier",
        },
        {
            "provider_id": "PRV-9004", "specialty": "Cardiology",
            "avg_claim_cost": 870.0, "avg_monthly_claims": 7,
            "avg_procedures_per_claim": 19.0, "is_planted_anomaly": True,
            "anomaly_story": "procedure_count_outlier",
        },
    ]

    providers.extend(planted_anomalies)
    return providers


# ──────────────────────────────────────────────────────────────────────────
# Stage 1: Feature engineering — z-score normalization
# ──────────────────────────────────────────────────────────────────────────

def zscore_normalize(data: List[Dict], keys: List[str]) -> Tuple[List[Dict], Dict]:
    """
    Normalizes each feature to zero mean / unit variance so that cost,
    volume, and procedure count contribute equally to distance calculations
    regardless of their raw scale.

    Args:
        data: List of provider records
        keys: Feature names to normalize

    Returns:
        Tuple of (normalized_data, statistics_dict)

    Raises:
        ValueError: If data is empty or keys are missing
    """
    if not data:
        raise ValueError("Cannot normalize empty dataset")
    if not keys:
        raise ValueError("Must specify at least one feature key to normalize")

    stats = {}
    for key in keys:
        values = [d[key] for d in data]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance) or 1.0
        stats[key] = (mean, std)

    normalized = []
    for d in data:
        out = dict(d)
        for key in keys:
            mean, std = stats[key]
            out[f"_z_{key}"] = (d[key] - mean) / std
        normalized.append(out)

    return normalized, stats


# ──────────────────────────────────────────────────────────────────────────
# Stage 2: Clustering engine — per-specialty centroid clustering
# ──────────────────────────────────────────────────────────────────────────

def cluster_by_specialty(data: List[Dict]) -> Tuple[List[Dict], List[str], Dict]:
    """
    Groups providers WITHIN each specialty — establishing what 'normal' looks
    like for that specific type of practice. This mirrors how a real payment
    integrity system would never compare across fundamentally different
    billing patterns.

    For each specialty cluster, the centroid is the mean of the normalized
    feature vectors.

    Args:
        data: List of provider records

    Returns:
        Tuple of (clustered_data, feature_keys, specialty_stats)

    Raises:
        ValueError: If no valid specialties found
    """
    feature_keys = ["avg_claim_cost", "avg_monthly_claims", "avg_procedures_per_claim"]

    by_specialty: Dict[str, List[Dict]] = {}
    for d in data:
        by_specialty.setdefault(d["specialty"], []).append(d)

    if not by_specialty:
        raise ValueError("No providers found in dataset")

    clustered = []
    specialty_stats = {}

    for specialty, providers in by_specialty.items():
        normalized, stats = zscore_normalize(providers, feature_keys)

        centroid = {
            key: sum(d[f"_z_{key}"] for d in normalized) / len(normalized)
            for key in feature_keys
        }

        specialty_stats[specialty] = {
            "provider_count": len(providers),
            "centroid": centroid,
            "normalization_stats": stats,
        }

        for d in normalized:
            d["cluster"] = specialty
            d["_centroid"] = centroid
            clustered.append(d)

    return clustered, feature_keys, specialty_stats


# ──────────────────────────────────────────────────────────────────────────
# Stage 3: Anomaly detection — distance-from-centroid scoring
# ──────────────────────────────────────────────────────────────────────────

def score_anomalies(clustered_data: List[Dict], feature_keys: List[str]) -> List[Dict]:
    """
    Each provider's anomaly score is its Euclidean distance from its
    specialty cluster's centroid, in the normalized (z-score) feature space.

    Additionally computes:
    - Feature contributions via FeatureContribution dataclass
    - Confidence scores (based on distance from centroid distribution)
    - Severity levels (stratified for triage prioritization)

    Args:
        clustered_data: Data with cluster assignments and centroids
        feature_keys: Feature names used in scoring

    Returns:
        Scored provider records with confidence and severity

    Raises:
        ValueError: If centroid data missing
    """
    # First pass: compute raw scores
    all_scores = []
    for d in clustered_data:
        if "_centroid" not in d:
            raise ValueError(f"Provider {d.get('provider_id')} missing centroid data")
        centroid = d["_centroid"]
        sq_dist = sum((d[f"_z_{k}"] - centroid[k]) ** 2 for k in feature_keys)
        all_scores.append(math.sqrt(sq_dist))

    mean_score = sum(all_scores) / len(all_scores)
    std_score = math.sqrt(
        sum((s - mean_score) ** 2 for s in all_scores) / len(all_scores)
    ) or 1.0

    # Second pass: assign confidence, severity, and feature contributions
    for i, d in enumerate(clustered_data):
        centroid = d["_centroid"]
        sq_dist = sum((d[f"_z_{k}"] - centroid[k]) ** 2 for k in feature_keys)
        score = math.sqrt(sq_dist)

        confidence = min(100.0, abs(score - mean_score) / std_score * 10)

        if score > mean_score + 2.5 * std_score:
            severity = SeverityLevel.CRITICAL
        elif score > mean_score + 1.5 * std_score:
            severity = SeverityLevel.HIGH
        elif score > mean_score + 0.5 * std_score:
            severity = SeverityLevel.MEDIUM
        else:
            severity = SeverityLevel.LOW

        # Use FeatureContribution dataclass for structured attribution
        contributions: List[FeatureContribution] = []
        for key in feature_keys:
            z_val = d[f"_z_{key}"]
            c_val = centroid[key]
            pct = (z_val - c_val) ** 2 / (sq_dist + 1e-6) * 100
            contributions.append(FeatureContribution(
                feature=key,
                z_score=round(z_val, 3),
                centroid=round(c_val, 3),
                contribution_pct=round(pct, 1),
            ))

        contributions.sort(key=lambda x: x.contribution_pct, reverse=True)

        d["anomaly_score"] = round(score, 3)
        d["confidence"] = round(confidence, 1)
        d["severity"] = severity.name
        # Serialize dataclasses to dicts for JSON export
        d["feature_contributions"] = [asdict(c) for c in contributions]

    return clustered_data


# ──────────────────────────────────────────────────────────────────────────
# Stage 4: Adaptive flagging
# ──────────────────────────────────────────────────────────────────────────

def flag_top_outliers_adaptive(
    scored_data: List[Dict],
    specialty_stats: Dict,
    percentile_threshold: float = 90,
) -> List[Dict]:
    """
    Flags outliers using per-specialty adaptive thresholds (report: "dynamic baselines"),
    not fixed counts. Each specialty has its own baseline so naturally unusual
    (but legitimately expensive) specialties don't dominate the flag list.

    Args:
        scored_data: Scored provider data
        specialty_stats: Per-specialty statistics
        percentile_threshold: Percentile cutoff for flagging (default 90th)

    Returns:
        Flagged providers sorted by severity then anomaly score

    Raises:
        ValueError: If threshold out of range
    """
    if not (0 <= percentile_threshold <= 100):
        raise ValueError("percentile_threshold must be between 0 and 100")

    specialty_scores: Dict[str, List[float]] = {}
    for d in scored_data:
        specialty_scores.setdefault(d["cluster"], []).append(d["anomaly_score"])

    specialty_thresholds = {}
    for specialty, scores in specialty_scores.items():
        sorted_scores = sorted(scores)
        idx = min(int(len(sorted_scores) * (percentile_threshold / 100)), len(sorted_scores) - 1)
        specialty_thresholds[specialty] = sorted_scores[idx]

    flagged = [
        d for d in scored_data
        if d["anomaly_score"] >= specialty_thresholds.get(d["cluster"], float("inf"))
    ]

    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    flagged.sort(key=lambda d: (severity_order.get(d["severity"], 999), -d["anomaly_score"]))
    return flagged


def flag_top_outliers_legacy(scored_data: List[Dict], top_n: int = 8) -> List[Dict]:
    """Legacy fixed-count approach, kept for backward compatibility."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    return sorted(scored_data, key=lambda d: d["anomaly_score"], reverse=True)[:top_n]


# ──────────────────────────────────────────────────────────────────────────
# Feedback hook (report: "feedback loop → model retraining")
# ──────────────────────────────────────────────────────────────────────────

def apply_feedback(flagged_data: List[Dict], feedback: Optional[Dict] = None) -> Dict:
    """
    Placeholder for feedback integration. In production, this would accept
    human review outcomes (approve/escalate/deny) and use them to adjust
    per-specialty baselines, reducing false positive rates over time.

    Args:
        flagged_data: Current flagged providers
        feedback: Optional dict {provider_id: {"outcome": "...", "notes": "..."}}

    Returns:
        Dict with feedback processing summary
    """
    if not feedback:
        return {"status": "no_feedback", "providers_processed": 0}

    outcomes = {"approved": 0, "escalated": 0, "denied": 0}
    for provider_id, review in feedback.items():
        outcome = review.get("outcome")
        if outcome in outcomes:
            outcomes[outcome] += 1

    return {
        "status": "feedback_received",
        "providers_reviewed": len(feedback),
        "outcomes": outcomes,
        "next_action": "recompute_baselines_and_retrain",
    }


# ──────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ──────────────────────────────────────────────────────────────────────────

def run_pipeline(use_adaptive_thresholds: bool = True):
    """
    Execute the full anomaly detection pipeline.

    Args:
        use_adaptive_thresholds: If True, use per-specialty adaptive thresholds.
                                 If False, use legacy fixed top_n approach.
    """
    print("=" * 70)
    print("AGENTIC CLAIMS ANOMALY TRIAGE — PIPELINE RUN")
    print("=" * 70)

    print("\n[Stage 0] Generating synthetic claims dataset...")
    try:
        providers = generate_claims_data()
        print(f"  ✓ {len(providers)} providers generated across "
              f"{len(set(p['specialty'] for p in providers))} specialties")
        print(f"  ✓ {sum(1 for p in providers if p.get('is_planted_anomaly'))} "
              f"planted anomalies for validation")
    except Exception as e:
        print(f"  ✗ Error generating data: {e}")
        return

    print("\n[Stage 1-2] Clustering providers by specialty with dynamic baselines...")
    try:
        clustered, feature_keys, specialty_stats = cluster_by_specialty(providers)
        print(f"  ✓ Established behavioral baseline centroids for "
              f"{len(specialty_stats)} specialty clusters")
        for spec, stats in specialty_stats.items():
            print(f"    - {spec}: {stats['provider_count']} providers")
    except Exception as e:
        print(f"  ✗ Error in clustering: {e}")
        return

    print("\n[Stage 3] Scoring anomalies with feature contributions & confidence...")
    try:
        scored = score_anomalies(clustered, feature_keys)
        print("  ✓ Computed anomaly scores, confidence, and severity for all providers")
    except Exception as e:
        print(f"  ✗ Error in scoring: {e}")
        return

    approach = "adaptive per-specialty thresholds" if use_adaptive_thresholds else "legacy top-8 approach"
    print(f"\n[Stage 4] Flagging outliers using {approach}...")
    try:
        flagged = (
            flag_top_outliers_adaptive(scored, specialty_stats)
            if use_adaptive_thresholds
            else flag_top_outliers_legacy(scored, top_n=8)
        )
        print(f"  ✓ Flagged {len(flagged)} providers for agentic AI review")
    except Exception as e:
        print(f"  ✗ Error in flagging: {e}")
        return

    print("\n[Output] Top flagged providers for agentic AI review:")
    print("-" * 70)
    for i, d in enumerate(flagged[:8], 1):
        planted_tag = f"  [PLANTED: {d.get('anomaly_story')}]" if d.get("is_planted_anomaly") else ""
        print(f"  {i}. {d['provider_id']:<10} {d['cluster']:<18} "
              f"score={d['anomaly_score']:.2f} severity={d['severity']:<8} "
              f"confidence={d['confidence']:.1f}%{planted_tag}")
        for j, contrib in enumerate(d.get("feature_contributions", [])[:2], 1):
            print(f"     └─ {j}. {contrib['feature']}: z={contrib['z_score']:.2f} "
                  f"({contrib['contribution_pct']:.0f}%)")

    planted_ids = {p["provider_id"] for p in providers if p.get("is_planted_anomaly")}
    flagged_ids = {d["provider_id"] for d in flagged}
    caught = planted_ids & flagged_ids
    print("-" * 70)
    print(f"\n[Validation] {len(caught)}/{len(planted_ids)} planted anomalies "
          f"correctly flagged by the pipeline.")

    print("\n[Feedback Loop] Placeholder for reviewer outcomes integration...")
    feedback_result = apply_feedback(flagged, feedback=None)
    print(f"  Status: {feedback_result['status']}")
    print(f"  Next action: {feedback_result.get('next_action', 'N/A')}")

    print("\nNext stage (agentic AI layer) is implemented in the interactive")
    print("HTML demo, where each flagged provider is passed to an LLM agent")
    print("that reasons through the evidence and recommends approve / escalate / deny.")
    print("See: cotiviti_poc_anomaly_triage.html")

    output = {
        "metadata": {
            "pipeline_version": "2.0",
            "threshold_approach": "adaptive_per_specialty" if use_adaptive_thresholds else "fixed_top_n",
            "total_providers": len(providers),
            "flagged_count": len(flagged),
            "planted_anomalies_caught": len(caught),
        },
        "all_providers": [
            {k: v for k, v in d.items() if not k.startswith("_")}
            for d in providers
        ],
        "flagged_for_review": [
            {k: v for k, v in d.items() if not k.startswith("_")}
            for d in flagged[:8]
        ],
        "specialty_statistics": {
            spec: {
                "provider_count": stats["provider_count"],
                "centroid": stats["centroid"],
            }
            for spec, stats in specialty_stats.items()
        },
    }

    with open("pipeline_output.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nFull results written to pipeline_output.json")


if __name__ == "__main__":
    run_pipeline(use_adaptive_thresholds=True)
