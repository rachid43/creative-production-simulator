"""Pydantic configuration schema for the Lean Creative Production Simulator.

Every tunable number used by the simulator lives in one of these models, so
there are no hidden constants in the engine.  Defaults are set in
``config/defaults.py``; values flagged "illustrative" are assumptions, not
measured performance.

Time convention
---------------
The simulation clock counts *working hours only* (Mon-Fri, ``hours_per_day``
per day).  Weekends do not exist on the clock, so every lead time and waiting
time is expressed in working hours.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

COMPLEXITIES = ("Simple", "Medium", "Complex")


class _Base(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")


# --------------------------------------------------------------------------- #
# A. Organization
# --------------------------------------------------------------------------- #
class OrgConfig(_Base):
    n_pods: int = Field(6, ge=1, le=12, description="Number of creative PODs")
    editors_per_pod: int = Field(3, ge=1, le=10)
    strategists_per_pod: int = Field(1, ge=1, le=4)
    editor_counts: Optional[List[int]] = Field(
        None, description="Optional explicit editors per POD (overrides editors_per_pod), e.g. after staffing shortage"
    )
    hours_per_day: float = Field(8.0, gt=0, le=16)
    days_per_week: int = Field(5, ge=1, le=7)
    productive_availability: float = Field(
        0.80, gt=0.05, le=1.0,
        description="Share of a shift that is productive (rest = meetings, asset search, coordination)",
    )
    planned_absenteeism: float = Field(
        0.0, ge=0.0, lt=1.0, description="Probability that a person is absent on any given working day"
    )
    overtime_hours_per_day: float = Field(
        0.0, ge=0.0, le=6.0,
        description="Max extra productive editing hours per editor-day, used only when the POD has a queue",
    )
    overtime_trigger_queue_per_editor: float = Field(
        1.0, ge=0.0, description="Overtime is activated for a POD-day when waiting editing work per editor >= this"
    )
    editor_speed_cv: float = Field(
        0.05, ge=0.0, le=0.5, description="Between-editor speed variability (lognormal CV, mean 1.0) - illustrative"
    )
    strategist_availability: Optional[float] = Field(
        None, gt=0.05, le=1.0,
        description="Productive availability of strategists, reviewers and the Lead Creative (None = same as editors)",
    )
    editor_speed_multiplier: float = Field(
        1.0, gt=0, le=3.0, description="Uniform editing-speed multiplier (training/tooling; also used for sensitivity tests)"
    )
    complex_capable_per_pod: Optional[int] = Field(
        None, ge=0, description="Editors per POD certified for Complex ads (None = all)"
    )

    @model_validator(mode="after")
    def _check_counts(self):
        if self.editor_counts is not None:
            if len(self.editor_counts) != self.n_pods:
                raise ValueError("editor_counts must have one entry per POD")
            if any(c < 0 for c in self.editor_counts):
                raise ValueError("editor_counts must be >= 0")
        return self

    def editors_in_pod(self, pod: int) -> int:
        if self.editor_counts is not None:
            return self.editor_counts[pod]
        return self.editors_per_pod

    @property
    def total_editors(self) -> int:
        return sum(self.editors_in_pod(p) for p in range(self.n_pods))

    @property
    def weekly_hours(self) -> float:
        return self.hours_per_day * self.days_per_week


# --------------------------------------------------------------------------- #
# B. Demand
# --------------------------------------------------------------------------- #
ArrivalPattern = Literal["uniform", "front_loaded", "back_loaded", "monday_peak", "midweek_peak"]


class DemandConfig(_Base):
    weekly_target_ads: float = Field(180.0, gt=0, le=2000)
    pod_share: Optional[List[float]] = Field(None, description="Demand share per POD (normalised); None = equal")
    arrival_pattern: ArrivalPattern = "uniform"
    demand_cv: float = Field(0.10, ge=0.0, le=1.0, description="Coefficient of variation of daily arrivals")
    initial_backlog_ads: int = Field(0, ge=0, le=2000, description="Ads already waiting for a brief at t=0")
    initial_ready_briefs_per_pod: int = Field(0, ge=0, le=50, description="Briefs already ready at t=0 (carry-in WIP)")
    sla_days: float = Field(3.0, gt=0, description="Due date = arrival + this many working days (illustrative)")
    urgent_share: float = Field(0.10, ge=0, le=1, description="Share of ads flagged urgent (illustrative)")
    urgent_sla_factor: float = Field(0.5, gt=0, le=1, description="Urgent ads' SLA = sla_days x factor")


# --------------------------------------------------------------------------- #
# C. Ad complexity
# --------------------------------------------------------------------------- #
class ComplexityConfig(_Base):
    mix: Dict[str, float] = Field(default_factory=lambda: {"Simple": 0.30, "Medium": 0.50, "Complex": 0.20})
    standard_hours: Dict[str, float] = Field(default_factory=lambda: {"Simple": 1.5, "Medium": 3.0, "Complex": 5.0})
    editing_time_cv: float = Field(0.25, ge=0, le=1.5, description="Variability of actual vs standard editing hours")
    pod_mix: Optional[List[Optional[Dict[str, float]]]] = Field(
        None, description="Optional per-POD complexity mix (None entry = organisation mix)"
    )

    @field_validator("mix")
    @classmethod
    def _mix_ok(cls, v):
        if set(v) != set(COMPLEXITIES):
            raise ValueError(f"mix must contain exactly {COMPLEXITIES}")
        if any(x < 0 for x in v.values()):
            raise ValueError("mix shares must be >= 0")
        if abs(sum(v.values()) - 1.0) > 1e-6:
            raise ValueError(f"mix must sum to 1.0 (got {sum(v.values()):.4f})")
        return v

    @field_validator("pod_mix")
    @classmethod
    def _pod_mix_ok(cls, v):
        for m in v or []:
            if m is None:
                continue
            if set(m) != set(COMPLEXITIES) or any(x < 0 for x in m.values()) or abs(sum(m.values()) - 1.0) > 1e-6:
                raise ValueError("each pod_mix entry must contain Simple/Medium/Complex, be >= 0 and sum to 1.0")
        return v

    @field_validator("standard_hours")
    @classmethod
    def _hours_ok(cls, v):
        if set(v) != set(COMPLEXITIES) or any(x <= 0 for x in v.values()):
            raise ValueError("standard_hours must be > 0 for all complexities")
        return v

    @property
    def mean_standard_hours(self) -> float:
        return sum(self.mix[c] * self.standard_hours[c] for c in COMPLEXITIES)


# --------------------------------------------------------------------------- #
# D. Quality and review
# --------------------------------------------------------------------------- #
ReviewMode = Literal["strategist", "lead_creative", "dedicated"]


class QualityConfig(_Base):
    first_pass_approval_rate: float = Field(0.85, ge=0.0, le=1.0)
    pod_first_pass: Optional[List[Optional[float]]] = Field(
        None, description="Optional per-POD first-pass approval (None entry = organisation rate)"
    )
    rework_probability: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Must equal 1 - first_pass_approval_rate if supplied (consistency check)",
    )
    rework_repeat_probability: float = Field(
        0.15, ge=0.0, le=1.0, description="Rejection probability at 2nd and later reviews"
    )
    rework_hours_fraction: float = Field(
        0.30, ge=0.0, le=2.0, description="Mean rework duration as a fraction of the ad's standard editing hours"
    )
    rework_cv: float = Field(0.30, ge=0, le=1.5)
    max_rework_loops: int = Field(2, ge=0, le=10)
    on_max_loops: Literal["accept", "cancel"] = "accept"
    review_hours: float = Field(0.25, gt=0, le=8, description="Mean review processing time (productive hours)")
    review_cv: float = Field(0.30, ge=0, le=1.5)
    review_response_delay_hours: float = Field(
        0.0, ge=0, le=48, description="Exogenous feedback/approval latency before an ad becomes reviewable (no resource busy)"
    )
    review_mode: ReviewMode = "strategist"
    dedicated_reviewers: int = Field(2, ge=1, le=20)
    lead_creative_review_share: float = Field(
        0.5, gt=0, le=1.0, description="Share of Lead Creative productive time available for review"
    )
    review_capacity_multiplier: float = Field(1.0, gt=0, le=5.0, description="Review speed adjustment (>1 = faster)")
    review_priority_over_briefing: bool = True
    # strategist briefing (assumptions - not in the playbook)
    brief_hours: float = Field(0.6, gt=0, le=8, description="Mean strategist time per brief (illustrative)")
    brief_cv: float = Field(0.30, ge=0, le=1.5)
    # unavailable assets
    asset_delay_probability: float = Field(0.05, ge=0, le=1, description="P(editing task waits for missing assets)")
    asset_delay_hours: float = Field(1.5, ge=0, le=24, description="Mean wait when assets are missing")
    interruption_rate_per_hour: float = Field(0.0, ge=0, le=2, description="Ad-hoc interruptions per productive hour")
    interruption_hours: float = Field(0.25, ge=0, le=4)

    @field_validator("pod_first_pass")
    @classmethod
    def _pod_fp_ok(cls, v):
        if any(x is not None and not (0.0 <= x <= 1.0) for x in (v or [])):
            raise ValueError("pod_first_pass entries must be between 0 and 1")
        return v

    @model_validator(mode="after")
    def _consistency(self):
        if self.rework_probability is not None:
            if abs(self.rework_probability - (1 - self.first_pass_approval_rate)) > 1e-6:
                raise ValueError(
                    "Inconsistent quality inputs: rework_probability must equal 1 - first_pass_approval_rate "
                    f"({1 - self.first_pass_approval_rate:.3f}), got {self.rework_probability:.3f}"
                )
        return self

    @property
    def first_pass_rework_probability(self) -> float:
        return 1.0 - self.first_pass_approval_rate


# --------------------------------------------------------------------------- #
# E. Lean controls
# --------------------------------------------------------------------------- #
AssignmentPolicy = Literal["pull", "equal_count", "equal_hours", "capacity_aware"]


class LeanConfig(_Base):
    ready_brief_buffer: int = Field(6, ge=1, le=50)
    editing_wip_limit: int = Field(3, ge=1, le=50)
    review_wip_limit: int = Field(3, ge=1, le=50)
    enforce_wip_limits: bool = True
    cross_pod_balancing: bool = False
    cross_pod_review_sharing: bool = False
    cross_pod_min_queue: int = Field(2, ge=1, description="Donor POD must hold at least this many ready briefs")
    priority_scheduling: bool = False
    brief_standardization: bool = False
    assignment_policy: AssignmentPolicy = "pull"
    # explicit assumed effects of brief standardization (illustrative)
    std_brief_time_reduction: float = Field(0.25, ge=0, lt=1)
    std_first_pass_uplift: float = Field(0.05, ge=0, le=0.5)
    std_asset_delay_multiplier: float = Field(0.5, ge=0, le=1)


# --------------------------------------------------------------------------- #
# Thresholds and classification (transparent + configurable)
# --------------------------------------------------------------------------- #
class ThresholdConfig(_Base):
    # Muri utilization bands (management thresholds, NOT universal Lean standards)
    util_spare_below: float = 0.70
    util_normal_upto: float = 0.85
    util_high_upto: float = 0.95
    # POD status (Green/Amber/Red)
    attain_green: float = 0.95
    attain_amber: float = 0.85
    pod_util_red: float = 0.95
    pod_util_amber: float = 0.85
    pod_queue_red_per_editor: float = 3.0   # waiting editing work per editor
    # Mura index normalisation: a CV at/above this reference scores 100 for that component
    mura_cv_reference: float = 0.50
    mura_low: float = 25.0
    mura_high: float = 50.0
    consecutive_high_days: int = 3
    little_min_weeks: int = 3

    @model_validator(mode="after")
    def _ordered(self):
        if not (self.util_spare_below < self.util_normal_upto < self.util_high_upto):
            raise ValueError("utilization thresholds must be increasing")
        if not (self.attain_amber < self.attain_green):
            raise ValueError("attain_amber must be < attain_green")
        return self


class ClassificationConfig(_Base):
    """Lean classification of every activity (VA = value adding, NNVA = necessary non-value-adding, WASTE)."""
    touch: Dict[str, str] = Field(default_factory=lambda: {
        "brief": "VA", "edit": "VA", "review": "NNVA", "rework": "WASTE", "rework_review": "WASTE",
        "overhead": "NNVA",
    })
    waits: Dict[str, str] = Field(default_factory=lambda: {
        "wait_brief_queue": "WASTE", "wait_ready_queue": "WASTE", "wait_asset": "WASTE",
        "wait_blocked_review": "WASTE", "wait_review_external": "WASTE", "wait_review_queue": "WASTE",
        "wait_rework_queue": "WASTE", "wait_unavailable": "WASTE",
    })

    @field_validator("touch", "waits")
    @classmethod
    def _valid_classes(cls, v):
        bad = [k for k, c in v.items() if c not in ("VA", "NNVA", "WASTE")]
        if bad:
            raise ValueError(f"classification must be VA/NNVA/WASTE: {bad}")
        return v


# --------------------------------------------------------------------------- #
# Top level
# --------------------------------------------------------------------------- #
class SimConfig(_Base):
    horizon_weeks: int = Field(5, ge=1, le=12)
    warmup_weeks: int = Field(1, ge=0, le=6)
    seed: int = Field(42, ge=0)
    replications: int = Field(5, ge=1, le=50)
    snapshot_interval_hours: float = Field(1.0, gt=0)
    record_events: bool = True
    org: OrgConfig = Field(default_factory=OrgConfig)
    demand: DemandConfig = Field(default_factory=DemandConfig)
    complexity: ComplexityConfig = Field(default_factory=ComplexityConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    lean: LeanConfig = Field(default_factory=LeanConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)

    @model_validator(mode="after")
    def _cross(self):
        if self.warmup_weeks >= self.horizon_weeks:
            raise ValueError("warmup_weeks must be smaller than horizon_weeks")
        n = self.org.n_pods
        if self.complexity.pod_mix is not None and len(self.complexity.pod_mix) != n:
            raise ValueError("complexity.pod_mix must have one entry per POD")
        if self.quality.pod_first_pass is not None and len(self.quality.pod_first_pass) != n:
            raise ValueError("quality.pod_first_pass must have one entry per POD")
        if self.demand.pod_share is not None:
            if len(self.demand.pod_share) != self.org.n_pods:
                raise ValueError("demand.pod_share must have one entry per POD")
            if sum(self.demand.pod_share) <= 0 or any(s < 0 for s in self.demand.pod_share):
                raise ValueError("demand.pod_share must be non-negative with a positive sum")
        return self

    # ---- derived helpers ------------------------------------------------- #
    @property
    def n_days(self) -> int:
        return self.horizon_weeks * self.org.days_per_week

    @property
    def horizon_hours(self) -> float:
        return self.n_days * self.org.hours_per_day

    @property
    def pod_shares(self) -> List[float]:
        n = self.org.n_pods
        if self.demand.pod_share is None:
            return [1.0 / n] * n
        s = sum(self.demand.pod_share)
        return [x / s for x in self.demand.pod_share]

    @property
    def effective_first_pass(self) -> float:
        fp = self.quality.first_pass_approval_rate
        if self.lean.brief_standardization:
            fp = min(1.0, fp + self.lean.std_first_pass_uplift)
        return fp

    def first_pass_for(self, pod: int) -> float:
        """First-pass approval for a POD (override or organisation rate) incl. the standardisation uplift."""
        fp = self.quality.first_pass_approval_rate
        if self.quality.pod_first_pass is not None and self.quality.pod_first_pass[pod] is not None:
            fp = self.quality.pod_first_pass[pod]
        if self.lean.brief_standardization:
            fp = min(1.0, fp + self.lean.std_first_pass_uplift)
        return fp

    def mix_for(self, pod: int) -> Dict[str, float]:
        if self.complexity.pod_mix is not None and self.complexity.pod_mix[pod] is not None:
            return self.complexity.pod_mix[pod]
        return self.complexity.mix

    def mean_std_hours_for(self, pod: int) -> float:
        m = self.mix_for(pod)
        return sum(m[c] * self.complexity.standard_hours[c] for c in COMPLEXITIES)

    @property
    def support_availability(self) -> float:
        a = self.org.strategist_availability
        return self.org.productive_availability if a is None else a

    @property
    def effective_brief_hours(self) -> float:
        h = self.quality.brief_hours
        if self.lean.brief_standardization:
            h *= 1 - self.lean.std_brief_time_reduction
        return h

    @property
    def effective_asset_delay_probability(self) -> float:
        p = self.quality.asset_delay_probability
        if self.lean.brief_standardization:
            p *= self.lean.std_asset_delay_multiplier
        return p

    def weekday_weights(self) -> List[float]:
        n = self.org.days_per_week
        pat = self.demand.arrival_pattern
        if pat == "uniform":
            w = [1.0] * n
        elif pat == "front_loaded":
            w = [n - i + 1.0 for i in range(n)]
        elif pat == "back_loaded":
            w = [i + 2.0 for i in range(n)]
        elif pat == "monday_peak":
            w = [2.5] + [1.0] * (n - 1)
        else:  # midweek_peak
            mid = (n - 1) / 2
            w = [1.0 + 1.5 * max(0.0, 1 - abs(i - mid) / max(mid, 1)) for i in range(n)]
        s = sum(w)
        return [x / s for x in w]

    def copy_with(self, **sections) -> "SimConfig":
        """Return a deep copy with nested overrides, e.g. copy_with(demand={'weekly_target_ads': 220})."""
        data = self.model_dump()
        for key, val in sections.items():
            if isinstance(val, dict) and isinstance(data.get(key), dict):
                if key == "quality" and "first_pass_approval_rate" in val and "rework_probability" not in val:
                    data[key]["rework_probability"] = None  # derived value: recomputed from the new first-pass rate
                data[key].update(val)
            else:
                data[key] = val
        return SimConfig.model_validate(data)
