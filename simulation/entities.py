"""Simulation entities: Job (one ad) and Worker (one person)."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# Job states
BACKLOG, BRIEFING, READY, EDITING, BLOCKED, REVIEW, REWORK, APPROVED, CANCELLED = (
    "backlog", "briefing", "ready", "editing", "blocked_for_review", "in_review_stage", "rework_wait",
    "approved", "cancelled",
)
OPEN_STATES = (BACKLOG, BRIEFING, READY, EDITING, BLOCKED, REVIEW, REWORK)

WAIT_CATEGORIES = (
    "wait_brief_queue", "wait_ready_queue", "wait_asset", "wait_blocked_review",
    "wait_review_external", "wait_review_queue", "wait_rework_queue", "wait_unavailable",
)
TOUCH_CATEGORIES = ("brief", "edit", "rework", "review", "rework_review")


@dataclass(eq=False)
class Job:
    job_id: str
    seq: int
    pod: int                       # owning POD (accounting never changes, even on cross-POD transfer)
    complexity: str
    est_hours: float               # standard editing hours
    arrival_time: float
    due_time: float
    urgent: bool
    # pre-drawn random inputs (common random numbers across scenarios)
    actual_hours: float            # editing hours at speed 1.0
    brief_hours: float
    asset_delay: float
    review_hours: List[float]
    review_u: List[float]
    review_resp: List[float]
    rework_hours: List[float]
    planned_editor: Optional[str] = None
    # assignments
    strategist: Optional[str] = None
    editor: Optional[str] = None
    reviewer: Optional[str] = None
    editor_history: List[str] = field(default_factory=list)
    # timestamps
    brief_start: Optional[float] = None
    brief_end: Optional[float] = None
    edit_start: Optional[float] = None
    edit_end: Optional[float] = None          # end of first editing pass
    last_edit_end: Optional[float] = None
    review_start: Optional[float] = None      # first review start
    review_end: Optional[float] = None        # last review end
    first_review_end: Optional[float] = None
    approval_time: Optional[float] = None
    # status
    state: str = BACKLOG
    status: str = "open"           # open / delivered / cancelled
    rework_count: int = 0
    reviews_done: int = 0
    first_pass_approved: Optional[bool] = None
    forced_accept: bool = False
    initial: bool = False
    transferred: bool = False
    transfer_count: int = 0
    handoffs: int = 0
    unnecessary_handoffs: int = 0
    actual_editing_hours: float = 0.0         # productive hours, first editing pass
    edit_elapsed_first: Optional[float] = None
    # accounting
    wait_cat: Optional[str] = None
    wait_since: float = 0.0
    review_ready_at: float = 0.0
    ledger: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    touch: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    overhead: float = 0.0

    # ----- ledger helpers ---------------------------------------------------
    def enter_wait(self, cat: str, now: float) -> None:
        self.wait_cat = cat
        self.wait_since = now

    def leave_wait(self, now: float) -> None:
        if self.wait_cat is not None:
            self.ledger[self.wait_cat] += max(0.0, now - self.wait_since)
            self.wait_cat = None

    # ----- derived ----------------------------------------------------------
    @property
    def total_touch(self) -> float:
        return float(sum(self.touch.values()))

    @property
    def total_wait(self) -> float:
        return float(sum(self.ledger.get(c, 0.0) for c in WAIT_CATEGORIES))

    @property
    def lead_time(self) -> Optional[float]:
        if self.approval_time is None:
            return None
        return self.approval_time - self.arrival_time

    @property
    def rework_hours_done(self) -> float:
        return self.touch.get("rework", 0.0)


@dataclass(eq=False)
class Worker:
    wid: str
    role: str                      # editor | strategist | reviewer | lead_creative
    pod: int                       # -1 for pooled roles
    base_rate: float               # productive share of a shift
    speed: float = 1.0
    can_complex: bool = True
    absent: Optional[np.ndarray] = None
    state: str = "idle"
    current_job: Optional[str] = None
    busy: Dict[tuple, float] = field(default_factory=lambda: defaultdict(float))        # (day, activity) -> productive h
    idle: Dict[tuple, float] = field(default_factory=lambda: defaultdict(float))        # (day, cause) -> productive-equivalent h
    overhead: Dict[int, float] = field(default_factory=lambda: defaultdict(float))      # day -> non-productive clock h while working
    jobs_completed: int = 0
