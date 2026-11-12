"""Finite resources: PODs, workers and the shift calendar.

Calendar model
--------------
The clock is in *working hours*.  Day ``d`` spans ``[d*H, (d+1)*H)``.  Inside a shift
a worker converts clock time to productive time at ``rate`` (= productive availability,
or 0 if the person is absent that day).  Work started before the end of a shift pauses
overnight and resumes on the next present day (pre-emptive resume), so **no resource ever
works outside its calendar** and capacity is never created by assigning more work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .entities import Job, Worker


@dataclass(eq=False)
class PodState:
    idx: int
    backlog: List[Job] = field(default_factory=list)   # demand arrived, brief not started
    ready: List[Job] = field(default_factory=list)     # approved briefs waiting for an editor
    rework: List[Job] = field(default_factory=list)    # rejected ads waiting for an editor
    review_q: List[Job] = field(default_factory=list)  # reviewable ads waiting for a reviewer
    briefing: int = 0                                   # briefs in progress
    editing: int = 0                                    # ads being edited BY this POD's editors (WIP-limit counter)
    editing_owned: int = 0                              # ads OWNED by this POD that are in editing (accounting)
    review_wip: int = 0                                 # ads in the review stage (external delay + queue + in review)
    in_review: int = 0
    arrived: int = 0
    accepted: int = 0
    cancelled: int = 0
    initial: int = 0

    @property
    def wip_released(self) -> int:
        """Ads started (brief in progress or later) but not yet accepted."""
        return self.briefing + len(self.ready) + self.editing_owned + len(self.rework) + self.review_wip

    @property
    def open_total(self) -> int:
        return len(self.backlog) + self.wip_released


def make_absence(seed: int, stream: int, n_days: int, p_absent: float) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3, stream]))
    return rng.random(n_days + 3) < p_absent


def productive_capacity(worker: Worker, n_days: int, hours_per_day: float) -> np.ndarray:
    """Regular productive capacity (hours) per day, excluding overtime; 0 on absent days."""
    cap = np.full(n_days, worker.base_rate * hours_per_day)
    if worker.absent is not None:
        cap[worker.absent[:n_days]] = 0.0
    return cap
