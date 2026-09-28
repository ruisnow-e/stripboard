"""
solver.py — Run the scheduler and report what it did.

`clustered_dp` does the work: partition into regions, then solve exactly inside
each region and across them. A production small enough for the exact solver has
nothing to partition, so it runs with a single region and the result is a proven
optimum.

This module times the run, records which case it landed in, and falls back to
the greedy draft when no partition can be formed. Whether one
can be formed depends on how the locations cluster and not only on how many
there are, so the fallback is triggered by a `PartitionError` actually being
raised rather than predicted from a location count. Any other `ValueError` is a
bad argument and is left to propagate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

from stripboard.scheduling.clustered_dp import (DEFAULT_REGION_CAP,
                                                PartitionError,
                                                clustered_schedule)
from stripboard.scheduling.greedy import greedy_nearest_neighbor
from stripboard.scheduling.subset_dp import INF, ScheduleResult


@dataclass
class SolveReport:
    """The schedule plus how it was produced and how long it took."""
    result: ScheduleResult
    mode: str
    regions: int
    guarantee: str
    elapsed_ms: float

    def __str__(self) -> str:
        return (f"[{self.mode}, {self.regions} region"
                f"{'' if self.regions == 1 else 's'}] "
                f"cost={self.result.total_cost:.2f} "
                f"in {self.elapsed_ms:.1f} ms — {self.guarantee}")


def _greedy_draft(cost: List[List[float]], start: int,
                  return_to_base: bool) -> ScheduleResult:
    """Wrap the greedy draft in the same result type as the exact solvers."""
    draft = greedy_nearest_neighbor(cost, start=start)
    order, legs = list(draft.order), list(draft.cost_breakdown)
    total = draft.total_cost
    if return_to_base and len(order) > 1:
        back = cost[order[-1]][start]
        if back != INF:
            legs.append(back)
            total += back
            order.append(start)
    return ScheduleResult(order=order, total_cost=total, leg_costs=legs,
                          n_locations=len(cost), closed=return_to_base)


def solve(cost: List[List[float]],
          start: int = 0,
          return_to_base: bool = False,
          region_cap: int = DEFAULT_REGION_CAP) -> SolveReport:
    """
    Produce a shooting order and report how it was arrived at.

    Args:
        cost          : n x n all-pairs cost matrix from the geographic layer.
        start         : Base / first location.
        return_to_base: If True the crew returns to `start` at wrap.
        region_cap    : Maximum locations per region.  Has no effect when the
                        whole production already fits the exact solver.

    Returns:
        SolveReport with the schedule, the number of regions used, the guarantee
        that holds for it, and the wall-clock time.

    Raises:
        ValueError: If an argument is invalid, or if the cost matrix is
                    disconnected so that no schedule reaches every location.
    """
    t0 = time.perf_counter()

    try:
        res = clustered_schedule(cost, start=start, region_cap=region_cap,
                                 return_to_base=return_to_base)
    except PartitionError:
        res = _greedy_draft(cost, start, return_to_base)
        return SolveReport(result=res, mode="draft", regions=0,
                           guarantee="fast draft, no optimality guarantee",
                           elapsed_ms=(time.perf_counter() - t0) * 1000.0)

    if res.total_cost == INF:
        # Reporting this as optimal would be a guarantee about a schedule that
        # does not exist.
        raise ValueError("No schedule visits every location: the cost matrix "
                         "is disconnected from the starting location.")

    regions = len(res.regions)
    guarantee = ("globally optimal (proven)" if regions == 1 else
                 "optimal within and between regions; "
                 "the partition is the only approximation")
    return SolveReport(result=res, mode="scheduled", regions=regions,
                       guarantee=guarantee,
                       elapsed_ms=(time.perf_counter() - t0) * 1000.0)
