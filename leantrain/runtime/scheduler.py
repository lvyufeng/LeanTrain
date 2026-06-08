"""Runtime scheduler placeholder."""

from __future__ import annotations

from leantrain.planner.graph import TaskGraph


class RuntimeScheduler:
    """Executes planner task graphs while enforcing memory budgets."""

    def run(self, graph: TaskGraph) -> None:
        raise NotImplementedError("runtime execution is not implemented yet")
