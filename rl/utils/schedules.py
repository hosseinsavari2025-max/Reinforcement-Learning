def linear_schedule(start: float, end: float, duration: int, step: int) -> float:
    """Linearly interpolate from ``start`` to ``end`` over ``duration`` steps, then hold at ``end``."""
    frac = min(step / max(duration, 1), 1.0)
    return start + frac * (end - start)
