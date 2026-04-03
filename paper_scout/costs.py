from __future__ import annotations

import threading


class CostTracker:
    """Accumulates token usage and computes dollar costs across API calls."""

    PRICING: dict[str, tuple[float, float]] = {
        "claude-sonnet-4-6": (3.00, 15.00),
        "claude-opus-4-6": (5.00, 25.00),
        "claude-haiku-4-5": (1.00, 5.00),
    }
    DEFAULT_PRICING: tuple[float, float] = (3.00, 15.00)

    def __init__(self) -> None:
        self._records: list[tuple[str, int, int, float]] = []
        self._lock = threading.Lock()

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Record usage from one API call. Returns the cost of this call in USD."""
        normalized_model = (model or "").strip().lower()
        input_count = _coerce_non_negative_int(input_tokens)
        output_count = _coerce_non_negative_int(output_tokens)
        input_price, output_price = self._pricing_for_model(normalized_model)

        cost_usd = ((input_count * input_price) + (output_count * output_price)) / 1_000_000.0

        with self._lock:
            self._records.append(
                (normalized_model or "unknown", input_count, output_count, cost_usd)
            )

        return cost_usd

    @property
    def total_input_tokens(self) -> int:
        with self._lock:
            return sum(record[1] for record in self._records)

    @property
    def total_output_tokens(self) -> int:
        with self._lock:
            return sum(record[2] for record in self._records)

    @property
    def total_cost_usd(self) -> float:
        with self._lock:
            return sum(record[3] for record in self._records)

    @property
    def call_count(self) -> int:
        with self._lock:
            return len(self._records)

    def summary(self) -> str:
        """Human-readable summary: 'X calls, Y input tokens, Z output tokens, $A.BC'"""
        with self._lock:
            call_count = len(self._records)
            total_input = sum(record[1] for record in self._records)
            total_output = sum(record[2] for record in self._records)
            total_cost = sum(record[3] for record in self._records)

        return (
            f"{call_count} calls, {total_input} input tokens, "
            f"{total_output} output tokens, ${total_cost:.2f}"
        )

    def _pricing_for_model(self, model: str) -> tuple[float, float]:
        if not model:
            return self.DEFAULT_PRICING
        if model in self.PRICING:
            return self.PRICING[model]
        for known_model, pricing in self.PRICING.items():
            if model.startswith(known_model):
                return pricing
        return self.DEFAULT_PRICING


def _coerce_non_negative_int(value: object) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)