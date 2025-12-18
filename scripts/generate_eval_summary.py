from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from fx_kline.analyst import data_manager


def parse_date(raw: str) -> date:
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:  # noqa: B904
        raise argparse.ArgumentTypeError(f"Invalid date format: {raw}") from exc


def format_percent(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def format_signed(value: Optional[float], digits: int = 1) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.{digits}f}"


def format_ratio(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def format_money(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:,.0f}"


def format_bool(value: Optional[bool]) -> str:
    if value is None:
        return "N/A"
    return "OK" if value else "NG"


@dataclass
class StrategyRow:
    pair: str
    rank: Optional[int]
    direction: str
    pips: Optional[float]
    dir_ok: Optional[bool]
    entry_hit: Optional[bool]
    sl_hit: Optional[bool]
    tp_hit: Optional[bool]
    rr: Optional[float]
    risk_percent: Optional[float]
    risk_amount_100k: Optional[float]
    reward_amount_100k: Optional[float]


@dataclass
class EvalSummary:
    label: str
    mode: str
    path: Path
    total_strategies: int
    correct: int
    entry_hits: int
    pips_total: Optional[float]
    pips_count: int
    rr_sum: float
    rr_count: int
    rank_total: Dict[str, int]
    rank_correct: Dict[str, int]
    avg_confidence: Optional[float]
    avg_calibration: Optional[float]
    aggregated_direction_accuracy: Optional[float]
    aggregated_rank1_accuracy: Optional[float]
    aggregated_entry_hit_rate: Optional[float]
    aggregated_avg_pips: Optional[float]
    total_risk_100k: float
    total_reward_100k: float
    rows: List[StrategyRow]

    @property
    def direction_accuracy(self) -> Optional[float]:
        if self.aggregated_direction_accuracy is not None:
            return float(self.aggregated_direction_accuracy)
        if self.total_strategies == 0:
            return None
        return self.correct / self.total_strategies

    @property
    def rank1_accuracy(self) -> Optional[float]:
        if self.aggregated_rank1_accuracy is not None:
            return float(self.aggregated_rank1_accuracy)
        total = self.rank_total.get("1", 0)
        if total == 0:
            return None
        return self.rank_correct.get("1", 0) / total

    @property
    def entry_hit_rate(self) -> Optional[float]:
        if self.aggregated_entry_hit_rate is not None:
            return float(self.aggregated_entry_hit_rate)
        if self.total_strategies == 0:
            return None
        return self.entry_hits / self.total_strategies

    @property
    def avg_rr(self) -> Optional[float]:
        if self.rr_count == 0:
            return None
        return round(self.rr_sum / self.rr_count, 2)

    @property
    def avg_pips(self) -> Optional[float]:
        if self.aggregated_avg_pips is not None:
            return float(self.aggregated_avg_pips)
        if self.pips_count == 0 or self.pips_total is None:
            return None
        return round(self.pips_total / self.pips_count, 1)


def load_json(path: Path) -> Dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # noqa: B904
        raise RuntimeError(f"Failed to parse JSON: {path}") from exc


def build_eval_summary(path: Path, mode: str) -> EvalSummary:
    data = load_json(path)
    strategies = data.get("strategy_evaluations", []) or []
    aggregated = data.get("aggregated_metrics", {}) or {}

    correct = 0
    entry_hits = 0
    rr_sum = 0.0
    rr_count = 0
    pips_values: List[float] = []
    rank_total: Dict[str, int] = {}
    rank_correct: Dict[str, int] = {}
    rows: List[StrategyRow] = []
    total_risk_100k = 0.0
    total_reward_100k = 0.0

    for strategy in strategies:
        metrics = strategy.get("metrics", {}) or {}
        rank = strategy.get("rank")
        rank_key = str(rank)
        rank_total[rank_key] = rank_total.get(rank_key, 0) + 1

        if metrics.get("direction_correct"):
            correct += 1
            rank_correct[rank_key] = rank_correct.get(rank_key, 0) + 1

        if metrics.get("entry_hit"):
            entry_hits += 1

        rr = metrics.get("risk_reward_realized")
        if rr is not None:
            rr_sum += float(rr)
            rr_count += 1

        pips = metrics.get("pips_outcome")
        if pips is not None:
            pips_values.append(float(pips))

        risk_percent = None
        risk_amount_100k = None
        reward_amount_100k = None

        prediction = strategy.get("prediction") or {}
        if isinstance(prediction, dict):
            rp = prediction.get("risk_percent")
            if rp is not None:
                risk_percent = float(rp)
                risk_amount_100k = round(100000 * risk_percent / 100, 2)

        if (
            risk_amount_100k is not None
            and metrics.get("entry_hit")
            and rr is not None
        ):
            reward_amount_100k = round(risk_amount_100k * float(rr), 2)
            total_risk_100k += risk_amount_100k
            total_reward_100k += reward_amount_100k

        rows.append(
            StrategyRow(
                pair=str(strategy.get("pair", "")),
                rank=int(rank) if isinstance(rank, int) else None,
                direction=str(strategy.get("direction", "")) if strategy.get("direction") is not None else "",
                pips=float(pips) if pips is not None else None,
                dir_ok=metrics.get("direction_correct"),
                entry_hit=metrics.get("entry_hit"),
                sl_hit=metrics.get("stop_loss_hit"),
                tp_hit=metrics.get("take_profit_hit"),
                rr=float(rr) if rr is not None else None,
                risk_percent=risk_percent,
                risk_amount_100k=risk_amount_100k,
                reward_amount_100k=reward_amount_100k,
            )
        )

    pips_total = round(sum(pips_values), 1) if pips_values else None

    return EvalSummary(
        label=path.stem,
        mode=mode,
        path=path,
        total_strategies=len(strategies),
        correct=correct,
        entry_hits=entry_hits,
        pips_total=pips_total,
        pips_count=len(pips_values),
        rr_sum=rr_sum,
        rr_count=rr_count,
        rank_total=rank_total,
        rank_correct=rank_correct,
        avg_confidence=aggregated.get("avg_confidence"),
        avg_calibration=aggregated.get("avg_confidence_calibration"),
        aggregated_direction_accuracy=aggregated.get("direction_accuracy"),
        aggregated_rank1_accuracy=(aggregated.get("accuracy_by_rank") or {}).get("1"),
        aggregated_entry_hit_rate=aggregated.get("entry_hit_rate"),
        aggregated_avg_pips=aggregated.get("avg_pips"),
        total_risk_100k=round(total_risk_100k, 2),
        total_reward_100k=round(total_reward_100k, 2),
        rows=rows,
    )


def combine_summaries(label: str, mode: str, summaries: Sequence[EvalSummary]) -> EvalSummary:
    total_strategies = sum(s.total_strategies for s in summaries)
    correct = sum(s.correct for s in summaries)
    entry_hits = sum(s.entry_hits for s in summaries)
    pips_values = [s.pips_total for s in summaries if s.pips_total is not None]
    pips_total = round(sum(pips_values), 1) if pips_values else None
    pips_count = sum(s.pips_count for s in summaries)
    rr_sum = sum(s.rr_sum for s in summaries)
    rr_count = sum(s.rr_count for s in summaries)
    total_risk_100k = round(sum(s.total_risk_100k for s in summaries), 2)
    total_reward_100k = round(sum(s.total_reward_100k for s in summaries), 2)

    rank_total: Dict[str, int] = {}
    rank_correct: Dict[str, int] = {}
    for summary in summaries:
        for rank, count in summary.rank_total.items():
            rank_total[rank] = rank_total.get(rank, 0) + count
        for rank, count in summary.rank_correct.items():
            rank_correct[rank] = rank_correct.get(rank, 0) + count

    return EvalSummary(
        label=label,
        mode=mode,
        path=Path(label),
        total_strategies=total_strategies,
        correct=correct,
        entry_hits=entry_hits,
        pips_total=pips_total,
        pips_count=pips_count,
        rr_sum=rr_sum,
        rr_count=rr_count,
        rank_total=rank_total,
        rank_correct=rank_correct,
        avg_confidence=None,
        avg_calibration=None,
        aggregated_direction_accuracy=None,
        aggregated_rank1_accuracy=None,
        aggregated_entry_hit_rate=None,
        aggregated_avg_pips=None,
        total_risk_100k=total_risk_100k,
        total_reward_100k=total_reward_100k,
        rows=[],
    )


def render_eval_block(lines: List[str], title: str, summary: EvalSummary, max_rows: int) -> None:
    lines.append(f"### {title}")
    lines.append(f"- file: {summary.path.as_posix()}")
    lines.append(
        f"- dir_acc: {format_percent(summary.direction_accuracy)} "
        f"({summary.correct}/{summary.total_strategies})"
    )
    lines.append(f"- rank1_acc: {format_percent(summary.rank1_accuracy)}")
    lines.append(f"- entry_hit_rate: {format_percent(summary.entry_hit_rate)}")
    lines.append(
        f"- total_pips: {format_signed(summary.pips_total)} "
        f"(avg {format_signed(summary.avg_pips)} per filled trade)"
    )
    lines.append(f"- avg_rr: {format_ratio(summary.avg_rr)} (samples={summary.rr_count})")
    lines.append(
        f"- filled risk @100k: {format_money(summary.total_risk_100k)} "
        f"=> reward: {format_money(summary.total_reward_100k)}"
    )

    if summary.avg_confidence is not None or summary.avg_calibration is not None:
        conf = f"{summary.avg_confidence:.2f}" if summary.avg_confidence is not None else "N/A"
        cal = f"{summary.avg_calibration:.3f}" if summary.avg_calibration is not None else "N/A"
        lines.append(f"- confidence: avg={conf}, calibration_error={cal}")

    if not summary.rows:
        lines.append("")
        return

    lines.append("")
    lines.append(
        "| pair | rank | dir | pips | dir_ok | entry | SL | TP | RR | "
        "risk% | risk@100k | reward@100k |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )

    sorted_rows = sorted(
        summary.rows,
        key=lambda r: (r.rank is None, r.rank if r.rank is not None else 9999, r.pair),
    )
    limited_rows = sorted_rows[:max_rows]

    for row in limited_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.pair or "-",
                    str(row.rank) if row.rank is not None else "-",
                    row.direction or "-",
                    format_signed(row.pips),
                    format_bool(row.dir_ok),
                    format_bool(row.entry_hit),
                    format_bool(row.sl_hit),
                    format_bool(row.tp_hit),
                    format_ratio(row.rr),
                    format_ratio(row.risk_percent),
                    format_money(row.risk_amount_100k),
                    format_money(row.reward_amount_100k),
                ]
            )
            + " |"
        )

    if len(sorted_rows) > len(limited_rows):
        remaining = len(sorted_rows) - len(limited_rows)
        lines.append(f"*... {remaining} more strategies not shown*")

    lines.append("")


def render_summary(
    target_date: date,
    l3_summaries: Sequence[EvalSummary],
    l4_summaries: Sequence[EvalSummary],
    output_path: Path,
    max_rows: int,
) -> None:
    lines: List[str] = []
    lines.append(f"# Daily Evaluation Summary ({target_date.isoformat()} JST)")
    lines.append("")

    lines.append("## L3 (AI)")
    if not l3_summaries:
        lines.append("- No L3 evaluation JSON found.")
        lines.append("")
    else:
        if len(l3_summaries) > 1:
            combined = combine_summaries("L3_combined", "ai", l3_summaries)
            lines.append("- Combined view across all L3 evaluations:")
            lines.append(
                f"  - dir_acc: {format_percent(combined.direction_accuracy)} "
                f"({combined.correct}/{combined.total_strategies})"
            )
            lines.append(f"  - rank1_acc: {format_percent(combined.rank1_accuracy)}")
            lines.append(
                f"  - total_pips: {format_signed(combined.pips_total)} "
                f"(avg {format_signed(combined.avg_pips)} per filled trade)"
            )
            lines.append(
                f"  - entry_hit_rate: {format_percent(combined.entry_hit_rate)}, "
                f"avg_rr: {format_ratio(combined.avg_rr)} (samples={combined.rr_count})"
            )
            lines.append(
                f"  - filled risk @100k: {format_money(combined.total_risk_100k)} "
                f"=> reward: {format_money(combined.total_reward_100k)}"
            )
            lines.append("")

        for summary in l3_summaries:
            render_eval_block(lines, summary.label, summary, max_rows)

    lines.append("## L4 (HITL)")
    if not l4_summaries:
        lines.append("- No L4 evaluation JSON found.")
        lines.append("")
    else:
        if len(l4_summaries) > 1:
            combined = combine_summaries("L4_combined", "hitl", l4_summaries)
            lines.append("- Combined view across all L4 evaluations:")
            lines.append(
                f"  - dir_acc: {format_percent(combined.direction_accuracy)} "
                f"({combined.correct}/{combined.total_strategies})"
            )
            lines.append(f"  - rank1_acc: {format_percent(combined.rank1_accuracy)}")
            lines.append(
                f"  - total_pips: {format_signed(combined.pips_total)} "
                f"(avg {format_signed(combined.avg_pips)} per filled trade)"
            )
            lines.append(
                f"  - entry_hit_rate: {format_percent(combined.entry_hit_rate)}, "
                f"avg_rr: {format_ratio(combined.avg_rr)} (samples={combined.rr_count})"
            )
            lines.append("")

        for summary in l4_summaries:
            render_eval_block(lines, summary.label, summary, max_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def find_evaluations(day_dir: Path) -> tuple[List[EvalSummary], List[EvalSummary]]:
    l3_summaries: List[EvalSummary] = []
    for path in sorted(day_dir.glob("L3_evaluation*.json")):
        l3_summaries.append(build_eval_summary(path, "ai"))

    l4_summaries: List[EvalSummary] = []
    for path in sorted(day_dir.glob("L4_evaluation*.json")):
        l4_summaries.append(build_eval_summary(path, "hitl"))

    return l3_summaries, l4_summaries


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Render Markdown summaries from L3/L4 evaluation JSON files.")
    parser.add_argument("--date", required=True, type=parse_date, help="Target JST date (YYYY-MM-DD).")
    parser.add_argument("--data-root", type=Path, default=data_manager.get_data_root(), help="Root data directory.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output Markdown file path. Defaults to data_root/YYYY/MM/DD/evaluation_summary.md",
    )
    parser.add_argument("--max-rows", type=int, default=10, help="Maximum strategies to include per table.")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args(argv)
    day_dir = args.data_root / f"{args.date.year:04d}" / f"{args.date.month:02d}" / f"{args.date.day:02d}"
    output_path = args.output or (day_dir / "evaluation_summary.md")

    l3_summaries, l4_summaries = find_evaluations(day_dir)
    render_summary(args.date, l3_summaries, l4_summaries, output_path, args.max_rows)

    if args.verbose:
        print(f"Wrote summary to {output_path.as_posix()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
