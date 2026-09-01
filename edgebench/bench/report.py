"""Render results/latest.json into a markdown table and compact chart JSON."""

from __future__ import annotations


def render_markdown_table(payload: dict) -> str:
    """Render the benchmark runs in ``payload`` as a GitHub-flavoured markdown table."""
    header = (
        "| Model | Backend | Graph Opt | Threads | p50 (ms) | p90 (ms) | p99 (ms) | "
        "FPS | CPU % | Peak RSS (MB) |"
    )
    separator = "|---|---|---|---|---|---|---|---|---|---|"
    rows = [header, separator]
    for run in payload["runs"]:
        latency = run["latency"]
        resources = run["resources"]
        rows.append(
            "| {model} | {backend} | {opt} | {threads} | {p50:.2f} | {p90:.2f} | "
            "{p99:.2f} | {fps:.1f} | {cpu:.1f} | {rss:.1f} |".format(
                model=run["model"],
                backend=run["backend"],
                opt=run["graph_opt_level"],
                threads=run["intra_op_threads"],
                p50=latency["p50_ms"],
                p90=latency["p90_ms"],
                p99=latency["p99_ms"],
                fps=latency["fps"],
                cpu=resources["mean_cpu_percent"],
                rss=resources["peak_rss_mb"],
            )
        )
    return "\n".join(rows)


def thread_scaling_series(payload: dict, *, model: str, backend: str, graph_opt_level: str) -> dict:
    """Extract (threads, p50_ms, cpu_percent) points for one model/backend/opt combination,
    sorted by thread count, for the thread-scaling chart."""
    points = [
        {
            "threads": run["intra_op_threads"],
            "p50_ms": run["latency"]["p50_ms"],
            "cpu_percent": run["resources"]["mean_cpu_percent"],
        }
        for run in payload["runs"]
        if run["model"] == model
        and run["backend"] == backend
        and run["graph_opt_level"] == graph_opt_level
    ]
    points.sort(key=lambda p: p["threads"])
    return {
        "model": model,
        "backend": backend,
        "graph_opt_level": graph_opt_level,
        "points": points,
    }


def stage_breakdown_series(payload: dict) -> list[dict]:
    """Extract per-run stage breakdowns for the stacked-bar chart."""
    return [
        {
            "label": f"{run['backend']}/{run['model']}/{run['graph_opt_level']}/t{run['intra_op_threads']}",
            "preprocess_ms": run["stage_breakdown"]["preprocess_ms"],
            "inference_ms": run["stage_breakdown"]["inference_ms"],
            "postprocess_ms": run["stage_breakdown"]["postprocess_ms"],
        }
        for run in payload["runs"]
    ]
