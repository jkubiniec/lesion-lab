#!/usr/bin/env python3
"""Rebuild auditable tables and offline SVGs from a Lesion Lab run (stdlib only).

Run: python3 analyze_large_run.py runs/large-16384-seed42 --output results/latest
This analyzes one evolved world. Bootstrap intervals resample sampled individuals;
they do not estimate reproducibility across independent evolutionary runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Iterable

ARMS = ("sham", "designated_region", "random")
LABELS = {"sham": "Sham", "designated_region": "Designated region", "random": "Scattered units"}
COLORS = {"sham": "#007C83", "designated_region": "#B53C36", "random": "#664FA3", "baseline": "#243B53"}
METRICS = {"accuracy": ("correct", "episodes", "Overall"), "go_accuracy": ("go_correct", "go_trials", "Go choice"), "nogo_accuracy": ("nogo_correct", "nogo_trials", "No-go withholding")}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def safe_rate(record: dict, metric: str):
    correct, total, _ = METRICS[metric]
    if correct in record and total in record:
        return record[correct] / record[total] if record[total] else None
    return record.get(metric)


def percentile(values: list[float], p: float) -> float:
    """Linearly interpolated percentile, with explicit finite-sample endpoints."""
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    low, high = math.floor(k), math.ceil(k)
    return ordered[low] + (ordered[high] - ordered[low]) * (k - low)


def bootstrap_mean(values: list[float], repeats: int = 2000, seed: int = 4927) -> tuple:
    if not values:
        return None, None, None
    average = mean(values)
    if len(values) == 1 or max(values) == min(values):
        return average, average, average
    rng = random.Random(seed)
    n = len(values)
    estimates = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(repeats)]
    return average, percentile(estimates, .025), percentile(estimates, .975)


def paired_effects(baseline: dict, intervention: dict, *, injured_only: bool = True,
                   repeats: int = 2000) -> list[dict]:
    """Pair only stable identities in the same task slot with identical trials.

    Each bootstrap item is an individual's average difference across trials. This
    avoids incorrectly treating every repeated trial as a new independent agent.
    The immediate injured cohort is selected by the realized mask, not all samples.
    """
    for key in ("seed", "trials"):
        if key in baseline and key in intervention and baseline[key] != intervention[key]:
            raise ValueError(f"Cannot pair assays with unmatched {key}")
    before = {row["id"]: row for row in baseline.get("per_sample", [])}
    pairs = []
    for after in intervention.get("per_sample", []):
        if injured_only and not (after.get("injured") or after.get("disabled_count", 0)):
            continue
        previous = before.get(after["id"])
        if previous is None:
            continue
        if previous.get("slot") != after.get("slot"):
            raise ValueError("Cannot pair an individual evaluated in different task slots")
        for key in ("episodes", "go_trials", "nogo_trials", "target_per_trial", "go_per_trial"):
            if key in previous and key in after and previous[key] != after[key]:
                raise ValueError(f"Cannot pair unmatched task trials ({key})")
        pairs.append((previous, after))
    rows = []
    for metric in METRICS:
        deltas = [safe_rate(a, metric) - safe_rate(b, metric) for b, a in pairs
                  if safe_rate(a, metric) is not None and safe_rate(b, metric) is not None]
        average, low, high = bootstrap_mean(deltas, repeats=repeats)
        trial_counts = {"paired_trials": None, "trials_gained": None, "trials_lost": None,
                        "actions_changed": None, "action_change_fraction": None}
        if pairs and all(all(key in record for key in ("correct_per_trial", "action_per_trial", "go_per_trial"))
                         for pair in pairs for record in pair):
            trial_counts = {key: 0 for key in ("paired_trials", "trials_gained", "trials_lost", "actions_changed")}
            for b, a in pairs:
                streams = [b["correct_per_trial"], a["correct_per_trial"],
                           b["action_per_trial"], a["action_per_trial"], a["go_per_trial"]]
                if len({len(v) for v in streams}) != 1:
                    raise ValueError("Cannot compare inconsistent per-trial stream lengths")
                for old_correct, new_correct, old_action, new_action, is_go in zip(*streams):
                    if metric == "go_accuracy" and is_go != "1": continue
                    if metric == "nogo_accuracy" and is_go != "0": continue
                    trial_counts["paired_trials"] += 1
                    trial_counts["trials_gained"] += old_correct == "0" and new_correct == "1"
                    trial_counts["trials_lost"] += old_correct == "1" and new_correct == "0"
                    trial_counts["actions_changed"] += old_action != new_action
            trial_counts["action_change_fraction"] = (trial_counts["actions_changed"] / trial_counts["paired_trials"]
                                                       if trial_counts["paired_trials"] else None)
        rows.append({"metric": metric, "cohort": "injured_at_immediate" if injured_only else "all_matched",
                     "paired_individuals": len(deltas), "mean_delta": average,
                     "bootstrap_ci95_low": low, "bootstrap_ci95_high": high,
                     "individuals_improved": sum(d > 0 for d in deltas),
                     "individuals_worsened": sum(d < 0 for d in deltas),
                     "individuals_unchanged": sum(d == 0 for d in deltas),
                     **trial_counts, "bootstrap_repeats": repeats, "resampling_unit": "sampled_individual",
                     "inference_scope": "within_one_evolved_world_only"})
    return rows


def read_progress(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if "decisions" not in row:
            continue
        if rows and row["decisions"] == rows[-1]["decisions"]:
            rows[-1] = row
        else:
            rows.append(row)
    return rows


def warmup_progress(run: Path) -> list[dict]:
    """Stitch optional earlier metrics into a resumed run, auditing shared counters."""
    all_rows = read_progress(run / "prefix_warmup.jsonl") + read_progress(run / "warmup.jsonl")
    by_decision = {}
    counters = ("episodes", "correct", "go_trials", "go_correct", "nogo_trials", "nogo_correct", "births", "deaths", "neural_evaluations")
    for row in all_rows:
        previous = by_decision.get(row["decisions"])
        if previous is not None:
            for key in counters:
                if key in previous and key in row and previous[key] != row[key]:
                    raise ValueError(f"Conflicting prefix and resumed counters at {row['decisions']}: {key}")
        by_decision[row["decisions"]] = row
    result = [by_decision[k] for k in sorted(by_decision)]
    for before, after in zip(result, result[1:]):
        for key in counters:
            if key in before and key in after and after[key] < before[key]:
                raise ValueError(f"Cumulative counter moved backwards: {key}")
    return result


def interval_progress(rows: list[dict], arm: str) -> list[dict]:
    result = []
    for before, after in zip(rows, rows[1:]):
        item = {"arm": arm, "decisions": after["decisions"], "start_decisions": before["decisions"],
                "births": after.get("births", 0), "births_interval": after.get("births", 0) - before.get("births", 0),
                "injured_alive": after.get("injured_alive", 0), "injured_initial": after.get("injured_initial", 0)}
        for metric, (correct, total, _) in METRICS.items():
            denominator = after.get(total, 0) - before.get(total, 0)
            item[metric] = (after.get(correct, 0) - before.get(correct, 0)) / denominator if denominator else None
        result.append(item)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        if fields:
            writer = csv.DictWriter(f, fields)
            writer.writeheader()
            writer.writerows(rows)


class Svg:
    def __init__(self, width=1200, height=700, title="", desc=""):
        self.width, self.height = width, height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
                      f'<title id="title">{html.escape(title)}</title><desc id="description">{html.escape(desc)}</desc>',
                      '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#526579"/></marker></defs>',
                      f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>']
        self.text(38, 44, title, 26, "#172D42", weight="700")

    def text(self, x, y, value, size=15, color="#253C50", anchor="start", weight="400", halo=False):
        extra = ' stroke="#FFFFFF" stroke-width="3" stroke-linejoin="round" paint-order="stroke"' if halo else ''
        self.parts.append(f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}"{extra}>{html.escape(str(value))}</text>')

    def lines(self, x, y, values, size=15, color="#253C50", gap=23, anchor="start"):
        for i, value in enumerate(values):
            self.text(x, y+i*gap, value, size, color, anchor)

    def line(self, x1, y1, x2, y2, color="#AAB7C4", width=1, dash=None, arrow=False):
        extra = (' stroke-dasharray="6 5"' if dash else '') + (' marker-end="url(#arrow)"' if arrow else '')
        self.parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"{extra}/>')

    def rect(self, x, y, w, h, color, stroke=None, radius=0):
        self.parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{radius}" fill="{color}"' + (f' stroke="{stroke}"' if stroke else '') + '/>')

    def circle(self, x, y, radius, color):
        self.parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{color}"/>')

    def polyline(self, points: Iterable, color, width=3):
        pts = ' '.join(f'{x:.2f},{y:.2f}' for x, y in points)
        self.parts.append(f'<polyline fill="none" points="{pts}" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"/>')

    def box(self, x, y, w, h, title, lines, color="#EFF4F8", accent="#243B53"):
        self.rect(x, y, w, h, color, "#C5D1DA", 10)
        self.text(x+18, y+30, title, 18, accent, weight="700")
        self.lines(x+18, y+58, lines, 15, gap=23)

    def save(self, path: Path):
        path.write_text('\n'.join(self.parts) + '\n</svg>\n')


def axes(svg: Svg, x, y, w, h, xmin, xmax, ymin=0, ymax=1, *, percent=True, xticks=4, yticks=4, points=False):
    for k in range(yticks+1):
        value = ymin + (ymax-ymin)*k/yticks
        yy = y+h-h*k/yticks
        svg.line(x, yy, x+w, yy, "#E2E8EE")
        tick = f"{value*100:+.1f} pp" if points else f"{value*100:.0f}%" if percent else f"{value:,.0f}"
        svg.text(x-12, yy+5, tick, 13, anchor="end")
    for k in range(xticks+1):
        value = xmin+(xmax-xmin)*k/xticks
        xx = x+w*k/xticks
        svg.text(xx, y+h+25, f"{value:g}", 13, anchor="middle")
    svg.line(x, y+h, x+w, y+h, "#71859A", 1.3)
    svg.line(x, y, x, y+h, "#71859A", 1.3)
    return lambda xx, yy: (x + (xx-xmin)/(xmax-xmin or 1)*w, y+h-(yy-ymin)/(ymax-ymin or 1)*h)


def architecture_figure(path: Path, manifest: dict):
    h = manifest['hidden']
    svg = Svg(1200, 765, "What an acquired injury changes", "Inherited weights, transient memory, and injury mask are distinct. Injury clamps selected recurrent units to zero without editing weights. Newborns inherit mutated weights and a healthy mask.")
    svg.text(38, 76, f"{manifest['population']:,} resident agents · {h} recurrent units each · {manifest.get('parameters_per_agent', h*h+12*h+3):,} parameters per agent", 16)
    svg.box(38, 112, 292, 134, "Inherited weights W", ["Input + recurrent + action weights", "Copied and mutated at birth", "Not changed by the intervention"], accent="#007C83")
    svg.box(454, 112, 292, 134, "Temporary state h", ["Carries information within a trial", "Reset at each cue", "Not inherited by offspring"])
    svg.box(870, 112, 292, 134, "Acquired mask m", ["Selected units forced to zero", "Persists in the same individual", "New offspring have m = 0"], color="#FFF1EC", accent="#B53C36")
    svg.line(330, 180, 442, 180, arrow=True)
    svg.line(870, 180, 758, 180, color="#B53C36", arrow=True)
    svg.text(600, 291, "h(t+1) = (1 − m) × tanh(Wrec h(t) + Win x(t) + b)", 23, anchor="middle", weight="700")
    svg.box(38, 329, 300, 131, "Eight-step task", ["Cue → distractors → probe", "Probe supplies the current rule", "Choose left, right, or withhold"])
    svg.box(451, 329, 300, 131, "Recurrent controller", [f"{h} interacting hidden units", "Masked units send no activity", "Three action scores → argmax"], color="#EDF7F7", accent="#007C83")
    svg.box(864, 329, 300, 131, "Energy and selection", ["Correct responses earn energy", "Threshold triggers supplied birth", "Low-energy agents can die"])
    svg.line(338, 395, 438, 395, arrow=True)
    svg.line(751, 395, 851, 395, arrow=True)
    svg.box(38, 509, 538, 159, "Injured survivor", ["Same identity, same inherited weights, same mask", "Disabled units remain disabled during follow-up", "Changes in behavior can come from altered dynamics", "No plasticity or repair process is implemented"], color="#FFF1EC", accent="#B53C36")
    svg.box(624, 509, 538, 159, "New individual at a population slot", ["New identity and parent record", "Inherited weights copied with random mutation", "Empty memory state and healthy injury mask", "Replacement is not healing of the injured individual"], color="#EDF7F7", accent="#007C83")
    svg.lines(38, 709, ["The final quarter is a designated partition, not a validated frontal lobe analogue.", "This is a computational lesion model; biological and clinical validity remain untested."], 15, gap=23)
    svg.save(path)


def bff_figure(path: Path):
    svg = Svg(1200, 717, "From BFF inspiration to an engineered neural population", "The linked apankrat/bff repository is a Brainfuck interpreter. The computational-life experiment uses self-modifying programs and emergent replication. Lesion Lab supplies neural controllers and birth rules explicitly.")
    svg.box(38, 100, 340, 191, "Linked apankrat/bff repository", ["Brainfuck interpreter", "Executes source instructions", "Not a neural controller", "Does not supply a population or", "the interview's replication world"])
    svg.box(430, 100, 730, 191, "Computational Life / CuBFF research inspiration", ["Mutable instruction tapes interact in a shared environment", "Programs can copy instructions into other tapes", "Replication is an empirical emergence criterion", "This is the relevant reproduction reference", "Neural weights are a different substrate"], color="#EDF7F7", accent="#007C83")
    svg.text(600, 333, "Conceptual inspiration: evolvable computation + controlled perturbation", 17, anchor="middle")
    svg.line(600, 350, 600, 384, arrow=True)
    svg.box(38, 401, 340, 179, "Lesion Lab neural substrate", ["Randomly initialized recurrent agents", "Weights are inherited genotypes", "Task performance changes energy", "Birth and mutation are explicit", "Evolution selects among lineages"], color="#EDF7F7", accent="#007C83")
    svg.box(430, 401, 340, 179, "Controlled acquired injury", ["Fork a single common checkpoint", "Sham vs designated vs scattered", "Match disabled-unit counts", "Apply a persistent identity mask", "Keep inherited weights intact"], color="#FFF1EC", accent="#B53C36")
    svg.box(822, 401, 340, 179, "Auditable comparisons", ["Frozen-weight task assays", "Pair immediate tests by identity", "Track births and injured survivors", "Test cue/memory dependence", "Save code, data, and checkpoints"])
    svg.line(378, 490, 418, 490, arrow=True)
    svg.line(770, 490, 810, 490, arrow=True)
    svg.lines(38, 628, ["Expansion: a neural state-and-inheritance model with persistent lesions and matched controls.", "The neural experiment does not demonstrate spontaneous reproduction, consciousness, psychological trauma,", "or human traumatic brain injury. Reproduction is supplied; functional localization is unverified."], 15, gap=23)
    svg.save(path)


def relevance_figure(path: Path):
    svg = Svg(1200, 1056, "What this experiment can add to the BFF research question",
              "Computational Life asks whether self-copying emerges from mutable program interactions. Lesion Lab separately asks how acquired capacity loss affects evolved task behavior, using common-checkpoint lesion forks, paired assays and identity retention. Its birth rule is engineered; it makes no claim of BFF emergence or a biological frontal lobe.")
    svg.text(38, 78, "Different hypotheses and measurement units; a shared interest in evolvable computation under perturbation.", 16)
    svg.box(38, 108, 540, 215, "Computational Life / CuBFF", [
        "Question: can self-copying emerge?",
        "Substrate: interacting, self-modifying instruction tapes",
        "Copying must be performed by executable programs",
        "Evidence: operational replication from random origins",
        "Units: program interactions / instruction execution",
        "This is the reproduction research reference"], accent="#007C83")
    svg.box(622, 108, 540, 215, "Lesion Lab: a separate testable question", [
        "Question: what changes when capacity is lost?",
        "Substrate: evolved recurrent neural controllers",
        "Birth copies and mutates weights by an explicit rule",
        "Evidence: task effects and original-identity retention",
        "Units: neural updates; eight updates per task episode",
        "Its counts are not equivalent to BFF interactions"], accent="#B53C36", color="#FFF1EC")
    svg.text(600, 362, "Extension in experimental method: controlled acquired lesions + matched task measurements", 16, anchor="middle", weight="700")
    svg.box(385, 396, 430, 86, "Shared healthy evolved checkpoint", ["Same weights and world state before intervention"], color="#EDF7F7", accent="#007C83")
    svg.line(600, 482, 600, 509, arrow=True)
    svg.line(208, 509, 992, 509, width=1.6)
    for x in (208, 600, 992):
        svg.line(x, 509, x, 535, arrow=True)
    svg.box(38, 545, 340, 114, "Sham", ["No acquired unit mask", "Same assay seed as injured arms"])
    svg.box(430, 545, 340, 114, "Designated partition", ["Mask within the final quarter of units", "Match the scattered lesion unit count"], color="#FFF1EC", accent="#B53C36")
    svg.box(822, 545, 340, 114, "Scattered units", ["Mask across the full recurrent layer", "Same exposed identities and dose"], color="#F4F0FA", accent="#664FA3")
    svg.text(38, 708, "Three measurements keep different explanations separate", 21, weight="700")
    svg.box(38, 737, 340, 172, "Paired before / immediate probes", [
        "Same individual, slot and task trials",
        "Frozen weights; no births in the assay",
        "Measure changes in go and no-go scores",
        "Supports: acute computational effects"], color="#FFF1EC", accent="#B53C36")
    svg.box(430, 737, 340, 172, "Cue and memory interventions", [
        "Erase the cue or reset recurrent state",
        "Compare with the normal frozen assay",
        "Input/state distributions also change",
        "Supports: conditional task dependence"], color="#EDF7F7", accent="#007C83")
    svg.box(822, 737, 340, 172, "Identity retention and births", [
        "Follow original injured identities",
        "Masks persist; healthy newborns enter",
        "Population scores can rise after turnover",
        "Distinguishes replacement from repair"], color="#F4F0FA", accent="#664FA3")
    svg.lines(38, 951, [
        "Relevance: a reproducible way to test how acquired computational capacity loss alters learned function and population dynamics.",
        "Limits: birth is engineered; spontaneous BFF replication is not demonstrated. The designated partition has no validated",
        "frontal specialization. This does not establish biological brain injury, subjective trauma, consciousness, or clinical validity."], 15, gap=24)
    svg.save(path)


def assay_figure(path: Path, baseline: dict, arms: dict):
    svg = Svg(1200, 620, "Task performance before and after intervention", "Frozen-weight assays sample living agents. Follow-up samples may contain different identities; these bars do not show within-person recovery.")
    svg.text(38, 77, "Frozen-weight assays · overall population samples · follow-up may include replacement individuals", 16)
    for i, (metric, (_, _, label)) in enumerate(METRICS.items()):
        x = 70+i*386
        svg.text(x+153, 123, label, 20, anchor="middle", weight="700")
        plot = axes(svg, x, 171, 306, 274, 0, 1, xticks=1)
        # Hide numeric horizontal ticks beneath descriptive labels with white cover.
        svg.rect(x-6, 451, 320, 32, "#FFFFFF")
        base = safe_rate(baseline, metric)
        if base is not None:
            yy = plot(0, base)[1]
            svg.line(x, yy, x+306, yy, COLORS['baseline'], 2, dash=True)
            svg.text(x+153, 147, f"Baseline {base*100:.1f}%", 13, COLORS['baseline'], anchor="middle")
        for j, arm in enumerate(ARMS):
            if arm not in arms:
                continue
            center = x+51+j*103
            for k, key in enumerate(('immediate_assay', 'final_assay')):
                value = safe_rate(arms[arm].get(key, {}), metric)
                if value is None:
                    continue
                bx = center-30+k*31
                by = plot(0, value)[1]
                color = COLORS[arm] if k == 0 else '#FFFFFF'
                svg.rect(bx, by, 26, 445-by, color, COLORS[arm])
                svg.text(bx+13, by-7, f"{value*100:.1f}", 11, COLORS[arm], anchor="middle", halo=True)
            svg.lines(center, 469, ['Sham' if arm=='sham' else 'Designated' if arm=='designated_region' else 'Scattered', 'region' if arm=='designated_region' else ''], 12, anchor="middle", gap=17)
    svg.rect(39, 526, 18, 18, "#526579")
    svg.text(67, 541, "Immediate", 14)
    svg.rect(190, 526, 18, 18, "#FFFFFF", "#526579")
    svg.text(219, 541, "Follow-up", 14)
    svg.line(368, 535, 404, 535, COLORS['baseline'], 2, dash=True)
    svg.text(416, 541, "Shared baseline", 14)
    svg.lines(38, 578, ['Follow-up differences combine selection, births, deaths, altered dynamics, and population composition.', 'These are measurements from one evolved world; see paired effects for the directly injured individuals.'], 14, gap=20)
    svg.save(path)


def paired_figure(path: Path, rows: list[dict]):
    svg = Svg(1200, 600, "Immediate change in the directly injured individuals", "Each point is the mean within-individual immediate change relative to baseline. Error bars are bootstrap intervals over sampled individuals within a single evolved world; they do not represent evolutionary replication.")
    svg.text(38, 77, "Same identities + same task probes · only sampled individuals with a realized injury mask", 16)
    relevant = [r for r in rows if r['arm'] != 'sham' and r['mean_delta'] is not None]
    limit = max([.025] + [abs(r[k]) for r in relevant for k in ('bootstrap_ci95_low', 'bootstrap_ci95_high')]) * 1.32
    limit = math.ceil(limit*100)/100
    for i, (metric, (_, _, label)) in enumerate(METRICS.items()):
        x = 78+i*385
        svg.text(x+146, 125, label, 20, anchor='middle', weight='700')
        plot = axes(svg, x, 153, 282, 255, 0, 1, -limit, limit, xticks=1, yticks=4, points=True)
        svg.rect(x-6, 415, 300, 28, '#FFFFFF')
        svg.line(*plot(0, 0), *plot(1, 0), '#526579', 1.8, dash=True)
        for j, arm in enumerate(('designated_region', 'random')):
            row = next((r for r in relevant if r['arm']==arm and r['metric']==metric), None)
            xx = .26+j*.48
            svg.lines(x+xx*282, 432, ['Designated' if j==0 else 'Scattered', 'region' if j==0 else 'units'], 12, anchor='middle', gap=17)
            if not row:
                svg.text(x+xx*282, 275, 'No paired data', 12, anchor='middle')
                continue
            px, py = plot(xx, row['mean_delta'])
            _, lo = plot(xx, row['bootstrap_ci95_low'])
            _, hi = plot(xx, row['bootstrap_ci95_high'])
            svg.line(px, lo, px, hi, COLORS[arm], 2.5)
            svg.line(px-7, lo, px+7, lo, COLORS[arm], 2)
            svg.line(px-7, hi, px+7, hi, COLORS[arm], 2)
            svg.circle(px, py, 6, COLORS[arm])
            svg.text(px, min(lo, hi)-13, f"{row['mean_delta']*100:+.2f} pp", 13, COLORS[arm], anchor='middle', weight='700', halo=True)
            svg.text(px, 479, f"n = {row['paired_individuals']}", 12, anchor='middle')
    svg.lines(38, 532, ['Y axis: change in accuracy (percentage points); lower values mean worse task scores.', 'Bars: 95% percentile bootstrap over sampled individual means, within this world only.', 'This is not a confidence interval over independent runs, and no biological or clinical inference is established.'], 14, gap=22)
    svg.save(path)


def evolution_figure(path: Path, progress: list[dict], manifest: dict):
    svg = Svg(1200, 676, "Population evolution and supplied reproduction", "Warm-up shows task accuracy in consecutive progress intervals and cumulative simulator-supplied births. Follow-up branches share the same starting checkpoint.")
    warm = [r for r in progress if r['arm']=='warmup']
    xmin = warm[0]['start_decisions']/1e6 if warm else 0
    subtitle = ('Warm-up uses consecutive intervals; stitched earlier and resumed counters retain cumulative identities.'
                if xmin == 0 else f'Metrics begin at a resumed checkpoint ({xmin:g} million decisions); interval scores use counter differences.')
    svg.text(38, 76, subtitle, 16)
    xmax = max([r['decisions']/1e6 for r in warm] or [1])
    svg.text(79, 122, 'Interval task accuracy', 20, weight='700')
    plot = axes(svg, 79, 151, 477, 318, xmin, xmax, xticks=4)
    for metric, color, label in [('accuracy', '#243B53', 'Overall'), ('go_accuracy', '#007C83', 'Go'), ('nogo_accuracy', '#B53C36', 'No-go')]:
        points = [plot(r['decisions']/1e6, r[metric]) for r in warm if r.get(metric) is not None]
        svg.polyline(points, color, 3)
    svg.text(317, 520, 'Warm-up decisions (millions)', 15, anchor='middle')
    svg.text(676, 122, 'Cumulative births during warm-up', 20, weight='700')
    birthmax = max([r['births'] for r in warm] or [1]) or 1
    plot2 = axes(svg, 676, 151, 477, 318, xmin, xmax, 0, math.ceil(birthmax/1000)*1000, percent=False)
    svg.polyline([plot2(r['decisions']/1e6, r['births']) for r in warm], '#664FA3', 3)
    svg.text(915, 520, 'Warm-up decisions (millions)', 15, anchor='middle')
    for i, (label, color) in enumerate([('Overall','#243B53'), ('Go','#007C83'), ('No-go','#B53C36')]):
        xx = 79+i*150
        svg.line(xx, 558, xx+29, 558, color, 3)
        svg.text(xx+39, 564, label, 14)
    wf=manifest.get('warmup_final', {})
    svg.lines(38, 611, [f"{wf.get('decisions', 0):,} warm-up decisions · {wf.get('episodes', 0):,} scored episodes · {wf.get('births', 0):,} supplied births", 'Births copy parental weights with mutation; they are an explicit simulator rule, not emergent program replication.'], 15, gap=23)
    svg.save(path)


def cohort_figure(path: Path, traces: dict, arms: dict):
    svg = Svg(1200, 668, 'Persistent injury versus population replacement', 'Original injured individuals retain their masks. A fall in injured survivors is attrition or replacement, not healing. Healthy newborns have new identities.')
    svg.text(38, 77, 'Same injured identities tracked through follow-up; the mask has no repair or plasticity rule.', 16)
    xmax = max([(rows[-1]['decisions']-rows[0]['decisions'])/1e6 for rows in traces.values() if rows] or [1]) or 1
    svg.text(79, 122, 'Original injured individuals still alive', 20, weight='700')
    plot=axes(svg,79,151,477,317,0,xmax)
    maxbirth = max([arms.get(arm,{}).get('final',{}).get('births',0)-arms.get(arm,{}).get('immediate',{}).get('births',0) for arm in ARMS] or [1]) or 1
    svg.text(677,122,'Additional births during follow-up',20,weight='700')
    plot2=axes(svg,677,151,477,317,0,xmax,0,math.ceil(maxbirth/1000)*1000,percent=False)
    for arm in ARMS:
        rows=traces.get(arm,[])
        if not rows:
            continue
        start=rows[0]['decisions']
        initial=rows[0].get('injured_initial',0)
        if initial:
            svg.polyline([plot((r['decisions']-start)/1e6,r.get('injured_alive',0)/initial) for r in rows],COLORS[arm],3)
        b0=rows[0].get('births',0)
        svg.polyline([plot2((r['decisions']-start)/1e6,r.get('births',0)-b0) for r in rows],COLORS[arm],3)
    svg.text(317,518,'Additional decisions (millions)',15,anchor='middle')
    svg.text(915,518,'Additional decisions (millions)',15,anchor='middle')
    for i,arm in enumerate(ARMS):
        xx=79+i*300
        svg.line(xx,554,xx+29,554,COLORS[arm],3)
        svg.text(xx+40,560,LABELS[arm],14)
    svg.lines(38,609,['A population can regain task performance while injured individuals disappear and healthy offspring enter.', 'This model cannot establish neural repair: acquired masks persist, and lifetime weight learning is absent.', 'Sham has no injured cohort; its births appear in the right panel.'],14,gap=22)
    svg.save(path)


def cue_figure(path: Path, baseline: dict, controls: list[dict]):
    svg=Svg(1200,580,'Does successful behavior depend on the remembered cue?', 'Baseline assay and paired controls erase the initial cue or reset recurrent memory at the probe. Similar scores indicate weak evidence for successful cue-memory use in this run.')
    svg.text(38,78,'Same frozen checkpoint and task seed; erasure controls remove information without changing weights.',16)
    datasets=[('Normal task',baseline,'#243B53')]+[(r['label'],r['assay'],r['color']) for r in controls]
    for i,(metric,(_,_,label)) in enumerate(METRICS.items()):
        x=78+i*386
        svg.text(x+145,124,label,20,anchor='middle',weight='700')
        plot=axes(svg,x,151,282,270,0,1,xticks=1)
        if metric == 'accuracy':
            yy=plot(0,.625)[1]
            svg.line(x,yy,x+282,yy,'#526579',1.7,dash=True)
            svg.text(x+141,488,'Dashed: expected cue-blind bound (62.5%)',11,'#526579',anchor='middle')
        svg.rect(x-7,428,300,28,'#FFFFFF')
        for j,(name,record,color) in enumerate(datasets):
            value=safe_rate(record,metric)
            if value is None:continue
            xx=x+30+j*91
            yy=plot(0,value)[1]
            svg.rect(xx,yy,48,421-yy,color)
            svg.text(xx+24,yy-9,f'{value*100:.2f}%',12,color,anchor='middle',halo=True)
            svg.lines(xx+24,445,name.split(' ',1),12,anchor='middle',gap=17)
    svg.lines(38,518,['Cue and memory interventions change input/state distributions; effects are conditional task diagnostics.', 'Uniform three-action chance = 33.3%; a cue-blind policy can score up to 62.5% in expectation using the go signal.'],14,gap=22)
    svg.save(path)


def analyze(run: Path, output: Path, figures: Path, *, repeats: int=2000) -> dict:
    manifest=load_json(run/'manifest.json')
    if manifest.get('status')!='complete':
        raise ValueError('Refusing to publish analysis of an incomplete run')
    baseline=load_json(run/'baseline_assay.json')
    arms=manifest.get('arms',{})
    assay_rows=[]
    paired=[]
    for metric in METRICS:
        assay_rows.append({'arm':'baseline','time':'pre_injury','metric':metric,'accuracy':safe_rate(baseline,metric),'samples':baseline.get('samples'),'episodes':baseline.get('episodes')})
    traces={}
    warmup_rows=warmup_progress(run)
    intervals=interval_progress(warmup_rows,'warmup')
    cohort=[]
    for arm in ARMS:
        if arm not in arms:continue
        for time,key,filename in [('immediate','immediate_assay','immediate_assay.json'),('followup','final_assay','final_assay.json')]:
            a=load_json(run/arm/filename) if (run/arm/filename).exists() else arms[arm].get(key,{})
            arms[arm][key]=a
            for metric in METRICS:
                assay_rows.append({'arm':arm,'time':time,'metric':metric,'accuracy':safe_rate(a,metric),'samples':a.get('samples'),'episodes':a.get('episodes')})
        for row in paired_effects(baseline,arms[arm]['immediate_assay'],injured_only=arm!='sham',repeats=repeats):
            paired.append({'arm':arm,**row})
        trace=read_progress(run/arm/'followup.jsonl')
        traces[arm]=trace
        intervals.extend(interval_progress(trace,arm))
        immediate,final=arms[arm].get('immediate',{}),arms[arm].get('final',{})
        baseline_hash=manifest.get('warmup_final',{}).get('genome_hash')
        if baseline_hash and immediate.get('genome_hash') != baseline_hash:
            raise ValueError('The immediate intervention changed inherited weights')
        original=immediate.get('injured_initial',0)
        retained=final.get('injured_alive',0)
        # ID sets, rather than masks or slot numbers, make the retention claim auditable.
        if 'injured_ids' in immediate and 'individuals' in final:
            injured_ids=set(immediate['injured_ids'])
            retained=sum(p['id'] in injured_ids and p['alive'] for p in final['individuals'])
            if any(p['id'] in injured_ids and p['alive'] and 'injured' in p and not p['injured'] for p in final['individuals']):
                raise ValueError('An original injured survivor has lost its persistent mask')
            if retained != final.get('injured_alive',retained):
                raise ValueError('Stable-ID retention disagrees with simulator injury count')
        cohort.append({'arm':arm,'original_injured':original,'original_injured_alive':retained,'original_injured_lost':original-retained,
                       'disabled_units_immediate':immediate.get('disabled_units'),
                       'disabled_units_per_injured':immediate.get('disabled_units',0)/original if original else 0,
                       'retained_fraction':retained/original if original else None,
                       'additional_births':final.get('births',0)-immediate.get('births',0),
                       'injured_birth_replacements':final.get('injured_birth_replacements',0)-immediate.get('injured_birth_replacements',0),
                       'repair_implemented':False})
    placement_effects=[]
    if 'designated_region' in arms and 'random' in arms:
        reference=arms['designated_region']['immediate_assay']
        reference={**reference,'per_sample':[r for r in reference.get('per_sample',[]) if r.get('injured') or r.get('disabled_count',0)]}
        for row in paired_effects(reference,arms['random']['immediate_assay'],injured_only=True,repeats=repeats):
            placement_effects.append({'comparison':'scattered_minus_designated','reference_arm':'designated_region',
                                      'intervention_arm':'random',**row,'cohort':'injured_in_both_immediate_arms'})
    controls=[]
    for file in sorted(run.glob('*assay*.json')):
        a=load_json(file)
        mode=a.get('cue_mode','normal')
        if mode not in ('erased','memory-reset'):continue
        label='Cue erased' if mode=='erased' else 'Memory reset'
        controls.append({'label':label,'assay':a,'color':'#B53C36' if mode=='erased' else '#664FA3','source_file':file.name})
    control_effects=[]
    for c in controls:
        for row in paired_effects(baseline,c['assay'],injured_only=False,repeats=repeats):
            control_effects.append({'control':c['assay']['cue_mode'],**row})
    control_assays=[{'control':c['assay']['cue_mode'],'source_file':c['source_file'],
                     'samples':c['assay'].get('samples'),'trials':c['assay'].get('trials'),'episodes':c['assay'].get('episodes'),
                     **{metric:safe_rate(c['assay'],metric) for metric in METRICS}} for c in controls]
    source_files=[run/'manifest.json',*sorted(run.rglob('*assay*.json')),*sorted(run.rglob('*.jsonl'))]
    source_hashes={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    summary={'schema_version':1,'experiment':manifest.get('experiment'),'run_name':run.name,'population':manifest['population'],
             'hidden':manifest['hidden'],'seed':manifest.get('seed'),'parameters_per_agent':manifest.get('parameters_per_agent'),
             'total_parameters':manifest.get('total_parameters'),'elapsed_seconds':manifest.get('elapsed_seconds'),
             'warmup_decisions':manifest.get('warmup_final',{}).get('decisions'),
             'warmup_additional_decisions':manifest.get('warmup_additional_decisions'),
             'warmup_neural_evaluations':manifest.get('warmup_final',{}).get('neural_evaluations'),
             'warmup_scored_episodes':manifest.get('warmup_final',{}).get('episodes'),
             'warmup_metrics_start_decisions':warmup_rows[0]['decisions'] if warmup_rows else None,
             'warmup_prefix_metrics_included':(run/'prefix_warmup.jsonl').exists(),
             'followup_additional_decisions_per_arm':manifest.get('followup_additional_decisions'),
             'warmup_births':manifest.get('warmup_final',{}).get('births'),
             'intervention':{'severity_fraction_of_designated_region':manifest.get('severity'),
                             'fraction_of_living_population_exposed':manifest.get('exposure'),
                             'designated_region_units':max(1,manifest['hidden']//4),
                             'placement':'last quarter versus dose-matched scattered units',
                             'genotype_unchanged_immediately':(all(a.get('immediate',{}).get('genome_hash') == manifest['warmup_final']['genome_hash'] for a in arms.values())
                                                               if arms and manifest.get('warmup_final',{}).get('genome_hash') else None)},
             'baseline_assay':{k:v for k,v in baseline.items() if k not in ('per_sample','sample_slots')},
             'assay_summary':assay_rows,'paired_immediate_effects':paired,'cohort_retention':cohort,
             'cue_control_effects':control_effects,'cue_control_assays':control_assays,
             'paired_placement_effects':placement_effects,'source_sha256':source_hashes,
             'statistical_scope':'One evolved world. Percentile bootstrap resamples sampled individual mean changes, not task trials or independent evolutionary runs.',
             'limitations':['Birth is supplied by an energy-threshold rule; spontaneous self-replication is not demonstrated.',
                            'Learning means selection and mutation across generations; there is no gradient training or lifetime weight plasticity.',
                            'The final quarter of units has unverified functional specialization; it is not a validated frontal lobe analogue.',
                            'Acquired masks persist in the original identity. Newborn masks reset; replacement is not repair.',
                            'Population follow-up assays sample living slots and may include new identities.',
                            'Cue-erasure and memory-reset probes change input or state distributions. Their effects are conditional diagnostic measurements, not a complete mechanistic localization test.',
                            'No evidence of consciousness, felt trauma, human TBI, or clinical validity is established.']}
    output.mkdir(parents=True,exist_ok=True)
    figures.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    write_csv(output/'assay_summary.csv',assay_rows)
    write_csv(output/'paired_effects.csv',paired)
    write_csv(output/'evolution_intervals.csv',intervals)
    write_csv(output/'cohort_retention.csv',cohort)
    write_csv(output/'cue_control_effects.csv',control_effects)
    write_csv(output/'cue_control_assays.csv',control_assays)
    write_csv(output/'placement_comparison.csv',placement_effects)
    architecture_figure(figures/'architecture.svg',manifest)
    bff_figure(figures/'bff-to-neural.svg')
    relevance_figure(figures/'experimental-relevance.svg')
    assay_figure(figures/'assay-performance.svg',baseline,arms)
    paired_figure(figures/'paired-injury-effects.svg',paired)
    evolution_figure(figures/'evolution.svg',intervals,manifest)
    cohort_figure(figures/'injured-cohort.svg',traces,arms)
    if controls:
        cue_figure(figures/'cue-controls.svg',baseline,controls)
    elif (figures/'cue-controls.svg').exists():
        (figures/'cue-controls.svg').unlink()
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,default=Path('results/latest'))
    parser.add_argument('--figures',type=Path,default=Path('docs/figures'))
    parser.add_argument('--bootstrap-repeats',type=int,default=2000)
    args=parser.parse_args()
    if args.bootstrap_repeats<100:parser.error('--bootstrap-repeats must be at least 100')
    result=analyze(args.run,args.output,args.figures,repeats=args.bootstrap_repeats)
    print(f"Analyzed {result['run_name']}: {result['population']:,} agents; tables in {args.output}; figures in {args.figures}")
    print(result['statistical_scope'])


if __name__=='__main__':main()
