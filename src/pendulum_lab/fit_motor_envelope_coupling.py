"""Fit driven pendulum coupling from feedback-timed motor-envelope traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

from .plant.pendulum_geometry import PENDULUM_COM_M, PENDULUM_MASS_KG


UPSTREAM_TIP_MASS_KG = 0.00712


def _fit(parts: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    if not parts:
        raise ValueError("No usable samples: record both acceleration directions")
    x = np.vstack([item[0] for item in parts])
    y = np.concatenate([item[1] for item in parts])
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.linalg.matrix_rank(x) < 3:
        raise ValueError("Non-finite or rank-deficient coupling experiment")
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    denominator = np.sum((y - np.mean(y)) ** 2)
    if denominator <= 0:
        raise ValueError("No response variation for coupling fit")
    r2 = 1.0 - np.sum(residual**2) / denominator
    return {
        "tangential_coupling": float(beta[0]),
        "centrifugal_coupling": float(beta[1]),
        "constant_bias_rad_s2": float(beta[2]),
        "r2": float(r2),
        "samples_used": int(len(y)),
    }


def _parts_for_trace(
    path: Path, *, window_max: int, gravity_w2: float, damping: float
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    with np.load(path, allow_pickle=False) as src:
        if "complete" in src and not bool(src["complete"]):
            raise ValueError(f"Incomplete or failed recording: {path}")
        samples = np.asarray(src["samples"], dtype=np.float64)
    if (samples.ndim != 2 or samples.shape[1] != 7 or len(samples) < 5
            or not np.isfinite(samples).all() or np.any(np.diff(samples[:, 0]) <= 0)
            or np.any(samples[:, 1] != np.floor(samples[:, 1]))):
        raise ValueError(f"Invalid motor-envelope samples: {path}")
    phase_index = samples[:, 1].astype(np.int64)
    result = []
    for phase in np.unique(phase_index):
        if phase < 0:
            continue  # Center-return control is not a constant-acceleration phase.
        phase_samples = samples[phase_index == phase]
        window = min(
            window_max,
            len(phase_samples)
            if len(phase_samples) % 2 == 1
            else len(phase_samples) - 1,
        )
        if window < 5:
            continue
        dt_s = float(np.median(np.diff(phase_samples[:, 0])))
        motor_accel = savgol_filter(
            phase_samples[:, 5], window, 2, deriv=1, delta=dt_s
        )
        pendulum_accel = savgol_filter(
            phase_samples[:, 6], window, 2, deriv=1, delta=dt_s
        )
        edge = window // 2
        keep = slice(edge, len(phase_samples) - edge)
        kept = phase_samples[keep]
        phi = kept[:, 4]
        motor_vel = kept[:, 5]
        pendulum_vel = kept[:, 6]
        y = (
            pendulum_accel[keep]
            + gravity_w2 * np.sin(phi)
            + damping * pendulum_vel
        )
        x = np.column_stack(
            (
                np.cos(phi) * motor_accel[keep],
                np.sin(phi) * motor_vel**2,
                np.ones(len(phi)),
            )
        )
        result.append((float(np.sign(phase_samples[0, 2])), x, y))
    return result


def configure_parser(parser):
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--arm-length-m", type=float, default=0.065)
    parser.add_argument("--windows", type=int, nargs="+", default=[5, 7, 9, 11, 15])
    parser.add_argument("--output", type=Path, default=None)


def execute(args) -> int:
    if any(window < 5 or window % 2 == 0 for window in args.windows):
        raise ValueError("--windows values must be odd integers >= 5")
    if not np.isfinite(args.arm_length_m) or args.arm_length_m <= 0:
        raise ValueError("Arm length must be finite and positive")
    output = args.output or args.run_dir / "coupling_fit.json"
    if output.exists():
        raise FileExistsError(output)
    summary_path = args.run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary.get("complete") is False or any(not r["passed"] for r in summary.get("results", [])):
            raise ValueError("Motor-envelope session incomplete or physically rejected")

    params_doc = json.loads(args.params.read_text())
    if "omega_n_small_amp_rad_s" not in params_doc.get("pendulum", {}).get("fit", {}):
        raise ValueError("Use a fitted free-swing JSON with omega_n_small_amp_rad_s, not starter parameters")
    fit = params_doc["pendulum"]["fit"]
    derived = params_doc["pendulum"]["derived"]
    inertia = float(derived["inertia_measured_kg_m2"])
    if not np.isfinite(inertia) or inertia <= 0:
        raise ValueError("Invalid pendulum inertia")
    gravity_w2 = float(fit["omega_n_small_amp_rad_s"]) ** 2
    damping = float(derived["viscous_friction_N_m_s"]) / inertia
    mass = PENDULUM_MASS_KG
    tip = params_doc.get("run_meta", {}).get("pendulum_tip_mass", {})
    if tip.get("coin") == "New Taiwan dollar NT$10":
        mass += float(tip["mass_kg"]) - UPSTREAM_TIP_MASS_KG
    meta = params_doc.get("run_meta", {})
    mass = float(meta.get("pendulum_total_mass_kg", mass))
    com = float(meta.get("pendulum_com_m", PENDULUM_COM_M))
    if not np.isfinite([mass, com, gravity_w2, damping]).all() or min(mass, com, gravity_w2) <= 0 or damping < 0:
        raise ValueError("Invalid fitted physical parameters")
    geometric = mass * com * args.arm_length_m / inertia

    traces = sorted(args.run_dir.glob("*.npz"))
    if not traces:
        raise ValueError(f"no .npz traces found in {args.run_dir}")
    sweeps = []
    for window in args.windows:
        named_parts = {
            trace.stem: _parts_for_trace(
                trace,
                window_max=window,
                gravity_w2=gravity_w2,
                damping=damping,
            )
            for trace in traces
        }
        all_parts = [
            (x, y)
            for parts in named_parts.values()
            for _, x, y in parts
        ]
        positive = [
            (x, y)
            for parts in named_parts.values()
            for sign, x, y in parts
            if sign > 0
        ]
        negative = [
            (x, y)
            for parts in named_parts.values()
            for sign, x, y in parts
            if sign < 0
        ]
        global_fit = _fit(all_parts)
        positive_fit = _fit(positive)
        negative_fit = _fit(negative)
        for item in (global_fit, positive_fit, negative_fit):
            item["sim_coupling_gain"] = (
                -item["tangential_coupling"] / geometric
            )
        gains = [
            positive_fit["sim_coupling_gain"],
            negative_fit["sim_coupling_gain"],
        ]
        if abs(sum(gains)) < 1e-12:
            raise ValueError("Opposing or zero directional gains; inspect recordings")
        sweeps.append(
            {
                "savgol_window_max_samples": window,
                "global_fit": global_fit,
                "positive_accel_fit": positive_fit,
                "negative_accel_fit": negative_fit,
                "directional_half_range_fraction": (
                    (max(gains) - min(gains)) / (sum(gains))
                ),
            }
        )

    selected = min(sweeps, key=lambda row: abs(row["savgol_window_max_samples"] - 7))
    report = {
        "source_dir": str(args.run_dir),
        "trace_files": [trace.name for trace in traces],
        "hardware_profile": "drv8825_32",
        "fixed_terms": {
            "gravity_w2_s_2": gravity_w2,
            "viscous_damping_s_1": damping,
            "arm_length_m": args.arm_length_m,
            "unit_sim_geometric_coupling": geometric,
            "tip_mass": tip,
        },
        "selected_window_max_samples": selected["savgol_window_max_samples"],
        "selected_fit": selected,
        "window_sensitivity": sweeps,
    }
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    chosen = selected["global_fit"]
    print(
        f"coupling_gain={chosen['sim_coupling_gain']:.3f}, "
        f"R2={chosen['r2']:.3f}, samples={chosen['samples_used']}"
    )
    print(f"saved {output}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    configure_parser(parser)
    return execute(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
