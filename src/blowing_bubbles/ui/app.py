r"""BlowingBubbles Streamlit UI.

Launch with::

    streamlit run src/blowing_bubbles/ui/app.py

Three tabs: **Single** (load one spectrum, fit, view, refine, save),
**Batch** (fit a whole folder into an output folder), **Live Tune**
(adjust R / C / tau / e / L with sliders and watch the Nyquist move).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from blowing_bubbles.classes import CircuitParameters, EISData
from blowing_bubbles.scripts import fit_quality
from blowing_bubbles.ui import batch as batch_mod
from blowing_bubbles.ui import live as live_mod
from blowing_bubbles.ui import plots

st.set_page_config(
    page_title="BlowingBubbles",
    page_icon=str(Path(__file__).with_name("assets") / "icon-256.png"),
    layout="wide",
)

FREQ_GRID = np.logspace(-2, 5, 300)


def _fit_controls(key_prefix: str) -> dict:
    with st.expander("Fit settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            rc_groups = st.number_input(
                "RC branches (max)", 1, 10, 3, key=f"{key_prefix}_rc"
            )
            n_restarts = st.number_input(
                "Restarts", 1, 1000, 50, key=f"{key_prefix}_restarts"
            )
            seed = st.number_input("Seed", 0, 10**9, 42, key=f"{key_prefix}_seed")
        with col2:
            weighting = st.selectbox(
                "Weighting", ["modulus", "unit"], key=f"{key_prefix}_weight"
            )
            n_jobs = st.selectbox(
                "Worker processes", [1, -1], key=f"{key_prefix}_njobs"
            )
            refine = st.checkbox(
                "Refine with quick_fit", value=True, key=f"{key_prefix}_refine"
            )
        with col3:
            rs_b = st.text_input("Rs bounds (min,max)", "", key=f"{key_prefix}_rsb")
            r_b = st.text_input("Branch R bounds", "0.0,100.0", key=f"{key_prefix}_rb")
            e_b = st.text_input("Exponent bounds", "0.85,1.0", key=f"{key_prefix}_eb")
    return {
        "n_restarts": int(n_restarts),
        "rc_groups": int(rc_groups),
        "seed": int(seed),
        "weighting": weighting,
        "n_jobs": n_jobs,
        "refine": refine,
        "rs_bounds": _parse_bounds(rs_b),
        "r_bounds": _parse_bounds(r_b),
        "e_bounds": _parse_bounds(e_b),
    }


def _parse_bounds(text: str) -> tuple[float, float] | None:
    text = text.strip().strip("()[]")
    if not text:
        return None
    try:
        lo, hi = (float(v) for v in text.split(",")[:2])
    except ValueError:
        return None
    return (lo, hi)


def _fit_kwargs(c: dict) -> dict:
    return {
        "n_restarts": c["n_restarts"],
        "rc_groups": c["rc_groups"],
        "seed": c["seed"],
        "weighting": c["weighting"],
        "n_jobs": c["n_jobs"],
        "resistance_source_bounds": c["rs_bounds"],
        "resistance_branch_bounds": c["r_bounds"],
        "exponent_bounds": c["e_bounds"],
    }


def _quick_kwargs(c: dict) -> dict:
    return {
        "complex_fit": True,
        "seed": c["seed"],
        "resistance_source_bounds": c["rs_bounds"],
        "resistance_branch_bounds": c["r_bounds"],
        "exponent_bounds": c["e_bounds"],
    }


def _load_data(uploaded) -> EISData:
    frame = pd.read_csv(uploaded)
    return EISData(frame)


def _parameter_table(
    params: CircuitParameters, uncertainties: dict | None
) -> pd.DataFrame:
    n = len(params)
    rows = [
        ("Rs (Ohm)", params.resistance_source),
        ("L (H)", params.inductance),
        ("n_L", params.inductance_exponent),
    ]
    for i in range(n):
        rows.append((f"R_{i + 1} (Ohm)", float(params.resistances[i])))
    for i in range(n):
        rows.append((f"C_{i + 1} (F)", float(params.capacitances[i])))
    for i in range(n):
        rows.append((f"tau_{i + 1} (s)", live_mod.branch_tau(params, i)))
    for i in range(n):
        rows.append((f"e_{i + 1}", float(params.exponents[i])))
    table = pd.DataFrame(rows, columns=["parameter", "value"])
    if uncertainties is not None:
        errs = [
            uncertainties["resistance_source"],
            uncertainties["inductance"],
            uncertainties["inductance_exponent"],
        ]
        errs += list(uncertainties["resistances"]) + list(uncertainties["capacitances"])
        errs += [np.nan] * n + list(uncertainties["exponents"])
        table["std_err"] = errs
    return table


def _result_panel(data: EISData, label: str, result) -> None:
    st.markdown(
        f"**{label}** — success: `{result.success}`, "
        f"cost: `{result.cost:.3e}`, nfev: `{result.nfev}`, "
        f"branches: `{result.n_groups}`"
    )
    try:
        quality = fit_quality(data, result.parameters)
        st.caption(
            f"quality: {quality.get('verdict', quality)} "
            f"(rel. RMS {quality.get('relative_rms', float('nan')):.4f})"
        )
    except Exception:
        pass
    st.dataframe(_parameter_table(result.parameters, result.parameter_uncertainties()))


def single_tab() -> None:
    st.header("Single spectrum")
    uploaded = st.file_uploader("EIS CSV", type=["csv"])
    demo = st.checkbox("No file? Use synthetic demo spectrum")
    data = st.session_state.get("single_data")
    if uploaded is not None:
        data = _load_data(uploaded)
        st.session_state["single_data"] = data
    elif demo and data is None:
        buf = io.BytesIO()
        batch_mod.make_synthetic_csv(buf)
        buf.seek(0)
        data = _load_data(buf)
        st.session_state["single_data"] = data
    if data is None:
        st.info("Upload a CSV or enable the demo spectrum.")
        return

    st.caption(
        f"{len(data.Freq)} points, {data.Freq.min():.3g}–{data.Freq.max():.3g} Hz"
    )
    controls = _fit_controls("single")
    col_run, col_save = st.columns([1, 1])
    run = col_run.button("▶ Run fit", type="primary")
    if run:
        with st.spinner("Fitting..."):
            result = data.fit(**_fit_kwargs(controls))
            if controls["refine"]:
                result = data.quick_fit(
                    initial_parameters=result.parameters, **_quick_kwargs(controls)
                )
        fits = st.session_state.setdefault("single_fits", [])
        fits.append((f"fit {len(fits) + 1}", result))

    fits = st.session_state.get("single_fits", [])
    left, right = st.columns(2)
    with left:
        st.pyplot(plots.nyquist_figure(data, fits))
    with right:
        if fits:
            _result_panel(data, fits[-1][0], fits[-1][1])
            st.pyplot(plots.residual_figure(data, fits[-1][1]))
        else:
            st.info("Run a fit to see parameters and residuals.")
    with col_save:
        if fits:
            params = fits[-1][1].parameters
            st.download_button(
                "💾 Download parameters JSON",
                data=pd.Series(params.as_dict()).to_json(),
                file_name="parameters.json",
            )


def _folder_row(label: str, key: str) -> str:
    """Text field + native 'Browse' button (desktop app) for a folder path."""
    from blowing_bubbles.ui import folder_picker

    cols = st.columns([4, 1])
    value = cols[0].text_input(label, key=key)
    if cols[1].button("📂 Browse", key=f"{key}_browse"):
        picked = folder_picker.pick_folder()
        if picked:
            st.session_state[key] = picked
            st.rerun()
    return st.session_state.get(key, value)


def batch_tab() -> None:
    st.header("Batch: fit a folder")
    in_dir = _folder_row("Input folder (CSV spectra)", "batch_in")
    out_dir = _folder_row("Output folder", "batch_out")
    area_text = st.text_input(
        "Normalize by electrode area (cm^2, empty = none)", "", key="batch_area"
    )
    try:
        area = float(area_text) if area_text.strip() else None
    except ValueError:
        st.error("Area must be a number (e.g. 1.0 for 1 cm^2).")
        area = None
    controls = _fit_controls("batch")
    run = st.button("▶ Run batch", type="primary")
    if run:
        if not in_dir or not out_dir:
            st.error("Set both input and output folders.")
            return
        progress = st.progress(0.0, text="Starting...")

        def cb(i: int, n: int, name: str) -> None:
            progress.progress(i / max(n, 1), text=f"file {i}/{n} — {name}")

        with st.spinner("Running batch..."):
            report = batch_mod.run_batch(
                in_dir,
                out_dir,
                fit_kwargs=_fit_kwargs(controls),
                quick_fit_kwargs=_quick_kwargs(controls)
                if controls["refine"]
                else None,
                area=area,
                progress=cb,
            )
        progress.progress(1.0, text="done")
        st.session_state["batch_report"] = report
        st.session_state["batch_out_dir"] = out_dir

    report = st.session_state.get("batch_report")
    if report is None and out_dir:
        report_path = Path(out_dir) / "batch_report.csv"
        if report_path.exists():
            report = batch_mod.load_report(out_dir)
            st.session_state["batch_report"] = report
            st.session_state["batch_out_dir"] = out_dir
    if report is None:
        st.info("Run a batch or point the output folder at a previous run.")
        return
    st.dataframe(report)
    out = st.session_state["batch_out_dir"]
    summary = Path(out) / "batch_summary.png"
    if summary.exists():
        st.image(str(summary))
    ok_rows = (
        report[report.get("success", False).astype(bool)] if len(report) else report
    )
    if len(ok_rows):
        choice = st.selectbox(
            "Inspect a fitted spectrum", ok_rows["file"].tolist(), key="batch_pick"
        )
        if choice:
            stem = Path(choice).stem
            params = batch_mod.load_fit(out, stem)
            data = EISData.from_csv(str(Path(in_dir or ".") / choice), area=area)
            left, right = st.columns(2)
            with left:
                st.pyplot(plots.nyquist_figure(data, [("fit", params)]))
            with right:
                st.dataframe(_parameter_table(params, None))


def live_tab() -> None:
    st.header("Live tune")
    data = st.session_state.get("single_data")
    if data is not None:
        st.caption(f"Overlaying measured spectrum ({len(data.Freq)} points).")
    params = st.session_state.get("live_params")
    if params is None:
        source = st.session_state.get("single_fits", [])
        params = (
            source[-1][1].parameters.copy()
            if source
            else live_mod.default_parameters(3)
        )
        st.session_state["live_params"] = params

    col_sliders, col_plot = st.columns([1, 1])
    with col_sliders:
        rs = st.slider(
            "Rs (Ohm)", 0.1, 100.0, float(params.resistance_source), key="lv_rs"
        )
        ind = st.slider(
            "L (H)", 1e-9, 1e-4, float(params.inductance), format="%.1e", key="lv_l"
        )
        n_l = st.slider("n_L", 0.5, 1.0, float(params.inductance_exponent), key="lv_nl")
        params.resistance_source = rs
        params.inductance = ind
        params.inductance_exponent = n_l
        n = len(params)
        n_new = st.number_input("Branches", 1, 8, n, key="lv_n")
        if n_new != n:
            while len(params) > n_new:
                params.pop_branch()
            while len(params) < n_new:
                params.add_branch(20.0, 1e-3, 0.95)
            st.rerun()
        for i in range(len(params)):
            with st.expander(f"Branch {i + 1}", expanded=(i == 0)):
                r = st.slider(
                    f"R_{i + 1} (Ohm)",
                    0.1,
                    500.0,
                    float(params.resistances[i]),
                    key=f"lv_r{i}",
                )
                tau = st.slider(
                    f"tau_{i + 1} (s)",
                    1e-6,
                    1e2,
                    live_mod.branch_tau(params, i),
                    format="%.2e",
                    key=f"lv_t{i}",
                )
                e = st.slider(
                    f"e_{i + 1}", 0.5, 1.0, float(params.exponents[i]), key=f"lv_e{i}"
                )
                params.resistances[i] = r
                params.exponents[i] = e
                live_mod.set_branch_tau(params, i, tau)
                st.caption(
                    f"C = {params.capacitances[i]:.3e} F, "
                    f"f_peak = {live_mod.branch_peak_frequency(params, i):.3g} Hz"
                )
        if data is not None:
            st.metric(
                "live relative RMS residual",
                f"{live_mod.live_residual_rms(data, params):.4f}",
            )
        if st.button("⚡ Use as quick_fit initial guess"):
            st.session_state["quick_initial"] = params.copy()
            st.success("Stored — use it from the Single tab refinement.")
    with col_plot:
        fits = [("live model", params)]
        st.pyplot(plots.nyquist_figure(data, fits))


def analysis_tab() -> None:
    st.header("Analysis of fitted results")
    from blowing_bubbles.ui import analysis as analysis_mod
    from blowing_bubbles.ui import analysis_plots

    out = st.session_state.get("batch_out_dir") or st.text_input(
        "Batch output folder", key="analysis_out"
    )
    in_dir = st.session_state.get("batch_in") or st.text_input(
        "Batch input folder", key="analysis_in"
    )
    if not out or not Path(out).exists():
        st.info("Run a batch first, or point at an existing output folder.")
        return
    report_path = Path(out) / "batch_report.csv"
    if not report_path.exists():
        st.error(f"No batch_report.csv in {out}.")
        return
    report = batch_mod.load_report(out)
    table = analysis_mod.analysis_table(out, report)
    st.subheader("Resistance breakdown per file")
    st.dataframe(table)
    st.pyplot(analysis_plots.resistance_overview_figure(table))

    ok = report[report.get("success", False).astype(bool)]["file"].tolist()
    if ok and in_dir:
        st.subheader("Difference spectra (ΔZ vs reference)")
        reference = st.selectbox("Reference file", ok, key="analysis_ref")
        picks = st.multiselect(
            "Files to compare", ok, default=ok[:3], key="analysis_pick"
        )
        mode = st.radio(
            "Plot", ["-Im(ΔZ)", "|ΔZ|"], horizontal=True, key="analysis_mode"
        )
        if reference and picks:
            diff = analysis_mod.difference_spectra(in_dir, out, picks, reference)
            st.pyplot(
                analysis_plots.difference_figure(
                    diff, reference, mode="imag" if mode == "-Im(ΔZ)" else "mag"
                )
            )


def main() -> None:
    st.title("BlowingBubbles — EIS circuit fitting")
    tab_single, tab_batch, tab_live, tab_analysis = st.tabs(
        ["Single", "Batch", "Live Tune", "Analysis"]
    )
    with tab_single:
        single_tab()
    with tab_batch:
        batch_tab()
    with tab_live:
        live_tab()
    with tab_analysis:
        analysis_tab()


main()
