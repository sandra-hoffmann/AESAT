"""One run in depth: progress, generations, translation, per-query results,
terms, lineage, timing and the prompts of the run."""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common
import loader
import metrics

name = common.pick_run()
if name is None:
    common.no_data("No run folders found.")
    st.stop()

TRANSLATION_ORDER = ["accepted", "accepted after retry", "fidelity mismatch", "parser error",
                     "no expression", "refused", "failed (no details)"]


# --------------------------------------------------------------------------- header

def header(run, cands):
    cfg = run.config
    st.title(run.name)
    st.markdown(
        f"{common.status_text(run.status)} · started {cfg.get('started', '-')} · "
        f"task `{cfg.get('task', '-')}` · benchmark `{str(cfg.get('benchmark', '-')).split('/')[-1]}` · "
        f"models {loader.models_label(cfg) if cfg else '-'}")

    s = metrics.run_summary(run, cands)
    cols = st.columns(6)
    cols[0].metric("Best score", "-" if s["best score"] is None else f"{s['best score']:.4f}",
                   help=common.SCORE_HELP)
    cols[1].metric("Best candidate", "-" if s["best id"] is None else f"#{s['best id']}")
    cols[2].metric("Generations", s["generations"])
    cols[3].metric("Formulas measured", s["measured"], help="Distinct formulas; repeats not counted")
    cols[4].metric("Repeats", s["repeats"], help="Candidates whose formula was already measured")
    cols[5].metric("Translation failed", s["translation failed"])


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.0f}" if v.is_integer() else f"{v:.3f}".rstrip("0")
    return v


def live_panel(run):
    if run.status != "running":
        return
    act = metrics.current_activity(run)
    with st.container(border=True):
        if not act:
            st.markdown(f"Running · last write {common.ago(run.last_activity)}. "
                        "This run has no progress.jsonl, so no details.")
            return
        doing = (f"measuring candidate **#{act['measuring']}**" if "measuring" in act
                 else f"last event **{act['last_event'].replace('_', ' ')}**")
        st.markdown(f"Generation **{act['generation']}** of {run.config.get('generations', '?')}: "
                    f"{doing}, {common.ago(act['since'])} ({common.local_time(act['since'])})")
        p = run.progress
        ideas = p[(p["event"] == "idea") & (p["generation"] == act["generation"])]
        if not ideas.empty:
            st.caption("Idea of this generation")
            st.markdown(f"> {ideas.iloc[-1]['idea']}")
        recent = p.tail(8).copy()
        recent["time"] = recent["ts"].map(common.local_time)
        detail = recent.drop(columns=["ts", "time", "generation", "event"], errors="ignore")
        recent["detail"] = detail.apply(
            lambda r: ", ".join(f"{k}={_fmt(v)}" for k, v in r.items()
                                if k != "idea" and v is not None and not (isinstance(v, float) and np.isnan(v))),
            axis=1)
        st.dataframe(recent[["time", "generation", "event", "detail"]].iloc[::-1], hide_index=True,
                     width="stretch")


# --------------------------------------------------------------------------- progress

def tab_progress(run, cands):
    has_q = "solved_score" in cands and cands["solved_score"].notna().any()
    c1, c2 = st.columns([2, 1])
    measure = c1.radio("Score", ["score", "solved-only score"] if has_q else ["score"], horizontal=True,
                       help=common.SOLVED_HELP if has_q else common.SCORE_HELP)
    show_repeats = c2.toggle("Show repeated formulas", value=False)
    col = "score" if measure == "score" else "solved_score"

    d = cands[cands["measured"] & (cands["origin"] != "default")].copy()
    if not show_repeats:
        d = d[~d["is_repeat"]]
    if d.empty:
        common.no_data("No measured candidate yet.")
        return
    # spread candidates of one generation horizontally so they do not overlap
    d["rank"] = d.groupby("generation").cumcount()
    d["n"] = d.groupby("generation")["id"].transform("size")
    d["x"] = d["generation"] + (d["rank"] + 0.5) / d["n"] * 0.7 - 0.35

    c = common.colors()
    fig = go.Figure()
    fig.add_hline(y=1, line=dict(color=c["muted"], width=1, dash="dot"))
    for origin in common.ORIGIN_ORDER:
        o_all = d[d["origin"] == origin]
        if o_all.empty:
            continue
        failed_mask = o_all["errors"].fillna(0) + o_all["wrong_answers"].fillna(0) > 0
        for failed in (False, True):
            o = o_all[failed_mask == failed]
            if o.empty:
                continue
            fig.add_trace(go.Scatter(
                x=o["x"], y=o[col], mode="markers", name=origin, legendgroup=origin,
                showlegend=not failed or not (~failed_mask).any(),
                marker=dict(size=9, color=common.origin_color(origin),
                            symbol="circle-open" if failed else "circle",
                            line=dict(width=2, color=common.origin_color(origin))),
                customdata=np.stack([o["id"], o["generation"], o["score"],
                                     o["solved_score"] if has_q else o["score"],
                                     o["errors"].astype(str), o["wrong_answers"].astype(str),
                                     o["formula"].map(common.wrap)], axis=-1),
                hovertemplate=("<b>#%{customdata[0]}</b> gen %{customdata[1]}<br>score %{customdata[2]:.4f}"
                               + ("<br>solved-only %{customdata[3]:.4f}" if has_q else "")
                               + "<br>timeouts/errors %{customdata[4]}, wrong %{customdata[5]}"
                                 "<br>%{customdata[6]}<extra></extra>")))
    curve = metrics.best_curve(cands)
    fig.add_trace(go.Scatter(x=curve["generation"], y=curve["best_score"], mode="lines", name="best so far",
                             line=dict(color=c["ink"], width=2, shape="hv"),
                             hovertemplate="gen %{x}: best %{y:.4f}<extra></extra>"))
    common.log_axis(fig, pd.concat([d[col], curve["best_score"], pd.Series([1.0])]))
    fig.update_yaxes(title=measure)
    fig.update_xaxes(title="generation", dtick=1 if d["generation"].max() <= 30 else None)
    common.show(common.style(fig, height=430), key="progress")
    st.caption("Open circles: at least one timeout, solver error or wrong answer. "
               "Dotted line: the default heuristic (score 1).")
    if cands["pending"].fillna(False).any():
        st.caption("Candidates of the running generation come from progress.jsonl; their formulas "
                   "appear when the generation file is written.")

    st.subheader("Best distinct formulas")
    top = cands[cands["measured"] & ~cands["is_repeat"]].sort_values("score", ascending=False).head(15)
    show_cols = [k for k in ["id", "generation", "origin", "score", "solved_score", "wins", "losses",
                             "timeouts", "wrong_answers", "same_results_as", "formula"] if k in top]
    st.dataframe(top[show_cols], hide_index=True, width="stretch",
                 column_config={"score": st.column_config.NumberColumn(format="%.4f"),
                                "solved_score": st.column_config.NumberColumn("solved-only", format="%.4f"),
                                "same_results_as": st.column_config.NumberColumn(
                                    "same results as", help="Different formula, identical results")})


# --------------------------------------------------------------------------- generations

def tab_generations(run, cands, atts):
    gens = sorted(g for g in cands["generation"].dropna().unique())
    if not gens:
        common.no_data("No generation written yet.")
        return
    g = st.select_slider("Generation", options=gens, value=gens[-1])
    info = run.generations.get(g, {})
    left, right = st.columns(2)
    with left:
        st.markdown("**Idea**")
        st.markdown(info.get("idea") or "_not recorded_")
    with right:
        st.markdown("**Analysis** (input for the next idea)")
        st.markdown(info.get("analysis") or "_none_")

    gc = cands[cands["generation"] == g]
    st.markdown("**Candidates**")
    cols = [k for k in ["id", "origin", "parents", "score", "errors", "wrong_answers", "translation",
                        "n_attempts", "duplicate_of", "repeat_of", "same_results_as", "crossover_gain",
                        "formula"] if k in gc]
    st.dataframe(gc[cols], hide_index=True, width="stretch",
                 column_config={"score": st.column_config.NumberColumn(format="%.4f"),
                                "crossover_gain": st.column_config.NumberColumn(
                                    "gain vs best parent", format="%+.4f"),
                                "duplicate_of": st.column_config.NumberColumn(
                                    help="Detected by the loop, not measured again"),
                                "repeat_of": st.column_config.NumberColumn(
                                    help="Same formula as this earlier candidate")})

    rounds = info.get("crossover_rounds") or []
    if rounds:
        st.markdown("**Crossover rounds**")
        score = dict(zip(cands["id"], cands["score"]))
        for r in rounds:
            pool = list(dict.fromkeys(r["pool_before"] + [h for h in r["hybrids"] if h in r["fitness"]]))
            rows = [{"id": i, "new hybrid": i in r["hybrids"], "score": score.get(i),
                     "innovation": r["innovation"].get(i), "fitness": r["fitness"].get(i),
                     "kept": i in r["pool_after"]} for i in pool]
            dropped = [h for h in r["hybrids"] if h not in r["fitness"]]
            with st.expander(f"Round {r['round']}: pool {len(r['pool_before'])} → {len(r['pool_after'])}"):
                st.dataframe(pd.DataFrame(rows).sort_values("fitness", ascending=False, na_position="last"),
                             hide_index=True, width="stretch",
                             column_config={"score": st.column_config.NumberColumn(format="%.4f"),
                                            "fitness": st.column_config.NumberColumn(format="%.3f")})
                missing = [i for i in pool if i not in r["innovation"]]
                notes = []
                if dropped:
                    notes.append(f"Not in the pool (invalid, unmeasured or repeated formula): {dropped}")
                if missing:
                    notes.append(f"No innovation rating parsed, default used: {missing}")
                for n in notes:
                    st.caption(n)
    elif not info.get("has_rounds", False) and g > 0:
        st.caption("This run does not record crossover rounds.")

    ga = atts[atts["generation"] == g]
    interesting = ga.groupby("candidate_id").filter(
        lambda x: len(x) > 1 or x["error"].notna().any() or (x["fidelity"] == "MISMATCH").any())
    if not interesting.empty:
        st.markdown("**Translation attempts with errors**")
        for cid, a in interesting.groupby("candidate_id"):
            with st.expander(f"#{cid}: {len(a)} attempt(s)"):
                for _, row in a.iterrows():
                    st.markdown(f"Attempt {row['attempt'] + 1}: `{row['formula'] or '-'}`")
                    if row["error"]:
                        st.markdown(f"Error: {row['error']}")
                    if row["fidelity"]:
                        st.markdown(f"Fidelity: {row['fidelity']}"
                                    + (f": {row['fidelity_reason']}" if row["fidelity_reason"] else ""))
                    if row["response"]:
                        st.code(row["response"], language=None, wrap_lines=True)


# --------------------------------------------------------------------------- translation

def tab_translation(run, cands, atts):
    t = metrics.translation_by_generation(cands)
    if t.empty:
        common.no_data("No variants yet.")
        return
    c = common.colors()
    order = [s for s in TRANSLATION_ORDER if s in set(t["translation"])]
    order += [s for s in t["translation"].unique() if s not in order]
    fig = go.Figure()
    for i, status in enumerate(order):
        s = t[t["translation"] == status]
        fig.add_trace(go.Bar(x=s["generation"], y=s["count"], name=status,
                             marker=dict(color=c["series"][i % 8], line=dict(width=1, color="rgba(0,0,0,0)")),
                             hovertemplate=f"gen %{{x}}: %{{y}} {status}<extra></extra>"))
    fig.update_layout(barmode="stack", bargap=0.25)
    fig.update_xaxes(title="generation", dtick=1 if t["generation"].max() <= 30 else None)
    fig.update_yaxes(title="variants")
    common.show(common.style(fig, height=360), key="translation")
    st.caption("Variants per generation by outcome of the translation. A generation whose "
               "first response had no formula shows as one variant.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Errors over all attempts**")
        e = atts[atts["error_category"].notna()].groupby("error_category").size().reset_index(name="attempts")
        st.dataframe(e.sort_values("attempts", ascending=False), hide_index=True, width="stretch")
        fid = atts["fidelity"].value_counts().reset_index()
        if not fid.empty:
            st.markdown("**Fidelity verdicts**")
            st.dataframe(fid, hide_index=True, width="stretch")
    with right:
        refusals = atts[atts["error_category"] == "refused"]
        st.markdown("**Refusals** (quantities without a matching term)")
        if refusals.empty:
            st.caption("None.")
        else:
            st.dataframe(refusals[["generation", "candidate_id", "error"]], hide_index=True, width="stretch")
        mism = atts[atts["fidelity"] == "MISMATCH"]
        if not mism.empty:
            st.markdown("**Fidelity mismatches**")
            st.dataframe(mism[["generation", "candidate_id", "fidelity_reason"]], hide_index=True,
                         width="stretch")


# --------------------------------------------------------------------------- queries

def tab_queries(run, cands):
    q = metrics.query_outcomes(run)
    if q.empty:
        common.no_data("This run has no per-query results (queries.jsonl). They are recorded "
                       "from the loop version of 6 November on.")
        return
    c = common.colors()
    timeout = run.config.get("timeout_s", 30)

    measured = cands[cands["id"].isin(q["candidate_id"].unique())].sort_values("score", ascending=False)
    n_max = len(measured)
    if n_max == 0:
        common.no_data("No candidate with per-query results yet.")
        return
    n = st.slider("Candidates shown (best first)", 1, n_max, min(25, n_max)) if n_max > 1 else 1
    ids = measured["id"].head(n).tolist()
    base_order = (q.drop_duplicates("query").sort_values("baseline_backtracks")["query"].tolist())
    sub = q[q["candidate_id"].isin(ids)]
    z = sub.pivot(index="candidate_id", columns="query", values="log10_ratio").reindex(index=ids, columns=base_order)
    out = sub.pivot(index="candidate_id", columns="query", values="outcome").reindex(index=ids, columns=base_order)
    z_clip = z.clip(-2, 2)
    xs, ys = list(range(len(base_order))), list(range(len(ids)))
    fig = go.Figure(go.Heatmap(
        z=z_clip.values, x=xs, y=ys,
        customdata=np.dstack([out.values, z.values, np.tile(np.array(base_order), (len(ids), 1)),
                              np.tile(np.array([f"#{i}" for i in ids])[:, None], (1, len(base_order)))]),
        zmin=-2, zmax=2, colorscale=[[0, c["div_neg"]], [0.5, c["div_mid"]], [1, c["div_pos"]]],
        xgap=1, ygap=1,
        colorbar=dict(title="ratio", tickvals=[-2, -1, 0, 1, 2],
                      ticktext=["≤ 0.01", "0.1", "1", "10", "≥ 100"], len=0.8),
        hovertemplate="%{customdata[3]} on %{customdata[2]}<br>%{customdata[0]}, log10 ratio "
                      "%{customdata[1]:.2f}<extra></extra>"))
    to = sub[sub["outcome"].isin(["timeout", "solver error", "wrong answer"])]
    if not to.empty:
        fig.add_trace(go.Scatter(
            x=[base_order.index(x) for x in to["query"]], y=[ids.index(i) for i in to["candidate_id"]],
            mode="markers", marker=dict(symbol="x-thin", size=7, line=dict(width=1.5, color=c["ink"])),
            name="timeout / error / wrong", hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed", title=None, tickvals=ys, ticktext=[f"#{i}" for i in ids],
                     showgrid=False)
    step = max(1, len(xs) // 22)
    fig.update_xaxes(title="query (sorted by baseline backtracks, easiest left)", showgrid=False,
                     tickvals=xs[::step], ticktext=[f"q{i + 1}" for i in xs[::step]])
    common.show(common.style(fig, height=max(260, 22 * len(ids) + 110)), key="heatmap")
    st.caption("Blue: fewer backtracks than the default heuristic; red: more. Crosses: timeout, solver "
               "error or wrong answer.")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Per query** (over the candidates shown)")
        per = sub.groupby("query").agg(
            baseline_backtracks=("baseline_backtracks", "first"),
            wins=("outcome", lambda s: int((s == "win").sum())),
            losses=("outcome", lambda s: int((s == "loss").sum())),
            timeouts=("outcome", lambda s: int(s.isin(["timeout", "solver error"]).sum())),
            near_timeout=("near_timeout", "sum"),
            best_ratio=("ratio", "max"),
            median_runtime_s=("runtime_s", "median")).reset_index()
        per.insert(0, "q", per["query"].map(lambda x: f"q{base_order.index(x) + 1}"))
        st.dataframe(per.sort_values("timeouts", ascending=False), hide_index=True, width="stretch",
                     column_config={"best_ratio": st.column_config.NumberColumn(format="%.3f"),
                                    "median_runtime_s": st.column_config.NumberColumn(format="%.2f")})
    with right:
        st.markdown("**Runtime of solved queries**")
        solved = sub[sub["solved"]]
        fig = go.Figure(go.Histogram(x=solved["runtime_s"], nbinsx=30, marker=dict(color=c["series"][0]),
                                     hovertemplate="%{x} s: %{y}<extra></extra>"))
        fig.add_vline(x=timeout, line=dict(color=c["critical"], width=2))
        fig.add_vline(x=metrics.NEAR_TIMEOUT * timeout, line=dict(color=c["muted"], width=1, dash="dot"))
        fig.update_xaxes(title="seconds")
        fig.update_yaxes(title="queries")
        common.show(common.style(fig, height=300, legend=False), key="runtime")
        st.caption(f"Solid line: timeout ({timeout} s). Dotted: {metrics.NEAR_TIMEOUT:.0%} of it. "
                   "Queries close to the line can flip between solved and timed out, which makes "
                   "scores of the same formula differ between runs.")

    st.markdown("**One candidate against the baseline**")
    cid = st.selectbox("Candidate", ids, format_func=lambda i: f"#{i}  score "
                       f"{cands.set_index('id').loc[i, 'score']:.4f}")
    one = q[q["candidate_id"] == cid].set_index("query").reindex(base_order).reset_index()
    colors_ = one["outcome"].map({"win": c["series"][0], "loss": c["div_neg"], "tie": c["muted"]}).fillna(c["ink"])
    fig = go.Figure(go.Bar(
        x=[f"q{i + 1}" for i in range(len(one))], y=one["log10_ratio"], marker=dict(color=colors_),
        customdata=np.stack([one["query"], one["outcome"], one["backtracks"].fillna(-1),
                             one["baseline_backtracks"], one["runtime_s"]], axis=-1),
        hovertemplate=("%{customdata[0]}<br>%{customdata[1]}<br>backtracks %{customdata[2]:,.0f} "
                       "(baseline %{customdata[3]:,.0f})<br>%{customdata[4]:.1f} s<extra></extra>")))
    fig.update_yaxes(title="log10 ratio (> 0 better than default)")
    fig.update_xaxes(title="query (easiest left)")
    common.show(common.style(fig, height=300, legend=False), key="one_candidate")
    row = cands.set_index("id").loc[cid]
    st.caption(f"Wins {row.get('wins', '-')}, losses {row.get('losses', '-')}, ties {row.get('ties', '-')}, "
               f"timeouts {row.get('timeouts', '-')}. Timeouts and errors are shown at −2 (ratio 0.01).")


# --------------------------------------------------------------------------- terms

def tab_terms(run, cands):
    u = metrics.term_usage(cands)
    if u.empty:
        common.no_data("No measured candidate yet.")
        return
    c = common.colors()
    piv = u.pivot(index="term", columns="generation", values="count").reindex(
        [t for t in metrics.TERMS if t in set(u["term"])]).fillna(0)
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=piv.columns, y=piv.index, xgap=2, ygap=2,
        colorscale=[[i / (len(c["seq"]) - 1), col] for i, col in enumerate(c["seq"])],
        colorbar=dict(title="candidates"),
        hovertemplate="gen %{x}: %{z:.0f} candidates use %{y}<extra></extra>"))
    fig.update_xaxes(title="generation", dtick=1 if len(piv.columns) <= 30 else None, showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    common.show(common.style(fig, height=60 + 30 * len(piv), legend=False), key="terms")

    eff = metrics.term_effect(cands)
    if not eff.empty:
        st.markdown("**Median score with and without each term**")
        fig = go.Figure()
        for _, r in eff.iterrows():
            fig.add_trace(go.Scatter(x=[r["without"], r["with"]], y=[r["term"]] * 2, mode="lines",
                                     line=dict(color=c["axis"], width=2), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=eff["without"], y=eff["term"], mode="markers", name="without",
                                 marker=dict(size=10, color=c["muted"]),
                                 customdata=eff["n_without"],
                                 hovertemplate="%{y} absent: median %{x:.3f} (n=%{customdata})<extra></extra>"))
        fig.add_trace(go.Scatter(x=eff["with"], y=eff["term"], mode="markers", name="with",
                                 marker=dict(size=10, color=c["series"][0]),
                                 customdata=eff["n_with"],
                                 hovertemplate="%{y} used: median %{x:.3f} (n=%{customdata})<extra></extra>"))
        fig.update_xaxes(type="log", title="median score of distinct measured formulas")
        fig.update_yaxes(autorange="reversed")
        common.show(common.style(fig, height=60 + 34 * len(eff)), key="term_effect")
        st.caption("Descriptive only: terms appear together, so a difference is not the effect of one term.")


# --------------------------------------------------------------------------- lineage

def tab_lineage(run, cands):
    m = cands[cands["measured"]]
    if m.empty:
        common.no_data("No measured candidate yet.")
        return
    best = int(m.loc[m["score"].idxmax(), "id"])
    ids = m.sort_values("score", ascending=False)["id"].tolist()
    cid = st.selectbox("Ancestry of", ids, index=ids.index(best),
                       format_func=lambda i: f"#{i}  score {m.set_index('id').loc[i, 'score']:.4f}")
    st.graphviz_chart(metrics.lineage_dot(cands, cid, dark=common.mode() == "dark"), width="content")
    st.caption("Blue: variant, orange: crossover, grey: default heuristic. Arrows from parent to child.")

    x = cands[(cands["origin"] == "crossover") & cands["crossover_gain"].notna() & ~cands["is_repeat"]]
    if not x.empty:
        cols = st.columns(3)
        cols[0].metric("Crossovers measured", len(x))
        cols[1].metric("Better than best parent", f"{(x['crossover_gain'] > 0).mean():.0%}")
        cols[2].metric("Median gain", f"{x['crossover_gain'].median():+.4f}")


# --------------------------------------------------------------------------- timing

def tab_timing(run, cands):
    t = metrics.time_by_generation(run)
    if run.progress.empty:
        common.no_data("This run has no progress.jsonl. Timings are recorded from the loop version "
                       "of 6 November on.")
        return
    if t.empty:
        common.no_data("No model call or measurement finished yet.")
        return
    c = common.colors()
    total = metrics.run_duration(run) or 0
    model = t[t["part"].str.startswith("model")]["seconds"].sum()
    solver = t[t["part"] == "solver"]["seconds"].sum()
    cols = st.columns(3)
    cols[0].metric("Duration", f"{total / 3600:.2f} h")
    cols[1].metric("Model calls", f"{model / 60:.0f} min", f"{model / total:.0%} of duration" if total else None,
                   delta_color="off")
    cols[2].metric("Solver", f"{solver / 60:.0f} min", f"{solver / total:.0%} of duration" if total else None,
                   delta_color="off")

    parts = ["solver"] + [f"model: {r}" for r in loader.MODEL_ROLES]
    parts = [p for p in parts if p in set(t["part"])]
    fig = go.Figure()
    for i, part in enumerate(parts):
        s = t[t["part"] == part]
        fig.add_trace(go.Bar(x=s["generation"], y=s["seconds"] / 60, name=part, marker=dict(color=c["series"][i % 8]),
                             hovertemplate=f"gen %{{x}}: %{{y:.1f}} min {part}<extra></extra>"))
    fig.update_layout(barmode="stack", bargap=0.25)
    fig.update_xaxes(title="generation", dtick=1)
    fig.update_yaxes(title="minutes")
    common.show(common.style(fig, height=380), key="timing")

    calls = metrics.llm_calls(run)
    if not calls.empty:
        st.markdown("**Model calls per role**")
        agg = calls.groupby("role").agg(calls=("seconds", "size"), total_min=("seconds", lambda s: s.sum() / 60),
                                        mean_s=("seconds", "mean"), max_s=("seconds", "max"),
                                        failed=("ok", lambda s: int((s == False).sum()))).reset_index()  # noqa: E712
        st.dataframe(agg.sort_values("total_min", ascending=False), hide_index=True, width="stretch",
                     column_config={"total_min": st.column_config.NumberColumn("total min", format="%.1f"),
                                    "mean_s": st.column_config.NumberColumn("mean s", format="%.1f"),
                                    "max_s": st.column_config.NumberColumn("max s", format="%.1f")})


# --------------------------------------------------------------------------- config

def tab_config(run):
    left, right = st.columns(2)
    with left:
        st.markdown("**config.json**")
        st.code(json.dumps(run.config, indent=2), language="json")
    with right:
        b = run.baseline
        st.markdown("**Baseline**")
        if b:
            n_base = len(b.get("baseline_values", {}))
            st.markdown(f"{n_base} of {b.get('n_queries', '?')} queries have a baseline value "
                        f"(the others timed out or failed with the default heuristic and are not scored).")
        else:
            st.caption("No baseline.json in this run (recorded from 6 November on).")
    st.markdown("**Prompts of this run**")
    if not run.prompts:
        st.caption("No prompt files in this run.")
    for pname, text in run.prompts.items():
        with st.expander(pname):
            st.code(text, language=None, wrap_lines=True)


# --------------------------------------------------------------------------- page

def page():
    run, cands = common.load_by_name(name)
    header(run, cands)
    live_panel(run)
    tabs = st.tabs(["Progress", "Generations", "Translation", "Queries", "Terms", "Lineage", "Timing",
                    "Config & prompts"])
    with tabs[0]:
        tab_progress(run, cands)
    with tabs[1]:
        tab_generations(run, cands, run.attempts)
    with tabs[2]:
        tab_translation(run, cands, run.attempts)
    with tabs[3]:
        tab_queries(run, cands)
    with tabs[4]:
        tab_terms(run, cands)
    with tabs[5]:
        tab_lineage(run, cands)
    with tabs[6]:
        tab_timing(run, cands)
    with tabs[7]:
        tab_config(run)


# a running run reloads itself; a finished one only on interaction
_run, _ = common.load_by_name(name)
if _run.status == "running":
    st.fragment(run_every=common.REFRESH_SECONDS)(page)()
else:
    page()
