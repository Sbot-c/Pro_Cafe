# app_pro.py - Smart Café Pro dashboard (v2)
# Run:  pip install -r requirements.txt   then   streamlit run app_pro.py
# Needs (same folder): cafe_pro.py, policies_pro.py, agent_pro.py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Smart Café Pro", page_icon="☕", layout="wide")
try:
    from cafe_pro import CafePro, PRODUCTS
    import policies_pro as pp
    import agent_pro as ag
except Exception as e:
    st.error("Could not import cafe_pro.py / policies_pro.py / agent_pro.py. Keep them in the same folder as this file.")
    st.exception(e); st.stop()

# ---------------- look ----------------
AI_C, RULE_C, MUTED_C, GOOD_C, BAD_C, ACC_C = "#1F8A70", "#5B6B7A", "#B8C2CC", "#1F8A70", "#C2543A", "#E3A32B"
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700&display=swap');
h1,h2,h3,.hero h1{font-family:'Bricolage Grotesque',sans-serif !important;letter-spacing:-.01em}
.hero{padding:1.4rem 1.8rem;border-radius:14px;background:#12383A;color:#F2F7F6;margin-bottom:1.1rem;
      border-bottom:4px solid #E3A32B}
.hero h1{margin:0;font-size:2.1rem;color:#F2F7F6;padding:0}
.hero p{margin:.35rem 0 0;max-width:62ch;opacity:.88;line-height:1.5}
.note{border-left:4px solid #1F8A70;padding:.65rem 1rem;border-radius:8px;background:rgba(31,138,112,.09);margin:.6rem 0}
.note.warn{border-color:#E3A32B;background:rgba(227,163,43,.12)}
div[data-testid="stMetric"]{padding:.2rem .1rem}
button[data-baseweb="tab"]{font-weight:600}
@media (max-width:640px){.hero h1{font-size:1.5rem}}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>☕ Smart Café Pro</h1><p>One AI agent per product runs the stock room: 10 perishables, '
            'delivery delays, shelf lives, weather and weekends. Train it, stress-test it, then try to beat it.</p></div>',
            unsafe_allow_html=True)

# ---------------- sidebar: the business ----------------
PRESETS = {
    "☕ Normal week":        dict(temp=28, late=10, traffic=1.0, spoil=1.0, stock=1.0, store=1.0),
    "🔥 Heatwave":           dict(temp=40, late=10, traffic=1.2, spoil=1.5, stock=1.0, store=1.0),
    "🚚 Unreliable supplier": dict(temp=28, late=40, traffic=1.0, spoil=1.0, stock=1.5, store=1.0),
    "🎉 Festival rush":      dict(temp=30, late=15, traffic=1.5, spoil=1.0, stock=2.0, store=1.0),
    "🧊 Tiny fridge":        dict(temp=28, late=10, traffic=1.0, spoil=1.0, stock=1.0, store=0.5),
    "🗑️ Waste hurts":        dict(temp=28, late=10, traffic=1.0, spoil=3.0, stock=1.0, store=1.0),
}
for k, v in PRESETS["☕ Normal week"].items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("log", [])


def apply_preset():
    for k, v in PRESETS[st.session_state["preset"]].items():
        st.session_state[k] = v


sb = st.sidebar
sb.header("🏪 The café")
sb.selectbox("Start from a scenario", list(PRESETS), key="preset", on_change=apply_preset,
             help="Loads a set of sliders. You can still fine-tune them below.")
temp = sb.slider("🌡️ Average temperature (°C)", 18, 42, key="temp")
late = sb.slider("🚚 Supplier late-delivery chance (%)", 0, 50, step=5, key="late")
traffic = sb.slider("👥 Customer traffic (× normal)", 0.5, 1.5, step=0.1, key="traffic")
spoil_x = sb.slider("🗑️ Spoilage cost (× normal)", 0.5, 3.0, step=0.25, key="spoil")
stock_x = sb.slider("😡 Stockout penalty (× normal)", 0.5, 3.0, step=0.25, key="stock")
store_x = sb.slider("🧊 Storage space (× normal)", 0.4, 2.0, step=0.1, key="store")
sb.header("🧠 The AI")
episodes = sb.select_slider("📅 Training months", [2000, 4000, 6000], value=4000,
                            help="More months = more practice, but a longer wait. Repeat runs with identical settings load instantly.")
go_btn = sb.button("🚀 Train the AI", type="primary", width="stretch")
CFG = dict(temp_mean=float(temp), delay_prob=late / 100, demand_scale=float(traffic),
           spoil_mult=float(spoil_x), stockout_mult=float(stock_x), storage_scale=float(store_x))


# ---------------- engine ----------------
@st.cache_data(show_spinner=False, max_entries=8)
def run_all(cfg, episodes):
    env = CafePro(**cfg, seed=7)
    Q, curve, byp = ag.train(env, episodes=episodes, seed=1)
    ev = lambda pol, m=100: pp.evaluate(CafePro(**cfg, seed=99), pol, m)
    rule, safety = pp.tune_order_up_to(CafePro(**cfg, seed=5), months=60)
    res = {"🧠 AI agents": ev(ag.make_policy(Q)), "🧮 Tuned rule": ev(rule),
           "⚖️ Match the average": ev(pp.match_average), "💸 Generous (1.5×)": ev(pp.generous),
           "🎲 Random": ev(pp.random_manager), "😬 Cautious (0.5×)": ev(pp.cautious)}
    return dict(cfg=cfg, Q=Q, curve=curve, byp=byp, res=res, episodes=episodes, mean=env.mean, safety=safety)


def policies(R):
    return {"🧠 AI agents": ag.make_policy(R["Q"]), "🧮 Tuned rule": pp.make_order_up_to(R["safety"]),
            "⚖️ Match the average": pp.match_average, "💸 Generous (1.5×)": pp.generous,
            "😬 Cautious (0.5×)": pp.cautious, "🎲 Random": pp.random_manager}


def simulate(cfg, policy, seed):
    """One 30-day month with a fixed seed. Same seed = same weather, demand and late trucks for every manager."""
    env = CafePro(**cfg, seed=seed); obs = env.reset(); done = False
    keys = ("order", "demand", "sold", "spoiled", "stockout", "holding", "order_cost", "profit")
    rec = {k: [] for k in keys + ("on_hand",)}
    while not done:
        _, obs, done, info = env.step(policy(env, obs))
        for k in keys:
            rec[k].append(info[k])
        rec["on_hand"].append(obs["on_hand"])
    return {k: np.array(v) for k, v in rec.items()}


def breakdown(S, cfg):
    price = np.array([p["price"] for p in PRODUCTS])
    sp = np.array([p["spoil"] for p in PRODUCTS]) * cfg["spoil_mult"]
    pen = np.array([p["stockout"] for p in PRODUCTS]) * cfg["stockout_mult"]
    return {"Sales": (S["sold"] * price).sum(), "Purchases": -S["order_cost"].sum(), "Holding": -S["holding"].sum(),
            "Spoilage": -(S["spoiled"] * sp).sum(), "Stockouts": -(S["stockout"] * pen).sum()}


def tidy(fig, h=380, **kw):
    fig.update_layout(height=h, margin=dict(t=30, b=10, l=10, r=10), legend=dict(orientation="h", y=1.12), **kw)
    return fig


# ---------------- train / state ----------------
if go_btn:
    with st.status("Training: the agents are practising thousands of simulated months…", expanded=False) as s:
        R = run_all(CFG, episodes)
        s.update(label="Training complete", state="complete")
    st.session_state["R"] = R
    a, r = R["res"]["🧠 AI agents"], R["res"]["🧮 Tuned rule"]
    st.session_state["log"].append({
        "Run": len(st.session_state["log"]) + 1, "Scenario": st.session_state["preset"], "🌡️ °C": temp, "🚚 Late %": late,
        "👥 Traffic": traffic, "🗑️ Spoil ×": spoil_x, "😡 Stockout ×": stock_x, "🧊 Space ×": store_x,
        "🧠 AI profit": round(a["profit"].sum()), "🧮 Rule profit": round(r["profit"].sum()),
        "AI units / day": round(a["orders_per_day"].sum(), 1),
        "AI waste %": round(100 * np.average(a["spoil_rate"], weights=R["mean"]), 1)})
if "R" not in st.session_state:
    st.info("👈 Pick a scenario on the left, then press **🚀 Train the AI**. The first run takes about a minute.")
    st.stop()

R = st.session_state["R"]; res = R["res"]; ai = res["🧠 AI agents"]; rule = res["🧮 Tuned rule"]
names = [f"{p['emoji']} {p['name']}" for p in PRODUCTS]; w = R["mean"]; cfg = R["cfg"]
if cfg != CFG:
    st.markdown('<div class="note warn"><b>Sidebar settings changed.</b> The results below are for the previous settings. '
                'Press <b>🚀 Train the AI</b> to update them.</div>', unsafe_allow_html=True)

tabs = st.tabs(["🏠 Overview", "🎬 Month replay", "🥊 You vs AI", "📈 Learning", "🗺️ Strategy",
                "🏆 Scoreboard", "🧪 What-if lab", "💡 Insights"])

# ---------------- Overview ----------------
with tabs[0]:
    gap = ai["profit"].sum() - rule["profit"].sum()
    waste, rwaste = np.average(ai["spoil_rate"], weights=w), np.average(rule["spoil_rate"], weights=w)
    c = st.columns(4)
    with c[0].container(border=True):
        st.metric("💰 AI profit / month", f"AED {ai['profit'].sum():,.0f}", f"{gap:+,.0f} vs tuned rule")
    with c[1].container(border=True):
        st.metric("✅ Service level", f"{100 * np.average(ai['service'], weights=w):.0f}%",
                  f"{100 * (np.average(ai['service'], weights=w) - np.average(rule['service'], weights=w)):+.1f} pts vs rule")
    with c[2].container(border=True):
        st.metric("🗑️ Waste rate", f"{100 * waste:.1f}%", f"{100 * (waste - rwaste):+.1f} pts vs rule", delta_color="inverse")
    with c[3].container(border=True):
        st.metric("📦 Units ordered / day", f"{ai['orders_per_day'].sum():.0f}",
                  f"{ai['orders_per_day'].sum() - rule['orders_per_day'].sum():+.0f} vs rule", delta_color="off")
    verdict = "ahead of" if gap > 0 else "behind"
    best = names[int(np.argmax(ai["profit"] - rule["profit"]))]; worst = names[int(np.argmin(ai["profit"] - rule["profit"]))]
    st.markdown(f'<div class="note">In this setup the AI finishes <b>{verdict}</b> the tuned rule by AED {abs(gap):,.0f} a month. '
                f'Its best product against the rule is <b>{best}</b>; its weakest is <b>{worst}</b>.</div>', unsafe_allow_html=True)
    st.markdown("#### Product by product")
    df = pd.DataFrame({"Product": names, "Shelf (days)": [p["shelf"] for p in PRODUCTS],
                       "Delivery (days)": [p["lead"] for p in PRODUCTS],
                       "AI profit": ai["profit"].round(0), "Rule profit": rule["profit"].round(0),
                       "AI minus rule": (ai["profit"] - rule["profit"]).round(0),
                       "Service %": (100 * ai["service"]).round(0), "Waste %": (100 * ai["spoil_rate"]).round(1)})
    st.dataframe(df, hide_index=True, width="stretch", column_config={
        "Service %": st.column_config.ProgressColumn("Service %", min_value=0, max_value=100, format="%d%%"),
        "Waste %": st.column_config.ProgressColumn("Waste %", min_value=0, max_value=50, format="%.1f%%"),
        "AI profit": st.column_config.NumberColumn(format="%d"), "Rule profit": st.column_config.NumberColumn(format="%d"),
        "AI minus rule": st.column_config.NumberColumn(format="%+d")})
    st.download_button("⬇️ Download product table (CSV)", df.to_csv(index=False).encode(), "cafe_products.csv", "text/csv")
    st.caption("Averages over 100 simulated months. Prices, costs and shelf lives are illustrative placeholders.")

# ---------------- Month replay ----------------
with tabs[1]:
    st.subheader("Replay one month, day by day")
    st.caption("Every manager faces the same weather, demand and late trucks for a given month number, so differences come from ordering alone.")
    P = policies(R)
    c1, c2, c3, c4 = st.columns([2, 2, 1, 2])
    pa = c1.selectbox("Manager A", list(P), index=0)
    pb = c2.selectbox("Compare with", list(P), index=1)
    seed = int(c3.number_input("Month #", 1, 9999, 42))
    prod = c4.selectbox("Product to inspect", names)
    SA, SB = simulate(cfg, P[pa], seed), simulate(cfg, P[pb], seed)
    days = np.arange(1, SA["profit"].shape[0] + 1)
    k1, k2 = st.columns(2)
    for col, nm, S in ((k1, pa, SA), (k2, pb, SB)):
        with col.container(border=True):
            st.metric(f"{nm}: profit this month", f"AED {S['profit'].sum():,.0f}",
                      f"{100 * S['sold'].sum() / max(S['demand'].sum(), 1):.0f}% of demand met, {S['spoiled'].sum():.0f} units spoiled",
                      delta_color="off")
    fig = go.Figure()
    fig.add_scatter(x=days, y=SA["profit"].sum(axis=1).cumsum(), name=pa, line=dict(width=3, color=AI_C))
    fig.add_scatter(x=days, y=SB["profit"].sum(axis=1).cumsum(), name=pb, line=dict(width=3, color=RULE_C, dash="dot"))
    st.plotly_chart(tidy(fig, 320, xaxis_title="Day", yaxis_title="Cumulative profit (AED)", title="Cumulative profit"), width="stretch")
    L, Rt = st.columns(2)
    bd = pd.DataFrame({pa: breakdown(SA, cfg), pb: breakdown(SB, cfg)})
    fig = go.Figure()
    fig.add_bar(x=bd.index, y=bd[pa], name=pa, marker_color=AI_C)
    fig.add_bar(x=bd.index, y=bd[pb], name=pb, marker_color=MUTED_C)
    L.plotly_chart(tidy(fig, 340, barmode="group", yaxis_title="AED", title="Where the money went"), width="stretch")
    j = names.index(prod)
    fig = go.Figure()
    fig.add_bar(x=days, y=SA["demand"][:, j], name="Demand", marker_color=MUTED_C)
    fig.add_scatter(x=days, y=SA["on_hand"][:, j], name="Stock at close", line=dict(width=3, color=AI_C))
    fig.add_scatter(x=days, y=SA["order"][:, j], name="Ordered", mode="markers", marker=dict(size=8, color=ACC_C, symbol="diamond"))
    fig.add_scatter(x=days, y=SA["stockout"][:, j], name="Missed sales", mode="markers", marker=dict(size=9, color=BAD_C, symbol="x"))
    Rt.plotly_chart(tidy(fig, 340, xaxis_title="Day", yaxis_title="Units", title=f"{prod} under {pa}"), width="stretch")

# ---------------- You vs AI ----------------
with tabs[2]:
    st.subheader("Can you out-manage the AI?")
    st.markdown("Set how many **extra days of cover** you want to hold for each product. More cover means fewer stockouts "
                "but more spoilage. Your plan is scored over 60 simulated months against the AI and the tuned rule.")
    cols = st.columns(2); mine = []
    for i, nm in enumerate(names):
        mine.append(cols[i % 2].slider(nm, 0.0, 4.0, 1.0, 0.5, key=f"my_{i}",
                                       help=f"Shelf life {PRODUCTS[i]['shelf']}d, delivery {PRODUCTS[i]['lead']}d. Cover is capped at shelf life."))
    if st.button("▶️ Score my plan", type="primary"):
        with st.spinner("Running 60 simulated months…"):
            st.session_state["mine"] = pp.evaluate(CafePro(**cfg, seed=99), pp.make_order_up_to(np.array(mine)), 60)
    if "mine" in st.session_state:
        me = st.session_state["mine"]; mp, ap, rp = me["profit"].sum(), ai["profit"].sum(), rule["profit"].sum()
        c = st.columns(3)
        c[0].metric("🙋 Your plan", f"AED {mp:,.0f}", f"{mp - ap:+,.0f} vs AI")
        c[1].metric("🧠 AI", f"AED {ap:,.0f}")
        c[2].metric("🧮 Tuned rule", f"AED {rp:,.0f}", f"{mp - rp:+,.0f} vs you", delta_color="off")
        fig = go.Figure()
        fig.add_bar(x=names, y=me["profit"], name="You", marker_color=ACC_C)
        fig.add_bar(x=names, y=ai["profit"], name="AI", marker_color=AI_C)
        st.plotly_chart(tidy(fig, 340, barmode="group", yaxis_title="Profit per month (AED)"), width="stretch")
        st.caption("Results are for the settings the AI was trained on. If you changed the sidebar, retrain first.")

# ---------------- Learning ----------------
with tabs[3]:
    st.subheader("Did it actually learn?")
    cu = R["curve"]; first, last = np.mean(cu[:200]), np.mean(cu[-200:])
    c = st.columns(3)
    c[0].metric("First 200 months", f"AED {first:,.0f}"); c[1].metric("Last 200 months", f"AED {last:,.0f}", f"{last - first:+,.0f}")
    c[2].metric("Months practised", f"{R['episodes']:,}")
    pick = st.multiselect("Show products", names, default=[], placeholder="Whole café (default)")
    wnd = max(1, len(cu) // 80); sm = lambda y: np.convolve(y, np.ones(wnd) / wnd, mode="valid")
    fig = go.Figure()
    if not pick:
        fig.add_scatter(y=sm(cu), name="Whole café", line=dict(width=3, color=AI_C))
    for nm in pick:
        fig.add_scatter(y=sm(R["byp"][:, names.index(nm)]), name=nm)
    if not pick:
        fig.add_hline(y=rule["profit"].sum(), line_dash="dash", line_color=RULE_C, annotation_text="Tuned rule")
    st.plotly_chart(tidy(fig, 420, xaxis_title="Training month", yaxis_title="Profit per month (AED)"), width="stretch")
    st.caption("Training profit includes random experimenting, so it sits below the final scored profit. The early dip is exploration; the climb is learning.")

# ---------------- Strategy ----------------
with tabs[4]:
    st.subheader("The strategy the AI discovered")
    c1, c2 = st.columns(2)
    dt = c1.radio("📅 Day type", ["Mon–Thu", "Friday", "Weekend"], horizontal=True)
    ht = c2.radio("🌦️ Weather", ["Cool", "Mild", "Hot"], index=1, horizontal=True)
    di, hi = ["Mon–Thu", "Friday", "Weekend"].index(dt), ["Cool", "Mild", "Hot"].index(ht)
    Q = R["Q"]; z = ag.policy_table(Q, onhand=1, daytype=di, heat=hi).astype(float)
    z[np.abs(Q[:, :, 1, di, hi, :]).sum(axis=2) == 0] = np.nan
    fig = go.Figure(go.Heatmap(z=z, x=[f"{i / 2:.1f}" for i in range(z.shape[1])], y=names,
                               text=np.where(np.isnan(z), "", z.astype(str)), texttemplate="%{text}",
                               colorscale="Teal", colorbar=dict(title="Order ×")))
    fig.update_layout(xaxis_title="Days of stock already owned (shelf + on the road)", yaxis_autorange="reversed")
    st.plotly_chart(tidy(fig, 460), width="stretch")
    st.caption("Each cell shows how many times a normal day's demand the AI orders. Blank means the situation never came up in training. "
               "A good strategy orders a lot when stock is low and nothing when stock is high.")

# ---------------- Scoreboard ----------------
with tabs[5]:
    st.subheader("Who runs the better café?")
    sc = pd.DataFrame([{"Manager": k, "Profit / month": v["profit"].sum(),
                        "Service %": 100 * np.average(v["service"], weights=w),
                        "Waste %": 100 * np.average(v["spoil_rate"], weights=w),
                        "Units / day": v["orders_per_day"].sum()} for k, v in res.items()]).sort_values("Profit / month")
    fig = go.Figure(go.Bar(x=sc["Profit / month"], y=sc["Manager"], orientation="h",
                           marker_color=[AI_C if "AI" in m else (RULE_C if "rule" in m else MUTED_C) for m in sc["Manager"]],
                           text=[f"{v:,.0f}" for v in sc["Profit / month"]], textposition="outside"))
    st.plotly_chart(tidy(fig, 360, xaxis_title="Average profit per month (AED)"), width="stretch")
    st.dataframe(sc.sort_values("Profit / month", ascending=False), hide_index=True, width="stretch", column_config={
        "Profit / month": st.column_config.NumberColumn(format="%d"), "Service %": st.column_config.NumberColumn(format="%.0f%%"),
        "Waste %": st.column_config.NumberColumn(format="%.1f%%"), "Units / day": st.column_config.NumberColumn(format="%.0f")})
    fig = go.Figure()
    fig.add_bar(x=names, y=ai["profit"], name="AI agents", marker_color=AI_C)
    fig.add_bar(x=names, y=rule["profit"], name="Tuned rule", marker_color=RULE_C)
    st.plotly_chart(tidy(fig, 360, barmode="group", yaxis_title="Profit per month (AED)"), width="stretch")

# ---------------- What-if lab ----------------
with tabs[6]:
    st.subheader("What happens when the business changes?")
    st.markdown("Pick a scenario or move a slider, press **🚀 Train the AI**, and each run is logged here. "
                "Change one thing at a time and watch **AI units / day** and **AI waste %**.")
    if not st.session_state["log"]:
        st.info("No runs yet. Train the AI to log your first run.")
    else:
        Lg = pd.DataFrame(st.session_state["log"])
        st.dataframe(Lg, hide_index=True, width="stretch")
        if len(Lg) > 1:
            lab = Lg["Run"].astype(str) + " " + Lg["Scenario"]
            fig = go.Figure()
            fig.add_bar(x=lab, y=Lg["🧠 AI profit"], name="AI", marker_color=AI_C)
            fig.add_bar(x=lab, y=Lg["🧮 Rule profit"], name="Tuned rule", marker_color=RULE_C)
            st.plotly_chart(tidy(fig, 340, barmode="group", yaxis_title="Profit per month (AED)"), width="stretch")
        c1, c2, _ = st.columns([1, 1, 3])
        c1.download_button("⬇️ Download log (CSV)", Lg.to_csv(index=False).encode(), "whatif_log.csv", "text/csv")
        if c2.button("🧹 Clear the log"):
            st.session_state["log"] = []; st.rerun()

# ---------------- Insights ----------------
with tabs[7]:
    st.subheader("What we learned, including where the AI falls short")
    d = ai["profit"] - rule["profit"]
    gapdf = pd.DataFrame({"Product": names, "Delivery delay (days)": [str(p["lead"]) for p in PRODUCTS], "AI minus rule (AED/month)": d.round(0)})
    fig = px.bar(gapdf, x="Product", y="AI minus rule (AED/month)", color="Delivery delay (days)",
                 color_discrete_sequence=px.colors.sequential.Teal_r)
    fig.add_hline(y=0, line_dash="dash")
    st.plotly_chart(tidy(fig, 360), width="stretch")
    short = [i for i, p in enumerate(PRODUCTS) if p["lead"] <= 1]; longl = [i for i, p in enumerate(PRODUCTS) if p["lead"] >= 2]
    st.markdown(f'<div class="note"><b>Live reading of this run.</b> Same-day or next-day products: the AI is <b>{d[short].mean():+,.0f}</b> '
                f'AED/month per product vs the tuned rule. Products with a 2+ day delay: <b>{d[longl].mean():+,.0f}</b>. '
                f'Whole café: <b>{d.sum():+,.0f}</b>.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown("##### ✅ Where the AI shines\n- Beats every guessing manager\n"
                "- Learns *order a lot when stock is low, little when high* with no one telling it\n"
                "- Strong on fast products (e.g. 🥪 sandwiches) where cause and effect are quick\n"
                "- When costs change it **retrains** instead of someone re-tuning rules by hand")
    c2.markdown("##### ⚠️ Where it falls short\n- Can land below a well-tuned order-up-to rule overall\n"
                "- Weakest where deliveries are slow: an order is paid today but earns days later\n"
                "- Plain one-day Q-learning could not link the two; 5-day learning closed part of the gap\n"
                "- Tabular learning can't see stock by age; a neural network (DQN) could")
    with st.expander("🔬 Development log (indicative figures from earlier test runs)"):
        st.dataframe(pd.DataFrame([
            ["Cautious manager (half of normal)", "about -10,000", "Loses money: stockouts cost more than waste"],
            ["Random manager", "about 1,200", ""], ["Generous manager (1.5×)", "about 7,200", ""],
            ["Match the average", "about 11,800", "Strong simple baseline"],
            ["AI v1: stock on hand + in transit", "about 11,800", "Matches 'match the average'"],
            ["AI v2: total inventory position", "about 11,900", "Barely better"],
            ["AI v2, trained 3× longer", "about 11,300", "More training did not help: a design limit"],
            ["AI picks stock target / adjusts the rule", "about 9,200 - 11,600", "Worse: credit-assignment problem"],
            ["AI v3: 5-day (n-step) learning", "about 12,500 - 13,000", "Best AI so far"],
            ["Tuned order-up-to rule", "about 14,200", "Classic operations benchmark"]],
            columns=["Approach", "Profit AED / month", "Takeaway"]), hide_index=True, width="stretch")
    with st.expander("📌 Assumptions & limits"):
        st.markdown("- Prices, costs, shelf lives and delivery times are **illustrative placeholders**; use real café data.\n"
                    "- Orders arrive in full (apart from late days); deliveries that don't fit in storage are refused but still paid for.\n"
                    "- Demand depends on weekday, weather and a shared busy-day factor; no promotions or substitution.\n"
                    "- Results come from simulation and vary a little between runs.\n"
                    "- **Next steps:** neural-network agent, upload real sales data, forecast uncertainty.")
