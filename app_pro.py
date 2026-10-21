# app_pro.py - Smart Café Pro dashboard
# Run:  pip install numpy pandas plotly streamlit   then   streamlit run app_pro.py
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
    st.error("Could not import cafe_pro.py / policies_pro.py / agent_pro.py - keep them in this folder.")
    st.exception(e); st.stop()

st.markdown("""<style>
.hero{padding:1.2rem 1.6rem;border-radius:16px;background:linear-gradient(120deg,#7c3aed 0%,#2563eb 55%,#06b6d4 100%);color:#fff;margin-bottom:1rem}
.hero h1{margin:0;font-size:2rem;color:#fff}.hero p{margin:.2rem 0 0;opacity:.92}
.kpi{border:1px solid rgba(128,128,128,.28);border-radius:14px;padding:.9rem 1.1rem;background:rgba(128,128,128,.07)}
.kpi .l{font-size:.85rem;opacity:.75}.kpi .v{font-size:1.7rem;font-weight:700}.kpi .s{font-size:.8rem;opacity:.7}
.note{border-left:4px solid #7c3aed;padding:.6rem 1rem;border-radius:8px;background:rgba(124,58,237,.08);margin:.5rem 0}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>☕ Smart Café Pro</h1><p>An AI that runs the stock room of a whole café: 10 perishable products, '
            'delivery delays, shelf lives, weather and weekends.</p></div>', unsafe_allow_html=True)

# ---------------- sidebar: the business ----------------
sb = st.sidebar
sb.header("🏪 The Café")
temp = sb.slider("🌡️ Average temperature (°C)", 18, 42, 28)
late = sb.slider("🚚 Supplier late-delivery chance (%)", 0, 50, 10, 5)
traffic = sb.slider("👥 Customer traffic (× normal)", 0.5, 1.5, 1.0, 0.1)
spoil_x = sb.slider("🗑️ Spoilage cost (× normal)", 0.5, 3.0, 1.0, 0.25)
stock_x = sb.slider("😡 Stockout penalty (× normal)", 0.5, 3.0, 1.0, 0.25)
store_x = sb.slider("🧊 Storage space (× normal)", 0.4, 2.0, 1.0, 0.1)
sb.header("🧠 The AI")
episodes = sb.select_slider("📅 Training months", [2000, 4000, 6000], value=4000)
go_btn = sb.button("🚀 Train the AI", type="primary", width="stretch")
CFG = dict(temp_mean=float(temp), delay_prob=late / 100, demand_scale=traffic,
           spoil_mult=spoil_x, stockout_mult=stock_x, storage_scale=store_x)


def run_all(cfg, episodes):
    env = CafePro(**cfg, seed=7)
    Q, curve, byp = ag.train(env, episodes=episodes, seed=1)
    ev = lambda pol, m=100: pp.evaluate(CafePro(**cfg, seed=99), pol, m)
    rule, safety = pp.tune_order_up_to(CafePro(**cfg, seed=5), months=60)
    res = {"🧠 AI agents": ev(ag.make_policy(Q)), "🧮 Tuned order-up-to rule": ev(rule),
           "⚖️ Match the average": ev(pp.match_average), "💸 Generous (1.5×)": ev(pp.generous),
           "🎲 Random": ev(pp.random_manager), "😬 Cautious (0.5×)": ev(pp.cautious)}
    return dict(cfg=cfg, Q=Q, curve=curve, byp=byp, res=res, episodes=episodes, mean=env.mean)


st.session_state.setdefault("log", [])
if go_btn:
    with st.spinner("The AI is practising thousands of simulated months... (about a minute)"):
        R = run_all(CFG, episodes)
    st.session_state["R"] = R
    ai, rule = R["res"]["🧠 AI agents"], R["res"]["🧮 Tuned order-up-to rule"]
    st.session_state["log"].append({
        "Run": len(st.session_state["log"]) + 1, "🌡️ °C": temp, "🚚 Late %": late, "👥 Traffic": traffic,
        "🗑️ Spoil ×": spoil_x, "😡 Stockout ×": stock_x, "🧊 Space ×": store_x,
        "🧠 AI profit": round(ai["profit"].sum()), "🧮 Rule profit": round(rule["profit"].sum()),
        "AI units ordered / day": round(ai["orders_per_day"].sum(), 1),
        "AI waste %": round(100 * np.average(ai["spoil_rate"], weights=R["mean"]), 1)})
if "R" not in st.session_state:
    st.info("👈 Set up the café on the left, then press **🚀 Train the AI**.")
    st.stop()

R = st.session_state["R"]; res = R["res"]; ai = res["🧠 AI agents"]; rule = res["🧮 Tuned order-up-to rule"]
names = [f"{p['emoji']} {p['name']}" for p in PRODUCTS]
w = R["mean"]


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


tabs = st.tabs(["🏠 Overview", "📈 Learning", "🗺️ Strategy", "🏆 Scoreboard", "🧪 What-if Lab", "💡 Insights"])

# ---------------- Overview ----------------
with tabs[0]:
    c = st.columns(4)
    gap = ai["profit"].sum() - rule["profit"].sum()
    kpi(c[0], "💰 AI profit / month", f"AED {ai['profit'].sum():,.0f}", f"{gap:+,.0f} vs tuned rule")
    kpi(c[1], "✅ Service level", f"{100 * np.average(ai['service'], weights=w):.0f}%", "customer demand met")
    kpi(c[2], "🗑️ Waste rate", f"{100 * np.average(ai['spoil_rate'], weights=w):.1f}%", "units spoiled / ordered")
    kpi(c[3], "📦 Units ordered / day", f"{ai['orders_per_day'].sum():.0f}", "across all products")
    st.markdown("#### 📋 Product by product")
    df = pd.DataFrame({"Product": names, "⏳ Shelf life (days)": [p["shelf"] for p in PRODUCTS],
                       "🚚 Delivery (days)": [p["lead"] for p in PRODUCTS],
                       "🧠 AI profit": ai["profit"].round(0), "🧮 Rule profit": rule["profit"].round(0),
                       "✅ Service %": (100 * ai["service"]).round(0), "🗑️ Waste %": (100 * ai["spoil_rate"]).round(1)})
    st.dataframe(df, hide_index=True, width="stretch")
    st.caption("Numbers are averages over 100 simulated months. Product prices, costs and shelf lives are illustrative placeholders.")

# ---------------- Learning ----------------
with tabs[1]:
    st.subheader("Did it actually learn?")
    pick = st.multiselect("Show products", names, default=[], placeholder="Whole café (default)")
    wnd = max(1, len(R["curve"]) // 80)
    sm = lambda y: np.convolve(y, np.ones(wnd) / wnd, mode="valid")
    fig = go.Figure()
    if not pick:
        fig.add_scatter(y=sm(R["curve"]), name="Whole café", line=dict(width=3, color="#7c3aed"))
    for nm in pick:
        fig.add_scatter(y=sm(R["byp"][:, names.index(nm)]), name=nm)
    fig.update_layout(xaxis_title="Training month", yaxis_title="Profit per month (AED)", height=420,
                      margin=dict(t=20, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Whole café: AED {np.mean(R['curve'][:200]):,.0f}/month at the start → AED {np.mean(R['curve'][-200:]):,.0f}/month at the end. "
               "The early dip is the agents experimenting; the climb is the learning.")

# ---------------- Strategy ----------------
with tabs[2]:
    st.subheader("The strategy the AI discovered")
    c1, c2 = st.columns(2)
    dt = c1.radio("📅 Day type", ["Mon–Thu", "Friday", "Weekend"], horizontal=True)
    ht = c2.radio("🌦️ Weather", ["Cool", "Mild", "Hot"], index=1, horizontal=True)
    di, hi = ["Mon–Thu", "Friday", "Weekend"].index(dt), ["Cool", "Mild", "Hot"].index(ht)
    Q = R["Q"]; z = ag.policy_table(Q, onhand=1, daytype=di, heat=hi).astype(float)
    z[np.abs(Q[:, :, 1, di, hi, :]).sum(axis=2) == 0] = np.nan          # situations the AI never met
    fig = go.Figure(go.Heatmap(z=z, x=[f"{i / 2:.1f}" for i in range(z.shape[1])], y=names,
                               text=np.where(np.isnan(z), "", z.astype(str)), texttemplate="%{text}",
                               colorscale="Viridis", colorbar=dict(title="Order ×")))
    fig.update_layout(xaxis_title="Days of stock already owned (on the shelf + on the road)",
                      height=460, margin=dict(t=20, b=10), yaxis_autorange="reversed")
    st.plotly_chart(fig, width="stretch")
    st.caption("Each cell: how many times a normal day's demand the AI orders. Blank = situation never met in training. "
               "Look for: order a lot when stock is low, nothing when stock is high.")

# ---------------- Scoreboard ----------------
with tabs[3]:
    st.subheader("Who runs the better café?")
    tot = sorted(((k, v["profit"].sum()) for k, v in res.items()), key=lambda t: t[1])
    fig = go.Figure(go.Bar(x=[t[1] for t in tot], y=[t[0] for t in tot], orientation="h",
                           marker_color=["#7c3aed" if "AI" in t[0] else "#94a3b8" for t in tot],
                           text=[f"{t[1]:,.0f}" for t in tot], textposition="outside"))
    fig.update_layout(xaxis_title="Average profit per month (AED)", height=380, margin=dict(t=20, b=10))
    st.plotly_chart(fig, width="stretch")
    fig = go.Figure()
    fig.add_bar(x=names, y=ai["profit"], name="🧠 AI agents", marker_color="#7c3aed")
    fig.add_bar(x=names, y=rule["profit"], name="🧮 Tuned rule", marker_color="#94a3b8")
    fig.update_layout(barmode="group", yaxis_title="Profit per month (AED)", height=380, margin=dict(t=20, b=10))
    st.plotly_chart(fig, width="stretch")

# ---------------- What-if Lab ----------------
with tabs[4]:
    st.subheader("🧪 What happens when the business changes?")
    st.markdown("Change a slider on the left, press **🚀 Train the AI** again, and the run is logged here. "
                "Try one change at a time (e.g. spoilage cost × 3) and watch **AI units ordered / day** and **waste %**.")
    if st.session_state["log"]:
        L = pd.DataFrame(st.session_state["log"])
        st.dataframe(L, hide_index=True, width="stretch")
        if len(L) > 1:
            fig = go.Figure()
            fig.add_bar(x=L["Run"].astype(str), y=L["🧠 AI profit"], name="🧠 AI", marker_color="#7c3aed")
            fig.add_bar(x=L["Run"].astype(str), y=L["🧮 Rule profit"], name="🧮 Tuned rule", marker_color="#94a3b8")
            fig.update_layout(barmode="group", xaxis_title="Run", yaxis_title="Profit per month (AED)", height=340)
            st.plotly_chart(fig, width="stretch")
        if st.button("🧹 Clear the log"):
            st.session_state["log"] = []; st.rerun()

# ---------------- Insights ----------------
with tabs[5]:
    st.subheader("💡 What we learned - including where the AI falls short")
    gapdf = pd.DataFrame({"Product": names, "Delivery delay (days)": [p["lead"] for p in PRODUCTS],
                          "AI minus rule (AED/month)": (ai["profit"] - rule["profit"]).round(0)})
    gapdf["Delivery delay (days)"] = gapdf["Delivery delay (days)"].astype(str)
    fig = px.bar(gapdf, x="Product", y="AI minus rule (AED/month)", color="Delivery delay (days)",
                 color_discrete_sequence=px.colors.sequential.Viridis_r)
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(height=380, margin=dict(t=20, b=10))
    st.plotly_chart(fig, width="stretch")
    short = [i for i, p in enumerate(PRODUCTS) if p["lead"] <= 1]
    longl = [i for i, p in enumerate(PRODUCTS) if p["lead"] >= 2]
    d = ai["profit"] - rule["profit"]
    st.markdown(f'<div class="note"><b>Live reading of this run.</b> Products delivered same day or next day: the AI is '
                f'<b>{d[short].mean():+,.0f}</b> AED/month per product vs the tuned rule. Products with a 2+ day delay: '
                f'<b>{d[longl].mean():+,.0f}</b>. Whole café: <b>{d.sum():+,.0f}</b>.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown("##### ✅ Where the AI shines\n- Beats every guessing manager (cautious, generous, random, match-the-average)\n"
                "- Learns the pattern *order a lot when stock is low, little when high* with no one telling it\n"
                "- Strong on same-day products (e.g. 🥪 sandwiches) where cause and effect are quick\n"
                "- When costs change, it **retrains** instead of someone re-tuning rules by hand")
    c2.markdown("##### ⚠️ Where it falls short\n- Slightly below a well-tuned order-up-to rule overall\n"
                "- Weakest where deliveries are slow: an order is paid today but earns days later\n"
                "- Plain one-day Q-learning could not link the two; 5-day learning closed part of the gap\n"
                "- Tabular learning can't see stock by age; a neural network (DQN) could")
    st.markdown("##### 🔬 Development log (indicative figures from our test runs)")
    st.dataframe(pd.DataFrame([
        ["Cautious manager (order half of normal)", "about -10,000", "Loses money: stockouts cost more than waste"],
        ["Random manager", "about 1,200", ""],
        ["Generous manager (1.5×)", "about 7,200", ""],
        ["Match the average", "about 11,800", "Strong simple baseline"],
        ["AI v1: stock on hand + in transit", "about 11,800", "Matches 'match the average'"],
        ["AI v2: total inventory position", "about 11,900", "Barely better"],
        ["AI v2, trained 3× longer", "about 11,300", "More training did not help: a design limit"],
        ["AI picks stock target / adjusts the rule", "about 9,200 - 11,600", "Worse: credit-assignment problem"],
        ["AI v3: 5-day (n-step) learning", "about 12,500 - 13,000", "Best AI so far"],
        ["Tuned order-up-to rule", "about 14,200", "Classic operations benchmark"]],
        columns=["Approach", "Profit AED / month", "Takeaway"]), hide_index=True, width="stretch")
    with st.expander("📌 Assumptions & limits"):
        st.markdown("- Prices, costs, shelf lives and delivery times are **illustrative placeholders**; real café data should replace them.\n"
                    "- Orders arrive in full (apart from late days); deliveries that don't fit in storage are refused but still paid for.\n"
                    "- Demand depends on weekday, weather and a shared busy-day factor; no promotions or substitution between products.\n"
                    "- Results come from simulation and vary a little between runs.\n"
                    "- **Next steps:** neural-network agent, upload real sales data, a 'You vs AI' game, forecast uncertainty.")
