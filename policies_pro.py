"""
policies_pro.py - the 'human' managers and the classic operations benchmark.
Every policy is a function  policy(env, obs) -> array of order quantities (one per product).
"""
import numpy as np


def cautious(env, obs):
    return np.round(0.5 * env.mean)

def match_average(env, obs):
    return np.round(1.0 * env.mean)

def generous(env, obs):
    return np.round(1.5 * env.mean)

def random_manager(env, obs):
    return np.round(env.rng.uniform(0, 3, env.n) * env.mean)


def make_order_up_to(safety):
    """Classic (order-up-to) rule: top up inventory position to  mean * days_of_cover.
    safety = extra days of cover (a scalar, or one value per product)."""
    safety = np.broadcast_to(np.asarray(safety, dtype=float), (len(safety) if np.ndim(safety) else 1,))

    def policy(env, obs):
        s = np.broadcast_to(safety, (env.n,))
        out = np.zeros(env.n)
        for i, p in enumerate(env.P):
            cover = min(p["lead"] + 1 + s[i], p["lead"] + p["shelf"])   # never plan beyond shelf life
            target = env.mean[i] * cover
            out[i] = max(0.0, target - obs["on_hand"][i] - obs["in_transit"][i])
        return np.round(out)
    return policy


def evaluate(env, policy, months=200):
    """Average monthly profit per product (array) plus service/spoilage stats."""
    prof = np.zeros(env.n); dem = np.zeros(env.n); sold = np.zeros(env.n)
    spoil = np.zeros(env.n); orders = np.zeros(env.n)
    for _ in range(months):
        obs = env.reset(); done = False
        while not done:
            r, obs, done, info = env.step(policy(env, obs))
            prof += r; dem += info["demand"]; sold += info["sold"]
            spoil += info["spoiled"]; orders += info["order"]
    return dict(profit=prof / months, service=sold / np.maximum(dem, 1),
                spoil_rate=spoil / np.maximum(orders, 1), orders_per_day=orders / (months * env.days))


def tune_order_up_to(env, grid=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0), months=100):
    """Pick the best safety-days value PER PRODUCT (products only interact via storage)."""
    results = np.array([evaluate(env, make_order_up_to(g), months)["profit"] for g in grid])
    best = np.array(grid)[results.argmax(axis=0)]
    return make_order_up_to(best), best
