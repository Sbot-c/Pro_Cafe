"""
agent_pro.py - one Q-learning agent per product (all learning at the same time in the same café).

Each agent sees, for ITS product only:
    position : stock on hand + stock on the road, in half-days of typical demand (0..19)
    onhand   : stock physically on the shelf now, in days of typical demand        (0..3)
    daytype  : 0 = Mon-Thu, 1 = Fri, 2 = Sat/Sun
    heat     : 0 = cool, 1 = mild, 2 = hot  (cold drinks and juice sell more when hot)
and chooses how much to order as a MULTIPLE of the product's typical daily demand.
Using multiples (not raw units) lets the same design work for sugar and for cakes.
"""
import numpy as np

MULTIPLIERS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
N_POS, N_ONHAND, N_DAYTYPE, N_HEAT = 20, 4, 3, 3
HEAT_EDGES = (27.0, 31.0)


def _daytype(weekday):
    return 0 if weekday <= 3 else (1 if weekday == 4 else 2)


def encode(env, obs):
    """Return an (n_products, 4) integer array: the state of every product's agent."""
    position = np.clip(np.round((obs["on_hand"] + obs["in_transit"]) / env.mean * 2), 0, N_POS - 1)
    onhand = np.clip(np.round(obs["on_hand"] / env.mean), 0, N_ONHAND - 1)
    dt = _daytype(obs["weekday"])
    heat = int(obs["temp"] > HEAT_EDGES[0]) + int(obs["temp"] > HEAT_EDGES[1])
    out = np.empty((env.n, 4), dtype=int)
    out[:, 0] = position; out[:, 1] = onhand; out[:, 2] = dt; out[:, 3] = heat
    return out


def to_orders(env, actions):
    return np.round(MULTIPLIERS[actions] * env.mean)


def train(env, episodes=4000, alpha=0.05, gamma=0.95, n_step=5, eps_start=1.0, eps_min=0.05,
          eps_decay=None, seed=None):
    """n-step Q-learning. Why n-step? An order is paid for today but earns its money days later
    (delivery delay + time on the shelf). Looking only ONE day ahead can't connect the two;
    looking n_step days ahead can."""
    rng = np.random.default_rng(seed)
    n, nA = env.n, len(MULTIPLIERS)
    Q = np.zeros((n, N_POS, N_ONHAND, N_DAYTYPE, N_HEAT, nA))      # every agent starts knowing NOTHING
    if eps_decay is None:                                          # reach eps_min after 80% of training
        eps_decay = (eps_min / eps_start) ** (1.0 / (0.8 * episodes))
    eps = eps_start
    curve, curve_by_product = [], []
    idx = np.arange(n)

    def update(S, A, k0, G):
        st = S[k0]; a = A[k0]
        cur = Q[idx, st[:, 0], st[:, 1], st[:, 2], st[:, 3], a]
        Q[idx, st[:, 0], st[:, 1], st[:, 2], st[:, 3], a] = cur + alpha * (G - cur)

    for ep in range(episodes):
        obs = env.reset(); s = encode(env, obs); done = False
        S, A, R = [], [], []
        while not done:
            # epsilon-greedy, independently for every product
            explore = rng.random(n) < eps
            greedy = Q[idx, s[:, 0], s[:, 1], s[:, 2], s[:, 3]].argmax(axis=1)
            a = np.where(explore, rng.integers(0, nA, n), greedy)

            r, obs, done, _ = env.step(to_orders(env, a))
            s2 = encode(env, obs)
            S.append(s); A.append(a); R.append(r)

            t = len(R) - 1
            if t >= n_step - 1:                                    # learn from the step taken n_step days ago
                k0 = t - n_step + 1
                G = sum(gamma ** k * R[k0 + k] for k in range(n_step))
                if not done:
                    G = G + gamma ** n_step * Q[idx, s2[:, 0], s2[:, 1], s2[:, 2], s2[:, 3]].max(axis=1)
                update(S, A, k0, G)
            s = s2

        T = len(R)                                                 # month over: learn from the last few days
        for k0 in range(max(0, T - n_step + 1), T):
            G = sum(gamma ** k * R[k0 + k] for k in range(T - k0))
            update(S, A, k0, G)

        month = np.sum(R, axis=0)
        curve.append(month.sum()); curve_by_product.append(month)
        eps = max(eps_min, eps * eps_decay)
    return Q, np.array(curve), np.array(curve_by_product)


def make_policy(Q):
    """Greedy policy: policy(env, obs) -> orders for all products."""
    n = Q.shape[0]; idx = np.arange(n)

    def policy(env, obs):
        s = encode(env, obs)
        a = Q[idx, s[:, 0], s[:, 1], s[:, 2], s[:, 3]].argmax(axis=1)
        return to_orders(env, a)
    return policy


def policy_table(Q, onhand=1, daytype=0, heat=1):
    """For the heatmap: order multiplier for each product (rows) at each inventory-position level (columns)."""
    return MULTIPLIERS[Q[:, :, onhand, daytype, heat, :].argmax(axis=2)]
