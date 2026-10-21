"""
cafe_pro.py - The Smart Café, scaled up.

Many perishable products, each with its own shelf life, delivery delay, price and cost.
Demand follows the weekday and the weather, and is linked across products
(a busy day is busy for everything). Storage is shared per zone (fridge / dry store)
instead of a fixed cap per product.

ALL NUMBERS BELOW ARE ILLUSTRATIVE PLACEHOLDERS - replace them with real café data.
"""
import numpy as np

# ------------------------------------------------------------------
# Product catalogue (edit freely - the whole app reads from this list)
#   shelf  : days a unit can still be sold (1 = must be sold the day it arrives)
#   lead   : delivery delay in days (0 = same-morning delivery)
#   mean   : typical units sold on a normal day
#   price  : sales value per unit used     cost: purchase cost per unit
#   hold   : holding cost per leftover unit per day
#   temp_sens / weekend_sens : how strongly demand reacts to heat / the weekend
# ------------------------------------------------------------------
PRODUCTS = [
    dict(name="Coffee beans", emoji="☕", zone="dry",    shelf=60,  lead=3, mean=8,  price=12.0, cost=6.0, hold=0.10, temp_sens=0.0, weekend_sens=0.3),
    dict(name="Milk",         emoji="🥛", zone="fridge", shelf=5,   lead=1, mean=22, price=4.0,  cost=1.5, hold=0.15, temp_sens=0.2, weekend_sens=0.5),
    dict(name="Oat milk",     emoji="🌱", zone="fridge", shelf=7,   lead=2, mean=7,  price=6.0,  cost=2.5, hold=0.15, temp_sens=0.1, weekend_sens=0.5),
    dict(name="Cream",        emoji="🍦", zone="fridge", shelf=4,   lead=1, mean=6,  price=5.0,  cost=2.0, hold=0.15, temp_sens=0.0, weekend_sens=0.5),
    dict(name="Sugar",        emoji="🍬", zone="dry",    shelf=365, lead=3, mean=10, price=1.0,  cost=0.4, hold=0.02, temp_sens=0.0, weekend_sens=0.3),
    dict(name="Cakes",        emoji="🍰", zone="fridge", shelf=2,   lead=1, mean=10, price=14.0, cost=5.0, hold=0.40, temp_sens=0.0, weekend_sens=1.2),
    dict(name="Pastries",     emoji="🥐", zone="dry",    shelf=2,   lead=1, mean=16, price=7.0,  cost=2.5, hold=0.25, temp_sens=0.0, weekend_sens=0.8),
    dict(name="Sandwiches",   emoji="🥪", zone="fridge", shelf=1,   lead=0, mean=12, price=18.0, cost=7.0, hold=0.40, temp_sens=0.0, weekend_sens=0.6),
    dict(name="Cold drinks",  emoji="🥤", zone="fridge", shelf=90,  lead=2, mean=18, price=8.0,  cost=3.0, hold=0.15, temp_sens=0.8, weekend_sens=0.8),
    dict(name="Fresh juice",  emoji="🍋", zone="fridge", shelf=3,   lead=1, mean=9,  price=12.0, cost=4.0, hold=0.30, temp_sens=0.9, weekend_sens=0.7),
]
for _p in PRODUCTS:
    _p.setdefault("spoil", round(1.25 * _p["cost"], 2))      # waste cost on top of purchase cost
    _p.setdefault("stockout", round(1.2 * _p["price"], 2))   # penalty per unhappy customer

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
WEEKDAY_FACTOR = np.array([0.85, 0.90, 0.95, 1.00, 1.20, 1.40, 1.20])
BASE_TEMP = 28.0
ZONE_CAPACITY = {"fridge": 350, "dry": 400}      # units, before the storage slider


class CafePro:
    def __init__(self, products=None, days=30, temp_mean=28.0, delay_prob=0.10,
                 demand_scale=1.0, busy_chance=0.20, storage_scale=1.0,
                 spoil_mult=1.0, stockout_mult=1.0, seed=None):
        self.P = [dict(p) for p in (products or PRODUCTS)]
        self.n = len(self.P)
        self.names = [p["name"] for p in self.P]
        self.emojis = [p["emoji"] for p in self.P]
        self.mean = np.array([p["mean"] for p in self.P], dtype=float) * demand_scale
        self.days = days
        self.temp_mean = temp_mean
        self.delay_prob = delay_prob
        self.busy_chance = busy_chance
        self.spoil_mult = spoil_mult
        self.stockout_mult = stockout_mult
        self.cap = {z: int(c * storage_scale) for z, c in ZONE_CAPACITY.items()}
        self.demand_scale = demand_scale
        self.rng = np.random.default_rng(seed)
        self.max_lead = max(p["lead"] for p in self.P)

    # ---------------- helpers ----------------
    def _draw_temp(self):
        return float(self.rng.normal(self.temp_mean, 3.5))

    def _obs(self):
        return dict(on_hand=np.array([s.sum() for s in self.stock]),
                    in_transit=np.array([q.sum() for q in self.pipe]),
                    weekday=self.weekday, temp=self.temp, day=self.day)

    # ---------------- API ----------------
    def reset(self):
        self.day = 0
        self.weekday = int(self.rng.integers(7))
        self.temp = self._draw_temp()
        self.stock, self.pipe = [], []
        for p, m in zip(self.P, self.mean):
            s = np.zeros(p["shelf"]); s[0] = round(1.5 * m) if p["lead"] > 0 else 0
            q = np.zeros(self.max_lead + 2)
            for d in range(1, p["lead"] + 1):
                q[d] = round(m)                                  # shipments already on the road
            self.stock.append(s); self.pipe.append(q)
        return self._obs()

    def _demand(self):
        wf = WEEKDAY_FACTOR[self.weekday]
        busy = self.rng.uniform(1.2, 1.6) if self.rng.random() < self.busy_chance else 1.0
        noise = self.rng.lognormal(0, 0.10)                      # shared across products
        d = np.zeros(self.n)
        for i, p in enumerate(self.P):
            lam = (self.mean[i] * (1 + p["weekend_sens"] * (wf - 1))
                   * (1 + p["temp_sens"] * (self.temp - BASE_TEMP) / 10) * busy * noise)
            d[i] = self.rng.poisson(max(lam, 0.1))
        return d

    def step(self, orders):
        orders = np.maximum(np.round(np.asarray(orders, dtype=float)), 0)
        n = self.n
        sold = np.zeros(n); spoiled = np.zeros(n); stockout = np.zeros(n)
        overflow = np.zeros(n); holding = np.zeros(n); order_cost = np.zeros(n)

        # 1. place orders (each may be delivered a day late) and collect today's arrivals
        arrivals = np.zeros(n)
        for i, p in enumerate(self.P):
            delay = p["lead"] + (1 if p["lead"] > 0 and self.rng.random() < self.delay_prob else 0)
            self.pipe[i][delay] += orders[i]
            arrivals[i] = self.pipe[i][0]
            order_cost[i] = orders[i] * p["cost"]

        # 2. shared storage: deliveries that don't fit are refused (still paid for)
        for zone, cap in self.cap.items():
            idx = [i for i in range(n) if self.P[i]["zone"] == zone]
            space = cap - sum(self.stock[i].sum() for i in idx)
            for i in idx:
                take = min(arrivals[i], max(space, 0))
                overflow[i] = arrivals[i] - take
                self.stock[i][0] += take
                space -= take

        # 3. today's demand, sold oldest-first (FIFO)
        demand = self._demand()
        for i in range(n):
            arr = self.stock[i]
            sold[i] = min(arr.sum(), demand[i])
            stockout[i] = demand[i] - sold[i]
            rev = arr[::-1]
            take = np.clip(sold[i] - (np.cumsum(rev) - rev), 0, rev)
            arr[:] = (rev - take)[::-1]

        # 4. close the shop: holding cost, expiry, ageing, move the delivery pipeline
        for i, p in enumerate(self.P):
            arr = self.stock[i]
            holding[i] = arr.sum() * p["hold"]
            spoiled[i] = arr[-1]                                  # reached the end of its shelf life
            new = np.zeros_like(arr); new[1:] = arr[:-1]; self.stock[i] = new
            self.pipe[i] = np.append(self.pipe[i][1:], 0.0)

        price = np.array([p["price"] for p in self.P])
        spoil = np.array([p["spoil"] for p in self.P]) * self.spoil_mult
        pen = np.array([p["stockout"] for p in self.P]) * self.stockout_mult
        profit = sold * price - order_cost - holding - spoiled * spoil - stockout * pen

        # 5. tomorrow
        self.day += 1
        self.weekday = (self.weekday + 1) % 7
        self.temp = self._draw_temp()
        done = self.day >= self.days
        info = dict(order=orders, demand=demand, sold=sold, spoiled=spoiled, stockout=stockout,
                    overflow=overflow, holding=holding, order_cost=order_cost, profit=profit)
        return profit, self._obs(), done, info
