"""
Project 3: Critical-mineral price forecast and vendor scorecard for sourcing decisions
Stack: Python ML (forecast), PySpark + SQL (vendor scorecard), Azure Blob upload (optional), Power BI export
Data: synthetic monthly lithium-carbonate-like price series and vendor deliveries
"""
import os
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error as mape
from pyspark.sql import SparkSession

OUT = os.environ.get("OUT", "p3_out"); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(11)

# ---------- 1. Price series (USD/tonne), 10 years monthly with cycles + shocks ----------
n = 120
t = np.arange(n)
price = 15000 + 6000 * np.sin(2 * np.pi * t / 36) + np.cumsum(rng.normal(0, 450, n)) * 0.6 + rng.normal(0, 600, n)
price = np.clip(price, 6000, None)
demand_idx = 100 + 0.4 * t + 8 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 2, n)   # proxy for EV demand
df = pd.DataFrame({"month": pd.date_range("2016-01-01", periods=n, freq="MS"), "price": price, "demand_idx": demand_idx})
for k in (1, 2, 3, 6, 12): df[f"lag{k}"] = df.price.shift(k)
df["mom"] = df.price.pct_change(); df["roll3"] = df.price.rolling(3).mean().shift(1)
df["demand_lag1"] = df.demand_idx.shift(1)
df["target"] = df.price.shift(-3)                     # price 3 months ahead
df = df.dropna().reset_index(drop=True)
feats = ["price", "lag1", "lag2", "lag3", "lag6", "lag12", "mom", "roll3", "demand_lag1"]

# ---------- 2. Walk-forward validation vs naive ----------
split = int(len(df) * 0.7)
preds, naive, actual = [], [], []
for i in range(split, len(df)):
    m = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0)
    m.fit(df.loc[:i - 4, feats], df.loc[:i - 4, "target"])    # no leakage: only targets already known
    preds.append(m.predict(df.loc[[i], feats])[0]); naive.append(df.loc[i, "price"]); actual.append(df.loc[i, "target"])
mape_ml, mape_naive = mape(actual, preds), mape(actual, naive)

# ---------- 3. Buying policy backtest: forward-buy when forecast > spot by threshold ----------
# Each month needs 100 t. Naive = buy spot at month t+3. Policy: if model says price will rise >3%, lock price now.
spot_now = np.array(naive); future = np.array(actual); fc = np.array(preds)
lock = fc > spot_now * 1.03
cost_spot = (future * 100).sum()
cost_policy = np.where(lock, spot_now, future).sum() * 100
saving_pct = (cost_spot - cost_policy) / cost_spot * 100
saving_usd = cost_spot - cost_policy

# ---------- 4. Vendor scorecard with PySpark + SQL ----------
vend = pd.DataFrame({"vendor": rng.choice([f"V{i}" for i in range(1, 9)], 2000)})
vend["ordered_t"] = rng.integers(20, 120, 2000)
vend["on_time"] = rng.random(2000) < vend.vendor.map(dict(zip([f"V{i}" for i in range(1, 9)], [.95, .9, .82, .7, .88, .93, .76, .85])))
vend["quality_reject_pct"] = np.abs(rng.normal(vend.vendor.map(dict(zip([f"V{i}" for i in range(1, 9)], [.8, 1.1, 1.6, 2.4, 1.2, .9, 2.0, 1.4]))), .3))
vend["price_usd_t"] = rng.normal(15000 * vend.vendor.map(dict(zip([f"V{i}" for i in range(1, 9)], [1.03, 1.0, .985, .965, 1.0, 1.04, .97, .99]))), 400)
spark = SparkSession.builder.master("local[2]").appName("vendors").config("spark.ui.enabled", "false").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
spark.createDataFrame(vend.assign(on_time=vend.on_time.astype(int))).createOrReplaceTempView("deliveries")
score = spark.sql("""
    SELECT vendor,
           ROUND(AVG(on_time)*100,1)                                   AS on_time_pct,
           ROUND(AVG(quality_reject_pct),2)                            AS reject_pct,
           ROUND(SUM(ordered_t*price_usd_t)/SUM(ordered_t),0)          AS wavg_price,
           -- landed cost adds the cost of rejected material
           -- assumption: each 10 pts of late deliveries adds 5% (expediting, buffer stock, line-stop risk)
           ROUND(SUM(ordered_t*price_usd_t)/SUM(ordered_t) / (1 - AVG(quality_reject_pct)/100)
                 * (1 + 0.5*(1-AVG(on_time))), 0) AS effective_price
    FROM deliveries GROUP BY vendor ORDER BY effective_price
""").toPandas()
spark.stop()
cheapest_quote = score.loc[score.wavg_price.idxmin()]
best_effective = score.loc[score.effective_price.idxmin()]
gap = cheapest_quote.effective_price - best_effective.effective_price

df[["month", "price"]].to_csv(f"{OUT}/price_history.csv", index=False)
score.to_csv(f"{OUT}/vendor_scorecard.csv", index=False)

# ---------- 5. Optional: push outputs to Azure Blob (feeds Power BI via Synapse/ADLS) ----------
def upload_to_azure(conn_str, container="sourcing"):
    from azure.storage.blob import BlobServiceClient
    svc = BlobServiceClient.from_connection_string(conn_str)
    for f in os.listdir(OUT):
        with open(os.path.join(OUT, f), "rb") as fh:
            svc.get_blob_client(container, f).upload_blob(fh, overwrite=True)
if os.environ.get("AZURE_STORAGE_CONNECTION_STRING"):
    upload_to_azure(os.environ["AZURE_STORAGE_CONNECTION_STRING"])

print(f"3-month-ahead MAPE  ML: {mape_ml*100:.1f}%   naive: {mape_naive*100:.1f}%   improvement: {(1-mape_ml/mape_naive)*100:.0f}%")
print(f"Backtest months: {len(actual)} | forward-lock months: {lock.sum()}")
print(f"Spend (100 t/month): spot ${cost_spot/1e6:.1f}M vs policy ${cost_policy/1e6:.1f}M -> saving {saving_pct:.1f}% (${saving_usd/1e6:.2f}M)")
print("\nVendor scorecard:\n", score.to_string(index=False))
print(f"\nCheapest quote {cheapest_quote.vendor} effective ${cheapest_quote.effective_price:,.0f}/t vs best effective {best_effective.vendor} ${best_effective.effective_price:,.0f}/t (gap ${gap:,.0f}/t)")
