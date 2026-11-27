"""AlphaNova Season 2 Cycle 1 — High-Speed Autonomous Research Harness.

Provides:
1. Exact reproduction of official walkforward_full evaluation and scoring.
2. Fast in-memory evaluation for rapid hypothesis testing.
3. Automated logging to persistent experiment_ledger.csv.
4. City novelty and quality gate pre-checks.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure comp6s2_scientist_release is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from data_loader import list_periods, load_period
from evaluation import backtest, TRANSACTION_COST
from walkforward import compute_val_rows, ic_dispersion_corr, returns_to_equity
from city_tools import compute_gauge_fixed, compute_ic, compute_city, compare_to_cities
from predictor import Predictor

LEDGER_PATH = Path(r"C:\Users\Sachin\.gemini\antigravity\brain\52691b8b-f7c0-4a9b-bdb9-ffdba1143ce2\experiment_ledger.csv")
DATA_DIR = SCRIPT_DIR / "data" / "train"
GAUGE_DIR = DATA_DIR / "gauge"
UNIVERSE = "small"


class ResearchHarness:
    def __init__(self, data_dir=DATA_DIR, universe=UNIVERSE):
        self.data_dir = Path(data_dir)
        self.universe = universe
        self.periods = list_periods(str(self.data_dir), universe)
        self._cached_val_data = None
        self._cached_gauge = None
        self._cached_dispersion = None

    def preload_validation_cache(self, verbose=True):
        """Pre-cache validation slices for all 31 periods for ultra-fast evaluation."""
        if self._cached_val_data is not None:
            return

        t0 = time.time()
        if verbose:
            print(f"[Harness] Preloading validation cache for {len(self.periods)} periods...")

        cached_val = []
        cached_gauge = []
        cached_disp = []

        for p in self.periods:
            data = load_period(p, str(self.data_dir), self.universe)
            n_total = len(data["features"])
            n_val = compute_val_rows(n_total)
            n_train = n_total - n_val

            val_feat = data["features"].iloc[n_train:].copy()
            val_ret = data["returns"].iloc[n_train:].copy()
            val_tgt = data["target"].iloc[n_train:].copy()

            cached_val.append({
                "period": p,
                "n_train": n_train,
                "n_val": n_val,
                "val_features": val_feat,
                "val_returns": val_ret,
                "val_target": val_tgt,
            })

            # Gauge
            gauge_path = GAUGE_DIR / f"{p}.{self.universe}.gauge.npy"
            if gauge_path.exists():
                g_mat = np.load(str(gauge_path))
                cached_gauge.append(g_mat[:n_val])
            else:
                cached_gauge.append(None)

            # Dispersion
            if data.get("dispersion") is not None:
                d_val = data["dispersion"].iloc[n_train:].to_numpy(dtype=float)
                cached_disp.append(d_val[:n_val])
            else:
                cached_disp.append(None)

        self._cached_val_data = cached_val
        self._cached_gauge = cached_gauge
        self._cached_dispersion = cached_disp

        if verbose:
            print(f"[Harness] Validation cache ready in {time.time() - t0:.2f}s ({len(cached_val)} periods).")

    def run_walkforward(
        self,
        predictor_class,
        params=None,
        train_full=True,
        gauge_fix=True,
        verbose=False,
    ) -> dict:
        """Run full walk-forward validation matching official contest scorer."""
        self.preload_validation_cache(verbose=False)
        params = params or {}

        all_pnl = []
        all_gross_pnl = []
        all_gf = []
        all_disp = []
        per_period_sharpe = {}
        t_start = time.time()

        for idx, p_key in enumerate(self.periods):
            val_info = self._cached_val_data[idx]
            n_train = val_info["n_train"]
            n_val = val_info["n_val"]
            val_features = val_info["val_features"]
            val_returns = val_info["val_returns"]

            predictor = predictor_class(**params)

            # Check if predictor requires training
            if train_full and hasattr(predictor, "train"):
                # Load full training data for this period
                data = load_period(p_key, str(self.data_dir), self.universe)
                train_features = data["features"].iloc[:n_train]
                train_target = data["target"].iloc[:n_train]
                if hasattr(predictor, "generate_features"):
                    predictor.generate_features(data["features"])
                predictor.train(train_features, train_target)

            # Predict
            preds = predictor.predict(val_features)

            # Compute net PnL (with 5 bps transaction cost)
            net_pnl = backtest(preds, val_returns)
            all_pnl.append(net_pnl)

            # Compute gross PnL (without transaction cost)
            gross_pnl = (preds.shift(1).ffill() * val_returns).sum(axis=1)
            gross_pnl.iloc[0] = 0
            all_gross_pnl.append(gross_pnl)

            p_sharpe = float(net_pnl.mean() / net_pnl.std()) if net_pnl.std() > 0 else 0.0
            per_period_sharpe[p_key] = p_sharpe

            # Gauge fix
            if gauge_fix and self._cached_gauge[idx] is not None:
                g_mat = self._cached_gauge[idx]
                pred_vals = preds.values
                cols = preds.columns.tolist()
                numeric_order = sorted(cols, key=lambda c: int(c.split(".")[-1]))
                if cols != numeric_order:
                    pred_vals = preds[numeric_order].values
                pred_vals = pred_vals - pred_vals.mean(axis=1, keepdims=True)

                n_use = min(len(g_mat), len(pred_vals))
                gf = compute_gauge_fixed(pred_vals[:n_use], g_mat[:n_use])
                all_gf.append(gf)

                if self._cached_dispersion[idx] is not None:
                    all_disp.append(self._cached_dispersion[idx][:n_use])

        # Aggregate
        concat_net_pnl = pd.concat(all_pnl, ignore_index=True)
        concat_gross_pnl = pd.concat(all_gross_pnl, ignore_index=True)

        net_sharpe = float(concat_net_pnl.mean() / concat_net_pnl.std()) if concat_net_pnl.std() > 0 else 0.0
        gross_sharpe = float(concat_gross_pnl.mean() / concat_gross_pnl.std()) if concat_gross_pnl.std() > 0 else 0.0
        cost_drag = gross_sharpe - net_sharpe

        # Profitable periods win rate
        profitable_periods = sum(1 for s in per_period_sharpe.values() if s > 0)
        win_rate = (profitable_periods / len(per_period_sharpe)) * 100.0

        # Mean hourly L1 turnover
        # (gross_pnl - net_pnl) / TRANSACTION_COST
        total_cost_paid = (concat_gross_pnl - concat_net_pnl).sum()
        mean_turnover = float(total_cost_paid / (TRANSACTION_COST * len(concat_net_pnl)))

        metrics = {
            "net_sharpe": net_sharpe,
            "gross_sharpe": gross_sharpe,
            "cost_drag": cost_drag,
            "mean_turnover": mean_turnover,
            "win_rate_pct": win_rate,
            "profitable_periods": profitable_periods,
            "n_periods": len(self.periods),
            "n_obs": len(concat_net_pnl),
            "runtime_s": time.time() - t_start,
            "per_period_sharpe": per_period_sharpe,
        }

        # Gauge metrics
        if gauge_fix and all_gf:
            concat_gf = np.concatenate(all_gf, axis=0)
            ic_stats = compute_ic(concat_gf)
            city = compute_city(concat_gf)

            ic_disp = float("nan")
            if all_disp:
                ic_disp = ic_dispersion_corr(concat_gf[:, 0], np.concatenate(all_disp))

            metrics.update({
                "ic": ic_stats["ic_mean"],
                "ic_std": ic_stats["ic_std"],
                "ic_dispersion_corr": ic_disp,
                "concentration": ic_stats["concentration"],
                "city": city,
            })

            try:
                city_comp = compare_to_cities(city)
                metrics["city_novelty_deg"] = float(city_comp.iloc[0]["angular_distance_deg"])
                metrics["nearest_city"] = str(city_comp.iloc[0]["signal_id"])
            except Exception:
                metrics["city_novelty_deg"] = float("nan")
                metrics["nearest_city"] = "N/A"

        if verbose:
            print("=" * 65)
            print(f"WALK-FORWARD RESULTS: {predictor_class.__name__}")
            print(f"Net Sharpe (after 5bp fee): {net_sharpe:+.4f}")
            print(f"Gross Sharpe:               {gross_sharpe:+.4f} (Cost Drag: {cost_drag:.4f})")
            print(f"Mean Hourly L1 Turnover:    {mean_turnover:.2f}")
            print(f"Profitable Periods:         {profitable_periods}/{len(self.periods)} ({win_rate:.1f}%)")
            if "ic" in metrics:
                print(f"Mean IC:                    {metrics['ic']:+.4f} (Dispersion Corr: {metrics.get('ic_dispersion_corr', 0.0):+.4f})")
                print(f"City Novelty:               {metrics.get('city_novelty_deg', 0.0):.1f}° (Nearest: {metrics.get('nearest_city', 'N/A')})")
            print(f"Execution time:             {metrics['runtime_s']:.1f}s")
            print("=" * 65)

        return metrics

    def log_experiment(self, experiment_id: str, model_name: str, hypothesis: str, metrics: dict, notes: str = ""):
        """Record experiment to persistent experiment_ledger.csv."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "experiment_id": experiment_id,
            "model_name": model_name,
            "hypothesis": hypothesis,
            "net_sharpe": round(metrics.get("net_sharpe", 0.0), 6),
            "gross_sharpe": round(metrics.get("gross_sharpe", 0.0), 6),
            "cost_drag": round(metrics.get("cost_drag", 0.0), 6),
            "mean_turnover": round(metrics.get("mean_turnover", 0.0), 3),
            "ic": round(metrics.get("ic", float("nan")), 5),
            "ic_dispersion_corr": round(metrics.get("ic_dispersion_corr", float("nan")), 5),
            "win_rate_pct": round(metrics.get("win_rate_pct", 0.0), 1),
            "city_novelty_deg": round(metrics.get("city_novelty_deg", float("nan")), 2),
            "runtime_s": round(metrics.get("runtime_s", 0.0), 1),
            "notes": notes,
        }

        df_entry = pd.DataFrame([entry])
        if not LEDGER_PATH.exists():
            df_entry.to_csv(LEDGER_PATH, index=False)
        else:
            df_entry.to_csv(LEDGER_PATH, mode="a", header=False, index=False)
        print(f"[Ledger] Recorded experiment {experiment_id} ({model_name}) -> Net Sharpe: {entry['net_sharpe']:+.4f}")


if __name__ == "__main__":
    harness = ResearchHarness()
    harness.preload_validation_cache()
