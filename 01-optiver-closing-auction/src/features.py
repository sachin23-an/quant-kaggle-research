"""
Microstructure Feature Engineering for Optiver - Trading at the Close
Computes domain-specific order book, auction imbalance, and cross-sectional signals
with strict prevention of look-ahead bias and cross-temporal leakage.
"""

from typing import Dict, List, Optional, Union
import math


def compute_microstructure_features(row: Dict[str, float]) -> Dict[str, float]:
    """
    Extract single-row microstructure and order book features.

    Input features typically present:
    - bid_price, ask_price, bid_size, ask_size
    - wap (Weighted Average Price)
    - reference_price, near_price, far_price
    - imbalance_size, matched_size, imbalance_buy_sell_flag
    - seconds_in_bucket
    """
    feats = {}

    bid_p = row.get("bid_price", 0.0)
    ask_p = row.get("ask_price", 0.0)
    bid_s = row.get("bid_size", 0.0)
    ask_s = row.get("ask_size", 0.0)
    wap = row.get("wap", 0.0)
    ref_p = row.get("reference_price", 0.0)
    imb_s = row.get("imbalance_size", 0.0)
    mat_s = row.get("matched_size", 0.0)
    flag = row.get("imbalance_buy_sell_flag", 0.0)
    sec = row.get("seconds_in_bucket", 0.0)

    # 1. Spread and Depth Ratios
    spread = ask_p - bid_p
    mid_price = (ask_p + bid_p) / 2.0 if (ask_p + bid_p) > 0 else 0.0
    total_depth = bid_s + ask_s
    depth_imbalance = (bid_s - ask_s) / total_depth if total_depth > 0 else 0.0

    feats["spread"] = spread
    feats["mid_price"] = mid_price
    feats["total_depth"] = total_depth
    feats["order_book_depth_imbalance"] = depth_imbalance

    # 2. Micro-Price Divergence
    # Micro-price weights prices inversely by queue length (Stoikov 2018)
    micro_price = (bid_p * ask_s + ask_p * bid_s) / total_depth if total_depth > 0 else mid_price
    feats["micro_price"] = micro_price
    feats["micro_price_to_wap_spread"] = (micro_price - wap) if wap > 0 else 0.0

    # 3. Auction vs Continuous Book Imbalance
    total_auction_vol = imb_s + mat_s
    feats["auction_matched_ratio"] = imb_s / (mat_s + 1.0)
    feats["auction_imbalance_ratio"] = imb_s / (total_auction_vol + 1.0)
    feats["signed_auction_imbalance"] = imb_s * flag

    # 4. Reference Price Dislocation
    feats["wap_ref_spread"] = (wap - ref_p) / ref_p if ref_p > 0 else 0.0

    # 5. Near & Far Auction Price Indicators (Active after 300 seconds)
    near_p = row.get("near_price", 0.0)
    far_p = row.get("far_price", 0.0)
    feats["has_near_far"] = 1.0 if sec >= 300.0 else 0.0
    feats["near_wap_spread"] = (near_p - wap) / wap if (wap > 0 and sec >= 300 and near_p > 0) else 0.0
    feats["far_wap_spread"] = (far_p - wap) / wap if (wap > 0 and sec >= 300 and far_p > 0) else 0.0

    # 6. Auction Time Regime (Non-linear urgency towards closing cross)
    feats["seconds_in_bucket"] = sec
    feats["late_auction_urgency"] = 1.0 if sec >= 480.0 else 0.0  # Final 60 seconds
    feats["auction_time_fraction"] = sec / 540.0

    return feats


def compute_cross_sectional_neutralization(
    records: List[Dict[str, float]],
    feature_names: List[str],
) -> List[Dict[str, float]]:
    """
    Market-neutralize features cross-sectionally for a single time bucket (e.g. at one date_id & seconds_in_bucket).
    Removes broad-market auction drift, leaving pure idiosyncratic stock signal.
    """
    n = len(records)
    if n == 0:
        return records

    # Calculate cross-sectional means for each feature
    means = {}
    for feat in feature_names:
        vals = [r.get(feat, 0.0) for r in records]
        means[feat] = sum(vals) / n

    # Subtract market cross-sectional mean
    neutralized = []
    for r in records:
        r_copy = dict(r)
        for feat in feature_names:
            r_copy[f"{feat}_market_neutral"] = r.get(feat, 0.0) - means[feat]
        neutralized.append(r_copy)

    return neutralized


if __name__ == "__main__":
    print("Testing Optiver feature engineering...")
    sample_row = {
        "stock_id": 1,
        "date_id": 10,
        "seconds_in_bucket": 360,
        "imbalance_size": 25000.0,
        "imbalance_buy_sell_flag": 1,
        "reference_price": 100.5,
        "matched_size": 150000.0,
        "far_price": 101.2,
        "near_price": 100.8,
        "bid_price": 100.4,
        "bid_size": 5000.0,
        "ask_price": 100.6,
        "ask_size": 3000.0,
        "wap": 100.525,
    }

    feats = compute_microstructure_features(sample_row)
    for k, v in feats.items():
        print(f"  {k}: {v:.6f}")

    assert "micro_price" in feats
    assert "order_book_depth_imbalance" in feats
    print("\nFeature extraction self-test passed successfully!")
