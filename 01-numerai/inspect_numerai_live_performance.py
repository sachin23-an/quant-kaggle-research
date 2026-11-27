from numerapi import NumerAPI
import json
import pandas as pd

napi = NumerAPI()
model_name = "sachin23_an"

query = f"""
query {{
  v3UserProfile(modelName: "{model_name}") {{
    id
    username
    startDate
    isActive
    latestRanks {{
      corr
      corr20V2
      corr20d
      corr60
      fncV3
      mmc
    }}
    roundModelPerformances {{
      roundNumber
      roundOpenTime
      roundResolved
      corr20V2
      corr20d
      corr
      fncV3
      mmc
    }}
  }}
}}
"""
res = napi.raw_query(query)
data = res.get("data", {}).get("v3UserProfile", {})

print("=" * 70)
print(f"NUMERAI LIVE MODEL REPORT: {model_name}")
print("=" * 70)
print(f"Username: {data.get('username')}")
print(f"Start Date: {data.get('startDate')}")
print(f"Is Active: {data.get('isActive')}")
print("\nLatest Ranks:")
print(json.dumps(data.get("latestRanks"), indent=2))

rounds = data.get("roundModelPerformances", [])
print(f"\nTotal Round Performances Recorded: {len(rounds)}")
if rounds:
    df_rounds = pd.DataFrame(rounds)
    print(df_rounds.to_string(index=False))
else:
    print("No resolved rounds yet (model recently registered, awaiting round resolution window).")
print("=" * 70)
