import os
from dotenv import load_dotenv

load_dotenv("D:/Projects/Quant/quant-competition-research/.env")
from numerapi import NumerAPI

pub = os.getenv("NUMERAI_PUBLIC_ID")
sec = os.getenv("NUMERAI_SECRET_KEY")
model_name = os.getenv("NUMERAI_MODEL_NAME", "sachin23_an")

napi = NumerAPI(public_id=pub, secret_key=sec)

current_round = napi.get_current_round()
is_new_1h = napi.check_new_round(hours=1)
is_new_2h = napi.check_new_round(hours=2)

models = napi.get_models()
model_id = models.get(model_name)

# Query recent rounds info
query = """
query {
  rounds(tournament: 8, limit: 3) {
    number
    openTime
    closeTime
    resolveTime
  }
}
"""
rounds_data = napi.raw_query(query).get("data", {}).get("rounds", [])

print("=" * 60)
print(f"Current Numerai Round: {current_round}")
print(f"Has new round opened in last 1 hour: {is_new_1h}")
print(f"Has new round opened in last 2 hours: {is_new_2h}")
print(f"Model: {model_name} (ID: {model_id})")
print("-" * 60)
print("Recent Rounds:")
for r in rounds_data:
    print(f"  Round {r.get('number')}: Open {r.get('openTime')} | Close {r.get('closeTime')}")
print("=" * 60)
