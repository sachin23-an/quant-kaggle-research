import os
from dotenv import load_dotenv
from pathlib import Path
from numerapi import NumerAPI
import json

load_dotenv("D:/Projects/Quant/quant-competition-research/.env")

pub = os.getenv("NUMERAI_PUBLIC_ID")
sec = os.getenv("NUMERAI_SECRET_KEY")
napi = NumerAPI(public_id=pub, secret_key=sec)

model_id = "d8184e3c-2526-40c4-b510-8e1c1aab64cc"

query = f"""
query {{
  model(modelId: "{model_id}") {{
    name
    submissions {{
      id
      round {{
        number
        openTime
        closeTime
        resolveTime
      }}
      filename
      insertedAt
      status
    }}
  }}
}}
"""
res = napi.raw_query(query)
subs = res.get("data", {}).get("model", {}).get("submissions", [])
print(f"Total verified submissions for model sachin23_an: {len(subs)}")
for s in subs:
    r = s.get("round", {})
    print(f"  Round {r.get('number')}: ID={s.get('id')} | Status={s.get('status')} | Uploaded={s.get('insertedAt')} | Resolves={r.get('resolveTime')}")
