from numerapi import NumerAPI
import json

napi = NumerAPI()
model_id = "d8184e3c-2526-40c4-b510-8e1c1aab64cc"

try:
    intra = napi.intra_round_scores(model_id)
    print("Intra round scores:", intra)
except Exception as e:
    print("Intra round error:", e)

query = """
query {
  v3UserProfile(modelName: "sachin23_an") {
    username
    startDate
    historicalNetNmrEarnings
  }
}
"""
print(napi.raw_query(query))
