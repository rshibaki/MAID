from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

models = client.models.list()

model_ids = [model.id for model in models.data]
if "gpt-4.1" in model_ids:
    print("✅ このAPIキーでは gpt-4.1 を使用できます。")
else:
    print("❌ このAPIキーでは gpt-4.1 は使用できません。")
    print("利用可能なモデル一覧:", model_ids)
