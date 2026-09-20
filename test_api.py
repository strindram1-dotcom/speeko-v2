import urllib.request
import json
import sys

base = "http://127.0.0.1:8000"

print("--- Testing TaiwanBridge Local API ---")

# 1. Model Info
req = urllib.request.urlopen(f"{base}/api/model/info")
info = json.loads(req.read().decode("utf-8"))
print("[✓] Model Info:")
print(f"    Name: {info['model_name']}")
print(f"    Params: {info['actual_parameters']:,} ({info['total_millions']}M)")
print(f"    Under 50M: {info['is_under_50m']}")
print(f"    External API Used: {info['external_api_used']}")
assert info["is_under_50m"] is True, "Model parameters must be < 50M!"
assert info["actual_parameters"] < 50_000_000, "Parameter limit exceeded!"

# 2. Analyze
payload = json.dumps({"text": "我們明天下午2點在台北101開Q3 project review meeting，記得把slides準備好。"}).encode("utf-8")
req = urllib.request.Request(f"{base}/api/analyze", data=payload, headers={"Content-Type": "application/json"})
resp = urllib.request.urlopen(req)
res = json.loads(resp.read().decode("utf-8"))
print("\n[✓] Analyze Endpoint:")
print("    Event:", res["structured_representation"]["event"])
print("    Action:", res["structured_representation"]["action"])
print("    Time:", res["structured_representation"]["time"]["normalized"])
print("    Status:", res["structured_representation"]["status"])
print("    Entities:", [e["text"] for e in res["structured_representation"]["entities"]])
print("    Classification:", res["code_switch_analysis"]["classification"])
print("    Switch Points:", res["code_switch_analysis"]["switch_points"])

# 3. Generation
gen_payload = json.dumps({"prompt": "我們明天在台北開會，", "max_tokens": 12, "temperature": 0.7}).encode("utf-8")
req = urllib.request.Request(f"{base}/api/generate", data=gen_payload, headers={"Content-Type": "application/json"})
resp = urllib.request.urlopen(req)
gen_res = json.loads(resp.read().decode("utf-8"))
print("\n[✓] Autoregressive Generation:")
print(f"    Prompt: {gen_res['prompt']}")
print(f"    Continuation: {gen_res['continuation']}")
print(f"    Latency: {gen_res['latency_ms']} ms")

# 4. Benchmarks
req = urllib.request.urlopen(f"{base}/api/benchmarks")
bench = json.loads(req.read().decode("utf-8"))
print("\n[✓] GIBC TECH Benchmarks:")
print(f"    HellaSwag: {bench['gibc_tech_benchmarks']['hellaswag']['accuracy']}%")
print(f"    ARC-Easy: {bench['gibc_tech_benchmarks']['arc_easy']['accuracy']}%")
print(f"    PIQA: {bench['gibc_tech_benchmarks']['piqa']['accuracy']}%")
print(f"    WinoGrande: {bench['gibc_tech_benchmarks']['winogrande']['accuracy']}%")
print(f"    WikiText-103 PPL: {bench['gibc_tech_benchmarks']['wikitext_103']['perplexity']}")
print(f"    Code-Switch F1: {bench['taiwanbridge_benchmarks']['code_switch_and_semantics']['code_switch_detection_f1']}%")
print(f"    Semantic Slot F1: {bench['taiwanbridge_benchmarks']['code_switch_and_semantics']['semantic_slot_f1']}%")

# 5. UI HTML
req = urllib.request.urlopen(f"{base}/")
html = req.read().decode("utf-8")
print(f"\n[✓] UI Static Index: 200 OK, Received {len(html)} bytes")
print("\nALL VERIFICATIONS PASSED SUCCESSFULLY!")
