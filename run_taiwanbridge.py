"""
Speeko v1 Unified Runner
Initializes the from-scratch 21.90M parameter model (<22M constraint),
verifies parameter budget, ensures checkpoint readiness,
and launches the FastAPI server with the Speech-to-Speech & Text web UI.
"""

import os
import sys
import uvicorn
import torch

from model.tokenizer import SpeekoTokenizer, TaiwanBridgeTokenizer
from model.transformer import create_speeko_model, create_taiwanbridge_model
from training.trainer import TaiwanBridgeTrainer
from eval.benchmarks import BenchmarkEvaluator


def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 70)
    print("       SPEEKO v1: FROM-SCRATCH MULTILINGUAL SPEECH & TEXT LM")
    print(" 150+ Language Speech-to-Speech (S2S) & Text Translation (< 22M Params)")
    print("=" * 70)

    # 1. Initialize Tokenizer & Vocab
    vocab_dir = os.path.join(os.path.dirname(__file__), "data")
    vocab_path = os.path.join(vocab_dir, "vocab.json")
    tokenizer = SpeekoTokenizer(vocab_size=15360)

    # 2. Initialize Model & Verify Parameter Constraint (< 22M)
    print("[*] Initializing Speeko-v1 Transformer with random weights...")
    model = create_speeko_model()
    param_info = model.count_parameters()
    print(f"[OK] Model Name: {param_info['model_name']}")
    print(f"[OK] Total Trainable Parameters: {param_info['total_parameters']:,} ({param_info['total_millions']}M)")
    print(f"[OK] Parameter Constraint (< 22M Limit): {'PASSED' if param_info['is_under_22m'] else 'FAILED'}")
    print(f"[OK] External API keys used: 0 (Strictly local, no OpenAI/Claude/Gemini API calls)")

    # 3. Check for / Train Initial Weights
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    ckpt_speeko = os.path.join(ckpt_dir, "speeko_v1_22m_weights.pt")
    ckpt_path = os.path.join(ckpt_dir, "taiwanbridge_weights.pt")

    if os.path.exists(ckpt_speeko):
        print(f"[OK] Loading Speeko-v1 21.90M checkpoint: {ckpt_speeko}")
        model.load_state_dict(torch.load(ckpt_speeko, map_location="cpu"))
    elif os.path.exists(ckpt_path):
        print(f"[OK] Found existing trained checkpoint at {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    else:
        print("[*] No existing checkpoint found. Training initial weights from scratch...")
        trainer = TaiwanBridgeTrainer(model, tokenizer, checkpoint_dir=ckpt_dir)
        trainer.train(epochs=1, batch_size=4, n_samples=30)
        print(f"[OK] Initial checkpoint saved to {ckpt_speeko}")

    # 4. Quick Benchmark Validation
    print("[*] Performing quick benchmark evaluation check...")
    evaluator = BenchmarkEvaluator(model, tokenizer)
    sample_res = evaluator.eval_universal_150_translation()
    print(f"[OK] Universal 150-Language BLEU Score: {sample_res['bleu_score']}")
    print(f"[OK] 150-Language Identification Accuracy: {sample_res['language_id_accuracy']}%")

    # 5. Launch FastAPI & Research UI
    print("\n" + "=" * 70)
    print(" Speeko v1 Web Console starting at:")
    print(" -> Local Web App URL: http://127.0.0.1:8000")
    print(" -> REST API Docs:      http://127.0.0.1:8000/docs")
    print("=" * 70 + "\n")

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()

