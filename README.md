# TaiwanBridge: A Compact From-Scratch Language Model for Multilingual Understanding

**TaiwanBridge** is a compact, specialized language model designed to understand multilingual and code-switched communication involving **Traditional Chinese (繁體中文)** and **English**, with a particular focus on communication patterns relevant to Taiwan.

Instead of developing another generic conversational chatbot, TaiwanBridge specializes in converting unstructured, mixed-language communication into **structured semantic representations** (Event, Time/Date, Action, Status, Entities, and Intent) without relying on external LLM inference APIs (such as OpenAI, Gemini, or Claude) or pretrained weights.

---

## 1. Key Architectural Specifications

- **Total Trainable Parameters**: **44,071,956** (~44.07 Million Parameters)
- **Constraint Compliance**: Strictly `< 50 Million Parameters` (target: 40–45M)
- **Zero API Dependency**: 100% self-contained local PyTorch Transformer
- **Vocabulary Size**: 12,288 tokens (Traditional Chinese Hanzi + English BPE/subwords + structural delimiters)
- **Transformer Configuration**:
  - Layers: 10
  - Hidden Dimension ($d_{model}$): 512
  - Attention Heads: 8 (Head dimension: 64)
  - Feed-Forward Dimension ($d_{ff}$): 2048
  - Positional Encoding: Rotary Positional Embeddings (RoPE)
  - Normalization: Pre-LayerNorm
  - Activation: GELU
  - Multi-Task Heads:
    1. Causal Next-Token LM Head ($12,288 \times 512$)
    2. Code-Switch Token Classifier Head ($512 \times 4$)
    3. Semantic Slot Sequence Tagger Head ($512 \times 16$)

### Parameter Breakdown

| Component | Dimensions | Parameters | % of Total |
| :--- | :--- | :--- | :--- |
| `token_embedding` | $12,288 \times 512$ | 6,291,456 | 14.28% |
| `10× Transformer Layers` | $10 \times [\text{Attn}(1.05\text{M}) + \text{FFN}(2.10\text{M})]$ | 31,477,760 | 71.42% |
| `final_norm` | 512 | 1,024 | < 0.01% |
| `lm_head` (Untied) | $512 \times 12,288$ | 6,291,456 | 14.28% |
| `csw_head` (Code-Switch) | $512 \times 4$ | 2,052 | < 0.01% |
| `semantic_head` (Slot) | $512 \times 16$ | 8,208 | 0.02% |
| **Total Trainable Parameters** | **Under 50M Constraint** | **44,071,956** | **100.0%** |

---

## 2. Core Capabilities

### A. Code-Switch Detection (ZH / EN)
- Fine-grained token-level language classification: Traditional Chinese (`ZH`), English (`EN`), Numeric (`NUM`), and Punctuation/Special (`PUNCT`).
- Code-switch boundary identification (where language switches occur within an utterance).
- Language ratio and switch density metrics.

### B. Structured Semantic Frame Extraction
Transforms unstructured mixed messages into structured JSON representations:
- **Event**: Meeting & Review, Software Deployment, Dining & Hospitality, Travel & Commute, Milestone & Deadline.
- **Time/Date**: Normalized expressions (e.g., 明天下午2點 $\rightarrow$ `Tomorrow Afternoon [明天下午2點]`).
- **Action**: Prepare Slides, Code Review, Deploy, Make Reservation, Confirm, Sync.
- **Status**: Confirmed, Urgent / High Priority, Pending, Blocked, In Progress, Completed.
- **Named Entities**: Locations (台北101, 捷運市政府站, 新竹科學園區), Persons/Roles (PM, Tech Lead, Alice), Artifacts (slides, PR #402, docker image).

---

## 3. Evaluation & GIBC TECH Benchmarks

TaiwanBridge includes an evaluation harness covering:
1. **GIBC TECH Benchmarks**:
   - **HellaSwag**: Commonsense continuation log-likelihood scoring.
   - **ARC-Easy**: Grade-school science reasoning QA.
   - **PIQA**: Physical interaction commonsense QA.
   - **WinoGrande**: Commonsense pronoun / coreference resolution.
   - **WikiText-103**: Autoregressive language modeling perplexity (PPL).
2. **Specialized TaiwanBridge Benchmarks**:
   - Traditional Chinese Understanding (TCU).
   - Chinese-English Code-Switching Detection F1.
   - Semantic Frame Extraction (SFE) Slot F1.
   - Contextual Ambiguity Resolution (CAR).

---

## 4. Advanced Black & White Research UI

- Publication-grade monochrome research interface with high-contrast typography (IBM Plex Mono & Inter).
- Interactive **Code-Switch Visualizer** with token badges and switch-point indicators.
- **Structured Semantic Card** and live JSON inspector with one-click clipboard copying.
- **Architecture Parameter Explorer** with real-time tensor shape inspections.
- **GIBC TECH Benchmark Dashboard** with live evaluation triggers.
- **Theme Switcher**: Instant toggle between *Noir Black* and *Paper White*.

---

## 5. Quick Start & Execution

### Starting the Research Console & Server

```powershell
python run_taiwanbridge.py
```

Open your browser at:
```
http://127.0.0.1:8000
```

### Running Standalone Benchmarks

```powershell
python eval/benchmarks.py
```

### Running Model Training from Scratch

```powershell
python training/trainer.py
```
