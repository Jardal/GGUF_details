# GGUF_details
Get interesting informations about GGUF from local disk or HTTP without downloading to disk

AI slop continues... Sorry for mix czech-english :)

# GGUF Metadata & Documentation Extractor

A powerful, self-contained Python utility designed to extract comprehensive metadata, architecture configurations, chat templates, and advanced behavioral hints directly from any **GGUF** model file—**without requiring original `safetensors`, configuration files, or documentation.**

## 🚀 Key Features

* **Zero Heavy Dependencies:** Uses only Python's standard library (`struct`, `mmap`, `re`, `urllib`), meaning you don't need PyTorch, Transformers, or `llama.cpp` installed to inspect models.
* **Memory-Mapped (`mmap`) Parsing:** Efficiently maps large GGUF files (tens of gigabytes) directly into virtual memory for instant metadata extraction without overloading your RAM.
* **Automatic Data Decoding:** Converts raw byte/ASCII arrays into clean, readable strings (model names, bases, tags).
* **Deep MoE (Mixture of Experts) Analysis:** Automatically detects MoE architectures, total/active experts, shared experts, and routing parameters.
* **Thinking / Reasoning Mode Detection:** Scans both KV metadata, tokenizers, and massive Jinja chat templates (including complex >200kB custom templates like *DavidAU Turbo-Brilliance*) for XML thinking tags (`<think>`, `<thought>`), reasoning mode switches (`enable_think`, `reasoning_effort`), and prompt patterns.
* **Chat Template Extraction:** Exports the embedded Jinja chat template directly into a clean `.jinja` file.
* **Structured Markdown Report:** Generates a comprehensive `.md` manifest file and prints it to the console.
* **Offline Encyclopedic Explanations:** Translates numerical GGUF constants (quantization types like `IQ4_NL`, `Q4_K_M`, token types) into human-readable definitions.

---

## 🛠️ Installation

No installation or `pip install` required! Just make sure you have **Python 3.8+** installed.

```bash
git clone [https://github.com/your-username/gguf-doc-extractor.git](https://github.com/your-username/gguf-doc-extractor.git)
cd gguf-doc-extractor

```

---

## 💻 Usage

Run the script from your terminal and pass either a path to a local GGUF file or a direct HTTP/HTTPS URL:

```bash
python gguf_doc_extractor.py path/to/your_model.gguf

```

Or via URL (downloads header metadata via HTTP Range):

```bash
python gguf_doc_extractor.py [https://huggingface.co/organization/model-GGUF/resolve/main/model-Q4_K_M.gguf](https://huggingface.co/organization/model-GGUF/resolve/main/model-Q4_K_M.gguf)

```

### Outputs Generated:

1. **`<model_name>.md`**: A structured documentation manifest containing summary specs, MoE details, thinking capabilities, and categorized metadata tables.
2. **`<model_name>.jinja`**: The raw extracted chat template.

---

## 📋 Example Output Structure (`.md`)

```markdown
# Průvodní list a dokumentace modelu: `NeoHorse-1-9B-Uncensored`

## Souhrnný přehled modelu
> **NeoHorse 1 9 Uncensored** `Dense` - **9.0B** `(26 → MOST_IQ4_XS (Importance Matrix 4-bit Extra Small))`
> * **Původ:** Qwen3.5 9B
> * **Zaměření:** agentic
> * **Slovník:** Předzpracování: `qwen35` | Velikost slovníku: `151646` tokenů
> * **Detekované myslící (Thinking/Reasoning) režimy:** **Konfigurovatelné Thinking Přepínače (DavidAU/Custom)**

## Detailní rozbor Thinking / Reasoning funkcionality
- Šablona podporuje dynamické přepínání režimů myslení skrze proměnné: `enable_think`, `reasoning_effort`

## Architektonická analýza
- **Základní typ architektury:** `qwen2` (Dense)
- **Maximální kontextové okno:** `32768` tokenů
...

```

---

## 🧠 How It Works

GGUF is a self-describing binary format. This script reads the binary KV (Key-Value) store directly from the file header. By leveraging `mmap`, it bypasses file-size limitations and parses thousands of metadata items (such as the entire tokenizer vocabulary and complex Jinja templates) in milliseconds.

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.

```

```

###Fuck the EU AI Act:

LICENSE AGREEMENT & DISCLAIMER: This software and its source code were generated using 
automated data processing systems (AI) based on user instructions. The software is provided 
'AS IS', without warranties of title, originality, non-infringement, merchantability, or 
fitness for a particular purpose. Neither the author nor the provider guarantees freedom 
from third-party rights or training data patents. All risks associated with operation, 
maintenance, modification, or third-party claims are borne solely by the end user.
