import sys
import os
import urllib.request
import struct
import re
import mmap

# ==============================================================================
# STRUKTURA SEKCÍ (Pořadí sekcí v dokumentaci)
# ==============================================================================
SECTION_ORDER = [
    "IDENTIFIKACE A PŮVOD MODELU",
    "ARCHITEKTURA A KAPACITA",
    "KVANTOVÁNÍ A FORMÁT SOUBORU",
    "TOKENIZÉR A SPECIÁLNÍ TOKENY",
    "DOPORUČENÉ INFERENCE PARAMETRY",
    "OSTATNÍ A METADATA LICENCE",
]

# ==============================================================================
# STATICKÉ ENCYKLOPEDICKÉ VYSVĚTLIVKY (Offline znalostní báze pro GGUF)
# ==============================================================================
FILE_TYPE_MAP = {
    0: "ALL_F32 (Plná přesnost Float 32-bit)",
    1: "MOST_F16 (Většina vah v Float 16-bit)",
    2: "MOST_Q4_0 (Základní 4-bit kvantování)",
    3: "MOST_Q4_1 (Základní 4-bit s biasem)",
    7: "MOST_Q8_0 (8-bit kvantování, vysoká přesnost)",
    8: "MOST_Q5_0 (5-bit kvantování)",
    9: "MOST_Q5_1 (5-bit s biasem)",
    10: "MOST_Q2_K (2-bit K-quant, velmi vysoká komprese)",
    11: "MOST_Q3_K_S (3-bit K-quant Small)",
    12: "MOST_Q3_K_M (3-bit K-quant Medium)",
    13: "MOST_Q3_K_L (3-bit K-quant Large)",
    14: "MOST_Q4_K_S (4-bit K-quant Small)",
    15: "MOST_Q4_K_M (4-bit K-quant Medium - doporučený kompromis)",
    16: "MOST_Q5_K_S (5-bit K-quant Small)",
    17: "MOST_Q5_K_M (5-bit K-quant Medium)",
    18: "MOST_Q6_K (6-bit K-quant, téměř beztrátové)",
    19: "MOST_IQ2_XXS (Importance Matrix 2-bit Extra Extra Small)",
    20: "MOST_IQ2_XS (Importance Matrix 2-bit Extra Small)",
    21: "MOST_IQ3_XXS (Importance Matrix 3-bit Extra Extra Small)",
    22: "MOST_IQ1_S (Importance Matrix 1-bit Small)",
    23: "MOST_IQ4_NL (Importance Matrix 4-bit Non-Linear)",
    24: "MOST_IQ3_S (Importance Matrix 3-bit Small)",
    25: "MOST_IQ2_S (Importance Matrix 2-bit Small)",
    26: "MOST_IQ4_XS (Importance Matrix 4-bit Extra Small)",
    27: "MOST_IQ2_M (Importance Matrix 2-bit Medium)",
    28: "MOST_IQ3_XS (Importance Matrix 3-bit Extra Small)",
}

TOKEN_TYPE_MAP = {
    1: "BPE / Byte-Pair Encoding (GPT, Llama, Qwen)",
    2: "WordPiece (BERT)",
    3: "Unigram (T5, SentencePiece)",
    4: "UGM / Custom Unigram",
}

KEY_EXPLANATIONS = {
    "tokenizer.ggml.tokens": "Počet položek ve slovníku / celková velikost slovníku tokenizéru.",
    "tokenizer.ggml.token_type": "Algoritmus/Typ tokenizéru použitý pro rozklad textu na subwordy.",
    "tokenizer.ggml.eos_token_id": "ID tokenu pro označení konce sekvence/odpovědi (End Of Sequence).",
    "tokenizer.ggml.bos_token_id": "ID tokenu pro označení začátku sekvence (Beginning Of Sequence).",
    "tokenizer.ggml.padding_token_id": "ID tokenu používané pro zarovnání délky sekvencí (Padding).",
    "tokenizer.ggml.unknown_token_id": "ID tokenu pro neznámé/neevidované znaky (Unknown token).",
    "general.quantization_version": "Verze algoritmu kvantování v GGUF knihovně (v2 je aktuální standard).",
    "general.file_type": "Kódové označení hlavní použité kvantizační metody pro váhy modelu.",
    "general.size_label": "Kategorie velikosti modelu podle počtu parametrů (např. 9B, 70B).",
    "general.tags": "Štítky určené autorem pro specifikaci zaměření (např. agentic, code, math).",
    "general.base_model.0.name": "Přesné jméno výchozího (podkladového) modelu, ze kterého vznikl tento fine-tune/derivát.",
}

# ==============================================================================
# POMOCNÉ FUNKCE PRO PARSOVÁNÍ A PŘEVOD DATA
# ==============================================================================
def parse_gguf_metadata_stream(data):
    offset = 0
    magic = data[offset:offset+4]; offset += 4
    if magic != b'GGUF':
        raise ValueError("Neplatný GGUF formát (chybí magické číslo GGUF)")
    
    version = struct.unpack_from("<I", data, offset)[0]; offset += 4
    tensor_count = struct.unpack_from("<Q", data, offset)[0]; offset += 8
    kv_count = struct.unpack_from("<Q", data, offset)[0]; offset += 8
    
    metadata = {}
    
    def read_string():
        nonlocal offset
        if offset + 8 > len(data): raise EOFError("Konec dat při čtení délky řetězce")
        length = struct.unpack_from("<Q", data, offset)[0]; offset += 8
        if offset + length > len(data): raise EOFError("Konec dat při čtení obsahu řetězce")
        s = data[offset:offset+length].decode('utf-8', errors='ignore')
        offset += length
        return s

    def read_value(vtype):
        nonlocal offset
        if vtype == 0: val = data[offset]; offset += 1
        elif vtype == 1: val = struct.unpack_from("<b", data, offset)[0]; offset += 1
        elif vtype == 2: val = struct.unpack_from("<H", data, offset)[0]; offset += 2
        elif vtype == 3: val = struct.unpack_from("<h", data, offset)[0]; offset += 2
        elif vtype == 4: val = struct.unpack_from("<I", data, offset)[0]; offset += 4
        elif vtype == 5: val = struct.unpack_from("<i", data, offset)[0]; offset += 4
        elif vtype == 6: val = struct.unpack_from("<f", data, offset)[0]; offset += 4
        elif vtype == 7: val = bool(data[offset]); offset += 1
        elif vtype == 8: val = read_string()
        elif vtype == 9:
            array_type = struct.unpack_from("<I", data, offset)[0]; offset += 4
            array_len = struct.unpack_from("<Q", data, offset)[0]; offset += 8
            arr = [read_value(array_type) for _ in range(array_len)]
            val = arr
        elif vtype == 10: val = struct.unpack_from("<Q", data, offset)[0]; offset += 8
        elif vtype == 11: val = struct.unpack_from("<q", data, offset)[0]; offset += 8
        elif vtype == 12: val = struct.unpack_from("<d", data, offset)[0]; offset += 8
        else:
            raise ValueError(f"Neznámý GGUF typ hodnoty: {vtype}")
        return val

    for _ in range(kv_count):
        key = read_string()
        vtype = struct.unpack_from("<I", data, offset)[0]; offset += 4
        metadata[key] = read_value(vtype)
        
    return metadata, tensor_count

def clean_val(val):
    if isinstance(val, list) and all(isinstance(x, int) and 0 <= x <= 255 for x in val) and len(val) > 0:
        try:
            return bytes(val).decode('utf-8', errors='ignore')
        except Exception:
            pass
    if isinstance(val, str):
        return val
    return str(val)

def decode_val(key, val):
    if isinstance(val, list) and all(isinstance(x, int) and 0 <= x <= 255 for x in val) and len(val) > 0:
        if key not in ["tokenizer.ggml.tokens", "tokenizer.ggml.token_type"]:
            try:
                decoded = bytes(val).decode('utf-8', errors='ignore')
                return f"`{decoded}` *(převedeno z bajtového pole)*"
            except Exception:
                pass

    if key == "general.file_type" and isinstance(val, int):
        explanation = FILE_TYPE_MAP.get(val, "Neznámý kód kvantování")
        return f"**{val}** → *{explanation}*"

    if key == "tokenizer.ggml.token_type" and isinstance(val, int):
        explanation = TOKEN_TYPE_MAP.get(val, "Neznámý typ tokenizéru")
        return f"**{val}** → *{explanation}*"

    if isinstance(val, list):
        if "tags" in key:
            cleaned_items = [clean_val(item) for item in val]
            return f"`{', '.join(cleaned_items)}` *(pole o délce {len(val)})*"
        if len(val) > 10:
            return f"*[Pole o délce {len(val)} prvků]*"
        return f"`{val}`"

    return f"`{val}`"

def categorize_key(key):
    if any(x in key for x in ["name", "author", "base_model", "tags", "description", "url"]):
        return "IDENTIFIKACE A PŮVOD MODELU"
    elif any(x in key for x in ["architecture", "size_label", "block_count", "embedding_length", "head_count", "context_length", "expert", "attention", "rope"]):
        return "ARCHITEKTURA A KAPACITA"
    elif any(x in key for x in ["quantization", "file_type", "tensor"]):
        return "KVANTOVÁNÍ A FORMÁT SOUBORU"
    elif key.startswith("tokenizer.") and "chat_template" not in key:
        return "TOKENIZÉR A SPECIÁLNÍ TOKENY"
    elif any(x in key for x in ["temperature", "top_p", "top_k", "repeat_penalty", "penalty", "sample"]):
        return "DOPORUČENÉ INFERENCE PARAMETRY"
    else:
        return "OSTATNÍ A METADATA LICENCE"

def extract_sampling_params(metadata):
    params = {}
    sampling_keys = {
        "temperature": ["temperature", "sampling.temp", "general.temperature"],
        "top_p": ["top_p", "sampling.top_p", "general.top_p"],
        "top_k": ["top_k", "sampling.top_k", "general.top_k"],
        "repeat_penalty": ["repeat_penalty", "sampling.repeat_penalty"],
        "presence_penalty": ["presence_penalty", "sampling.presence_penalty"],
        "frequency_penalty": ["frequency_penalty", "sampling.frequency_penalty"],
        "min_p": ["min_p", "sampling.min_p"],
    }
    
    for param_name, possible_keys in sampling_keys.items():
        for k in possible_keys:
            if k in metadata:
                params[param_name] = metadata[k]
                break
            for meta_key in metadata.keys():
                if meta_key.endswith(f".{param_name}") or meta_key == param_name:
                    params[param_name] = metadata[meta_key]
                    break
    return params

def analyze_thinking_capabilities(metadata, chat_template_str):
    detected_modes = []
    details = []

    if chat_template_str:
        tpl_len_kb = round(len(chat_template_str) / 1024, 1)
        
        reasoning_tags = set(tag.lower() for tag in re.findall(r'<(think|thought|reasoning|reflection|internal_monologue|chain_of_thought)', chat_template_str, re.IGNORECASE))
        
        if reasoning_tags:
            detected_modes.append("Explicitní XML Thinking Tagy")
            details.append(f"V Jinja šabloně ({tpl_len_kb} kB) nalezeny značky: " + ", ".join([f"`<{t}>`" for t in reasoning_tags]))
            
        mode_switches = re.findall(r'(enable_think|enable_reasoning|thinking_mode|think_mode|reasoning_effort|deep_think|turbo_brilliance|brilliance_mode)', chat_template_str, re.IGNORECASE)
        if mode_switches:
            unique_switches = sorted(list(set(mode_switches)))
            detected_modes.append("Konfigurovatelné Thinking Přepínače (DavidAU/Custom)")
            details.append(f"Šablona podporuje dynamické přepínání režimů myslení skrze proměnné: " + ", ".join([f"`{s}`" for s in unique_switches]))

        if any(phrase in chat_template_str.lower() for phrase in ["think step by step", "detailed reasoning", "internal monologue", "show your thought process"]):
            if "Instrukční System Prompt pro uvažování" not in detected_modes:
                detected_modes.append("Instrukční System Prompt pro uvažování")
                details.append("Šablona automaticky vkládá do systémového promptu pokyny k myšlenkovému rozboru před odpovědí.")

    gen_name = clean_val(metadata.get("general.name", ""))
    tags = str(metadata.get("general.tags", ""))
    
    if any(kw in gen_name.lower() or kw in tags.lower() for kw in ["thinking", "reasoning", "qwq", "r1", "deepseek-r1"]):
        detected_modes.append("Označení v Metadatech/Názvu")
        details.append(f"Model nese v metadatech/násobném jménu explicitní označení reasoning rodiny (`{gen_name}`).")

    tokens_val = metadata.get("tokenizer.ggml.tokens", [])
    if isinstance(tokens_val, list) and len(tokens_val) > 0 and isinstance(tokens_val[0], str):
        thinking_tokens = [t for t in tokens_val if any(kw in t.lower() for kw in ["<think>", "</think>", "<|thought|>", "[think]"])]
        if thinking_tokens:
            detected_modes.append("Speciální Tokenizérové Tokeny")
            details.append("Ve slovníku tokenizéru detekovány přímé kontrolní tokeny: " + ", ".join([f"`{t}`" for t in thinking_tokens]))

    return detected_modes, details

def extract_moe_and_architecture_details(metadata):
    arch = clean_val(metadata.get("general.architecture", "unknown"))
    
    expert_count = None
    expert_used_count = None
    shared_expert_count = None
    
    for k, v in metadata.items():
        if "expert_count" in k or "expert_per_node" in k or "ffn_expert_count" in k:
            expert_count = v
        elif "expert_used_count" in k or "experts_per_tok" in k:
            expert_used_count = v
        elif "shared_expert" in k:
            shared_expert_count = v

    is_moe = expert_count is not None and expert_count > 1
    model_type_str = f"MoE (Mixture of Experts - {expert_count} expertů, aktivních {expert_used_count})" if is_moe else "Dense (hustá síť - standardní FFN)"

    lines = []
    lines.append(f"- **Základní typ architektury:** `{arch}` ({model_type_str})")
    
    if is_moe:
        lines.append(f"  - **Celkový počet expertů na vrstvu:** `{expert_count}`")
        lines.append(f"  - **Počet aktivně routovaných expertů na token:** `{expert_used_count}`")
        if shared_expert_count is not None:
            lines.append(f"  - **Sdílené experty (Shared Experts):** `{shared_expert_count}`")
        lines.append("  - **Princip fungování MoE:** Router dynamicky vybírá podskupinu expertů pro každý token, což zvyšuje kapacitu modelu bez navýšení výpočetní náročnosti při inferenci.")

    ctx = metadata.get(f"{arch}.context_length")
    if ctx:
        lines.append(f"- **Maximální kontextové okno:** `{ctx}` tokenů")

    rope_base = metadata.get(f"{arch}.rope.freq_base")
    if rope_base:
        lines.append(f"- **RoPE frequency base:** `{rope_base}`")

    return "\n".join(lines)

def process_gguf(source):
    source = source.strip('"').strip("'")
    
    if source.startswith(("http://", "https://")):
        print(f"[INFO] Stahuji úvodní blok metadat přes HTTP Range...")
        range_bytes = 134217728  # Zvýšeno na 128 MB pro webové stahování
        req = urllib.request.Request(source, headers={"Range": f"bytes=0-{range_bytes}"})
        try:
            with urllib.request.urlopen(req) as response:
                data = response.read()
        except Exception as e:
            print(f"[CHYBA] Selhalo stahování přes HTTP Range: {e}")
            return
        parsed_filename = source.split('/')[-1].split('?')[0]
        base_name = os.path.splitext(parsed_filename)[0]
    else:
        if not os.path.exists(source):
            print(f"[CHYBA] Soubor '{source}' neexistuje.")
            return
        base_name = os.path.splitext(os.path.basename(source))[0]
        print(f"[INFO] Otevírám lokální soubor pomocí paměťového mapování (mmap)...")
        with open(source, "rb") as f:
            # mmap bezpečně zpřístupní celý soubor jako pole bytů bez přetížení RAM
            data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
    if not base_name:
        base_name = "extracted_model"

    output_jinja = f"{base_name}.jinja"
    output_md = f"{base_name}.md"

    print("[INFO] Parsuji metadatové struktury...")
    try:
        metadata, tensor_count = parse_gguf_metadata_stream(data)
    except Exception as e:
        print(f"[CHYBA] Chyba při parsování GGUF binární struktury: {e}")
        return

    chat_template = metadata.get("tokenizer.chat_template")
    chat_template_str = ""
    if chat_template:
        if isinstance(chat_template, list):
            chat_template_str = bytes(chat_template).decode('utf-8', errors='ignore')
        else:
            chat_template_str = str(chat_template)
            
        with open(output_jinja, "w", encoding="utf-8") as f:
            f.write(chat_template_str)
        print(f"[OK] Chat template ({round(len(chat_template_str)/1024, 1)} kB) uložen do: {output_jinja}")
    else:
        print("[VAROVÁNÍ] Chat template nebyl v metadatech nalezen.")

    gen_name = clean_val(metadata.get("general.name", "Neznámý model"))
    size_label = clean_val(metadata.get("general.size_label", "Neznámá velikost"))
    file_type_raw = metadata.get("general.file_type", 0)
    file_type_desc = FILE_TYPE_MAP.get(file_type_raw, f"Typ {file_type_raw}")

    is_moe_check = any("expert_count" in k for k in metadata.keys())
    moe_or_dense = "MoE" if is_moe_check else "Dense"

    base_models = []
    i = 0
    while True:
        b_name = metadata.get(f"general.base_model.{i}.name")
        if b_name is not None:
            base_models.append(clean_val(b_name))
            i += 1
        else:
            break
    if not base_models:
        b_general = metadata.get("general.base_model.name")
        if b_general:
            base_models.append(clean_val(b_general))
        else:
            base_models.append("Neznámý původ / Originální model")

    origin_str = " + ".join(base_models)

    tags_raw = metadata.get("general.tags", [])
    if isinstance(tags_raw, list):
        tags_str = ", ".join([clean_val(t) for t in tags_raw])
    else:
        tags_str = clean_val(tags_raw)

    tok_pre = clean_val(metadata.get("tokenizer.ggml.pre", "neuvedeno"))
    tok_tokens = metadata.get("tokenizer.ggml.tokens")
    if isinstance(tok_tokens, list):
        tok_tokens_count = len(tok_tokens)
    else:
        tok_tokens_count = tok_tokens if tok_tokens else "neznámo"

    thinking_modes, thinking_details = analyze_thinking_capabilities(metadata, chat_template_str)

    sample_params = extract_sampling_params(metadata)
    arch_details_block = extract_moe_and_architecture_details(metadata)

    categorized = {sec: [] for sec in SECTION_ORDER}

    for k, v in metadata.items():
        if k == "tokenizer.chat_template":
            continue

        sec = categorize_key(k)
        formatted_val = decode_val(k, v)
        exp = KEY_EXPLANATIONS.get(k, "")
        categorized[sec].append((k, formatted_val, exp))

    md_lines = []
    md_lines.append(f"# Průvodní list a dokumentace modelu: `{base_name}`\n")
    
    md_lines.append("## Souhrnný přehled modelu")
    md_lines.append(f"> **{gen_name}** `{moe_or_dense}` - **{size_label}** `({file_type_desc})`")
    md_lines.append(f"> * **Původ:** {origin_str}")
    md_lines.append(f"> * **Zaměření:** {tags_str if tags_str else 'Neuvedeno'}")
    md_lines.append(f"> * **Slovník:** Předzpracování: `{tok_pre}` | Velikost slovníku: `{tok_tokens_count}` tokenů")
    
    if thinking_modes:
        modes_str = " | ".join([f"**{m}**" for m in thinking_modes])
        md_lines.append(f"> * **Detekované myslící (Thinking/Reasoning) režimy:** {modes_str}")
    else:
        md_lines.append(f"> * **Detekované myslící (Thinking/Reasoning) režimy:** *(Standardní model bez explicitního uvažování)*")

    if sample_params:
        params_formatted = ", ".join([f"`{k}`: {v}" for k, v in sample_params.items()])
        md_lines.append(f"> * **Doporučené inference parametry (v metadatech):** {params_formatted}")
    else:
        md_lines.append(f"> * **Doporučené inference parametry:** *(v metadatech nespecifikováno, použijí se výchozí hodnoty sampleru)*")
        
    md_lines.append("\n")

    if thinking_details:
        md_lines.append("## Detailní rozbor Thinking / Reasoning funkcionality")
        for detail in thinking_details:
            md_lines.append(f"- {detail}")
        md_lines.append("\n---\n")

    md_lines.append("## Architektonická analýza (včetně MoE a statistik)")
    md_lines.append(arch_details_block)
    md_lines.append("\n---\n")

    for sec in SECTION_ORDER:
        items = categorized[sec]
        if not items:
            continue
        
        md_lines.append(f"## {sec}\n")
        md_lines.append("| Klíč | Hodnota | Význam / Popis |")
        md_lines.append("| :--- | :--- | :--- |")
        
        for k, v, exp in items:
            exp_text = exp if exp else "-"
            md_lines.append(f"| `{k}` | {v} | {exp_text} |")
        
        md_lines.append("\n")

    md_content = "\n".join(md_lines)

    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[OK] Dokumentace uložena do: {output_md}\n")

    print("=" * 80)
    print(" VÝSTUPNÍ DOKUMENTACE MODELU (ZOBRAZENÍ NA KONZOLI)")
    print("=" * 80)
    print(md_content)
    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("--- GGUF Metadata & Chat Template Extractor ---")
        target = input("Zadej cestu k lokálnímu GGUF souboru nebo URL adresu: ").strip()
    else:
        target = sys.argv[1]
    
    if not target:
        print("[CHYBA] Nebyl zadán žádný vstup.")
        sys.exit(1)
        
    process_gguf(target)