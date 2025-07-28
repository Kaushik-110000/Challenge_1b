import os
import re
import sys
import ast
import json
import pickle
import torch
import dgl
import fitz
import numpy as np
import pandas as pd
import torch.nn.functional as F

from tqdm import tqdm
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer
from unstructured.partition.pdf import partition_pdf

from collections import defaultdict

from torch import nn
from torch.utils.data import DataLoader

from dgl.nn import SAGEConv

from sklearn.preprocessing import LabelEncoder


from glob import glob
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity


def intersects(e_pts, s_bbox, tol=1.0):
    x0, y0 = e_pts[0]
    x1, y1 = e_pts[2]
    sx0, sy0, sx1, sy1 = s_bbox
    return not (sx1 < x0 - tol or sx0 > x1 + tol or sy1 < y0 - tol or sy0 > y1 + tol)



def merge_adjacent_lines_only(items, line_diff_threshold=30):
    """
    Merge items ONLY if:
    - All properties match (page, type, font, size, bold)
    - Line difference (y_pos) is approximately 1
    """
    items.sort(key=lambda x: (x["page"], x["type"], x["font"], x["size"], x["bold"], x["y_pos"]))

    merged_items = []
    i = 0

    while i < len(items):
        current = items[i].copy()
        j = i + 1

        while j < len(items):
            next_item = items[j]

            properties_match = (
                current["page"] == next_item["page"] and
                current["type"] == next_item["type"] and
                current["font"] == next_item["font"] and
                abs(current["size"] - next_item["size"]) < 0.1
                and current["bold"] == next_item["bold"]
            )

            if not properties_match:
                break

            y_diff = abs(next_item["y_pos"] - current["y_pos"])

            if y_diff > line_diff_threshold:
                break

            # Merge text and positions
            current["text"] += " " + next_item["text"]
            current["y_pos"] = next_item["y_pos"]
            current["x_center"] = min(current["x_center"], next_item["x_center"])
            current["y_position"] = next_item["y_position"]

            j += 1

        # Remove y_pos from final output
        final_item = {k: v for k, v in current.items() if k != "y_pos"}
        merged_items.append(final_item)

        i = j

    return merged_items



def extract_title(candidates):
    # Step 1: Filter elements from page 1 and above center
    page_1_elements = [el for el in candidates if el["page"] == 1 and el["y_position"] < 0.5]
    if not page_1_elements:
        return None

    # Step 2: Find the highest font size
    max_size = max(el["size"] for el in page_1_elements)

    # Step 3: Find elements with that font size (with a small tolerance)
    top_candidates = [el for el in page_1_elements if abs(el["size"] - max_size) < 0.1]

    # Step 4: If 1 to 3 such elements exist, merge them as title
    if 1 <= len(top_candidates) <= 3:
        # Sort by vertical position (top to bottom)
        top_candidates.sort(key=lambda x: x["y_position"])
        # Merge their text content
        title_text = " ".join(el["text"] for el in top_candidates).strip()
        return title_text

    return None


def remove_exaggerated_elements_page1(items,
                                      gap_multiplier: float = 2,
                                      max_singleton_freq: int = 2):
    """
    Only look at page 1 items to decide if the top‐size on that page
    is a one‐off outlier, and if so drop *just* those page‑1 elements.
    """
    # 1️⃣ Collect sizes only from page 1
    page1 = [el for el in items if el["page"] == 1]
    sizes1 = [el["size"] for el in page1]
    uniq1  = sorted(set(sizes1), reverse=True)

    # nothing to do if fewer than two distinct sizes on page 1
    if len(uniq1) < 2:
        return items

    # 2️⃣ Compute gaps on page 1
    top_gap        = uniq1[0] - uniq1[1]
    background_gaps = [uniq1[i] - uniq1[i+1] for i in range(1, len(uniq1)-1)]

    # 3️⃣ If the top size is an outlier *and* rare on page 1, drop only those
    if top_gap > np.mean(background_gaps) + gap_multiplier * np.std(background_gaps):
        freq1 = Counter(sizes1)
        if freq1[uniq1[0]] <= max_singleton_freq:
            # filter out only the page‑1 elements whose size == uniq1[0]
            return [
                el for el in items
                if not (el["page"] == 1 and abs(el["size"] - uniq1[0]) < 1e-6)
            ]

    # otherwise return untouched
    return items


def add_minilm_embeddings_to_json(data, model_name='all-MiniLM-L6-v2',
                                 embedding_key='text_embedding'):
    """
    Add ONLY 384-dimensional MiniLM text embeddings to existing JSON data

    Args:
        input_json_path (str): Path to your existing JSON file
        output_json_path (str): Path for output (if None, overwrites input)
        model_name (str): MiniLM model name (default: 61MB model)
        embedding_key (str): Key name for embedding in JSON (default: 'text_embedding')

    Returns:
        list: Updated data with embeddings
    """

    print(f"✅ Loaded {len(data)} samples")

    # Load MiniLM model
    print(f"🤖 Loading MiniLM model: {model_name}")
    model = SentenceTransformer(model_name)

    # Extract text for batch processing (faster)
    texts = [item.get('text', '') for item in data]

    # Generate embeddings in batch
    print("🔄 Generating text embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=16)

    # Add embeddings to each item
    print("📝 Adding embeddings to JSON items...")
    for i, (item, embedding) in enumerate(zip(data, embeddings)):
        item[embedding_key] = embedding.tolist()  # Convert numpy array to list for JSON

    print(f"✅ Successfully added {len(embeddings[0])}-dimensional embeddings to {len(data)} samples")
    print(f"📊 Embedding key: '{embedding_key}'")

    return data



def process_single_json_to_csv(data):
    """
    Process a single JSON file and save it as a CSV file.
    Keeps embeddings as arrays in a single column.

    Parameters:
        input_json_path (str): Path to the JSON file
        output_csv_path (str): Path where the output CSV will be saved
    """


    all_data = []
    file_id = 1

    # Normalize structure
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                item['file_id']     = file_id
                all_data.append(item)
    elif isinstance(data, dict):
        data['file_id']     = file_id
        all_data.append(data)
    else:
        print("❌ Unsupported JSON structure.")
        return None

    if not all_data:
        print("❌ No valid records found in JSON.")
        return None

    # To DataFrame
    print(f"✅ Loaded {len(all_data)} records.")
    df = pd.DataFrame(all_data)

    # Handle embeddings
    if 'text_embedding' in df.columns:
        emb0 = df['text_embedding'].iloc[0]
        # If stringified, parse back to list
        if isinstance(emb0, str):
            print("🔄 Parsing string embeddings...")
            df['text_embedding'] = df['text_embedding'].apply(lambda s: ast.literal_eval(s))
        # Convert to numpy arrays
        df['text_embedding'] = df['text_embedding'].apply(lambda x: np.array(x) if x is not None else None)
        print(f"🔢 Embedding dimension: {len(df['text_embedding'].iloc[0])}")

    # Desired order
    desired = [
        'file_id', 'page', 'type', 'text', 'font', 'size',
        'bold', 'x_center', 'y_position', 'label', 'text_embedding'
    ]
    # Build final column list
    final_cols = [c for c in desired if c in df.columns] + [c for c in df.columns if c not in desired]
    df = df[final_cols]

    to_save = df.copy()
    if 'text_embedding' in to_save.columns:
        to_save['text_embedding'] = to_save['text_embedding'].apply(
            lambda arr: arr.tolist() if isinstance(arr, np.ndarray) else arr
        )
    print(f"✅ CSV saved: {df.shape[0]} rows × {df.shape[1]} cols")

    return to_save

def min_max_scale(group):
    min_val = group['size'].min()
    max_val = group['size'].max()
    print(min_val,max_val)
    if min_val == max_val:
        group['relative_size'] = 1.0
    else:
        group['relative_size'] = (group['size'] - min_val) / (max_val - min_val)
    return group


def create_edges(df_sub, y_thresh=60, x_thresh=250, size_thresh=0.05):
    src, dst = set(), set()
    num_nodes = len(df_sub)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Layout
            dx = abs(df_sub.iloc[i]['x_center'] - df_sub.iloc[j]['x_center'])
            dy = abs(df_sub.iloc[i]['y_position'] - df_sub.iloc[j]['y_position'])

            # Style similarity
            size_diff = abs(df_sub.iloc[i]['relative_size'] - df_sub.iloc[j]['relative_size'])
            bold_match = df_sub.iloc[i]['bold_encoded'] == df_sub.iloc[j]['bold_encoded']
            type_match = df_sub.iloc[i]['type_encoded'] == df_sub.iloc[j]['type_encoded']
            wc_i, wc_j = df_sub.iloc[i]['word_count'], df_sub.iloc[j]['word_count']
            short_texts = wc_i <= 10 and wc_j <= 10

            # Connection condition
            if (
                dx < x_thresh and
                dy < y_thresh and
                size_diff < size_thresh and
                bold_match and
                type_match and
                short_texts
            ):
                src.add((i, j))
                src.add((j, i))

    src_list, dst_list = zip(*src) if src else ([], [])
    return list(src_list), list(dst_list)

def build_graphs_from_df(df, y_thresh=60, x_thresh=250, size_thresh=0.05, include_metadata=False):
    graphs = []

    for file_id, group in df.groupby('file_id'):
        group = group.reset_index(drop=True)

        # Create edges
        src, dst = create_edges(group, y_thresh, x_thresh, size_thresh)
        g = dgl.graph((src, dst), num_nodes=len(group))

        g = dgl.add_self_loop(g)

        # Add node features
        features = np.stack(group['node_feature'].values)
        g.ndata['feat'] = torch.tensor(features, dtype=torch.float32)

        # Optional: Add text + page metadata for debugging or inspection
        if include_metadata:
            g.ndata['text'] = list(group['text'])  # Stored as list of strings
            g.ndata['page'] = torch.tensor(group['page'].values, dtype=torch.int)

        graphs.append(g)

    return graphs


def make_node_features(row):
    layout_features = np.array([
        row['relative_size'],
        row['word_count'],
        row['x_center'],
        row['y_position'],
        row['bold_encoded'],
        row['type_encoded']
    ], dtype=np.float32)

    return np.concatenate([layout_features, row['text_embedding']])


def parse_emb(x):
    if isinstance(x, str):
        lst = ast.literal_eval(x)
    else:
        lst = x
    return np.array(lst, dtype=np.float32)




class HeadingClassifier(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, dropout=0.2):
        super(HeadingClassifier, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_size, aggregator_type='lstm')  # or 'gcn'
        self.conv2 = SAGEConv(hidden_size, num_classes, aggregator_type='pool')
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        h = self.conv1(g, features)
        h = F.relu(h)
        h = self.dropout(h)           # 🔥 Dropout after activation
        h = self.conv2(g, h)
        return h


def is_valid_heading(txt):
    return bool(re.search(r'[A-Za-z]+', str(txt)))


def process_single_pdf(PDF_PATH):
    import os

    base_name = os.path.splitext(os.path.basename(PDF_PATH))[0]
    EMBEDDED_JSON = f"{base_name}_embedded.json"
    HEADING_JSON = f"{base_name}_headings.json"

    MAX_NARRATIVE_WORDS = 10

    # 1️⃣ Extract logical elements
    elements = partition_pdf(
        filename=PDF_PATH,
        strategy="fast",
        infer_table_structure=False,
        extract_images_in_pdf=False,
    )

    # 2️⃣ Extract spans and normalize position info
    doc = fitz.open(PDF_PATH)
    page_spans, page_dimensions = {}, {}
    for page in doc:
        page_rect = page.rect
        page_dimensions[page.number + 1] = {'width': page_rect.width, 'height': page_rect.height}
        spans = []
        for block in page.get_text("dict")["blocks"]:
            if block.get("type", 0) != 0: continue
            for line in block["lines"]:
                for s in line["spans"]:
                    spans.append({
                        "text": s["text"], "font": s["font"], "size": s["size"],
                        "bold": "Bold" in s["font"] or bool(s.get("flags", 0) & 2),
                        "bbox": s["bbox"],
                    })
        page_spans[page.number + 1] = spans

    # 3️⃣ Match unstructured elements with spans
    merged = []
    for el in elements:
        coords = getattr(el.metadata, "coordinates", None)
        page   = getattr(el.metadata, "page_number", None)
        if not coords or page not in page_spans: continue

        page_width = page_dimensions[page]['width']
        page_height = page_dimensions[page]['height']
        e_pts = coords.points
        candidates = [s for s in page_spans[page] if intersects(e_pts, s["bbox"])]
        if not candidates: continue

        # Group by lines
        lines = defaultdict(list)
        for s in candidates:
            key = round(s["bbox"][1], 1)
            lines[key].append(s)

        for key in sorted(lines.keys()):
            spans = sorted(lines[key], key=lambda s: s["bbox"][0])
            text = "".join(span["text"] for span in spans).strip()
            if not text: continue

            first = spans[0]
            bbox = first["bbox"]
            x_center = ((bbox[0] + bbox[2]) / 2) / page_width
            y_position = bbox[1] / page_height

            merged.append({
                "page": page,
                "type": type(el).__name__,
                "text": text,
                "font": first["font"],
                "size": first["size"],
                "bold": first["bold"],
                "x_center": round(x_center, 3),
                "y_position": round(y_position, 3),
                "y_pos": key,
            })

    # 4️⃣ Clean and merge
    final_merged = merge_adjacent_lines_only(merged)
    filtered_merged = []
    for item in final_merged:
        if item["type"] == "NarrativeText" and len(item["text"].split()) > MAX_NARRATIVE_WORDS:
            continue
        filtered_merged.append(item)

    if len(doc) > 2:
        final_filtered = remove_exaggerated_elements_page1(filtered_merged)
    else:
        final_filtered = filtered_merged

    # 5️⃣ Extract title from page 1
    title = extract_title(final_filtered)

    # 6️⃣ Add MiniLM embeddings
    embedded_data = add_minilm_embeddings_to_json(data=final_filtered, model_name='all-MiniLM-L6-v2')

    # 🔹 Save embedded JSON for later use (important for subsection analysis)
    with open(EMBEDDED_JSON, 'w', encoding='utf-8') as f:
        json.dump(embedded_data, f, indent=4)

    # 7️⃣ Convert to CSV DataFrame for heading classification
    df = process_single_json_to_csv(embedded_data)
    if df is None:
        return None, None

    df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
    df = df.reset_index(drop=True)
    df = df.groupby('file_id').apply(min_max_scale).reset_index(drop=True)
    df = df.drop(columns=['size'])
    df['index'] = range(len(df))

    # Font cleaning
    variants = ['bold', 'Bold']
    mask = (~df['bold']) & df['font'].str.lower().isin(variants)
    df.loc[mask, 'bold'] = True

    le = LabelEncoder()
    df['type_encoded'] = le.fit_transform(df['type'])
    df['bold_encoded'] = np.where(df['bold'] == False, 0, 1)
    df = df.drop(['bold', 'font', 'index'], axis=1)

    # 8️⃣ Graph classification setup
    df['text_embedding'] = df['text_embedding'].apply(parse_emb)
    df['node_feature'] = df.apply(make_node_features, axis=1)
    graphs = build_graphs_from_df(df)

    # 🔍 Heading classification
    in_feats = graphs[0].ndata['feat'].shape[1]
    model = HeadingClassifier(in_feats=in_feats, hidden_size=160, num_classes=4)
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    all_predictions = []
    for g in graphs:
        feats = g.ndata['feat']
        logits = model(g, feats)
        preds = torch.argmax(F.softmax(logits, dim=1), dim=1).numpy()
        all_predictions.extend(preds)

    df['predicted_label_encoded'] = all_predictions
    label_decoder = {0: 'H1', 1: 'H2', 2: 'H3', 3: 'O'}
    df['predicted_label'] = df['predicted_label_encoded'].map(label_decoder)

    # 🔹 Output heading JSON
    df_sorted = df.sort_values(['page', 'y_position'])
    df_hdrs = df_sorted[df_sorted['predicted_label'].isin(['H1', 'H2', 'H3'])]

    heading_json = [
        {
            "level": row.predicted_label,
            "text": row.text,
            "page": int(row.page)
        }
        for _, row in df_hdrs.iterrows()
    ]

    if not any(item['level'] == 'H1' for item in heading_json):
        # Fallback: Guess heading from size if H1 missing
        num_pages = df['page'].nunique()
        rel_size_counts = df['relative_size'].value_counts()
        excluded_sizes = set(rel_size_counts[rel_size_counts > num_pages * 5].index)
        filtered_md = df[~df['relative_size'].isin(excluded_sizes)]
        dk_sorted = filtered_md.sort_values('relative_size', ascending=False)
        for _, row in dk_sorted.iterrows():
            if is_valid_heading(row['text']) and (not title or row['text'].strip() != title.strip()):
                heading_json.insert(0, {
                    "level": "H1",
                    "text": row['text'],
                    "page": int(row['page'])
                })
                break

    final_output = {"title": title if title else "", "content": heading_json}
    with open(HEADING_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4)

    print(f"✅ Saved embedded JSON: {EMBEDDED_JSON}")
    print(f"✅ Saved heading predictions JSON: {HEADING_JSON}")

    # Return paths for next pipeline stages
    return EMBEDDED_JSON, HEADING_JSON



def preprocess_and_classify_all(pdf_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    records = []

    for pdf_path in glob(os.path.join(pdf_dir, "*.pdf")):
        embedded_json_path, heading_json_path = process_single_pdf(pdf_path)

        records.append((pdf_path, embedded_json_path, heading_json_path))

    return records

def rank_headings(record_list, persona, job, top_k):
    all_heads = []
    for pdf_path, _, heading_path in record_list:
        doc_name = os.path.basename(pdf_path)
        with open(heading_path, "r", encoding="utf-8") as f:
            heading_data = json.load(f)

        for h in heading_data["content"]:
            h["document"] = doc_name
            all_heads.append(h)

    cand = [h for h in all_heads if h["level"] in ("H1", "H2")]
    if not cand:
        return []

    embedder = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
    prompt = f"You are a {persona}. {job}"
    p_emb = embedder.encode([prompt], normalize_embeddings=True)[0]

    texts = [h["text"] for h in cand]
    embs = embedder.encode(texts, normalize_embeddings=True)

    sims = cosine_similarity([p_emb], embs)[0]
    for h, s in zip(cand, sims):
        h["score"] = float(s)

    top = sorted(cand, key=lambda x: -x["score"])[:top_k]
    for i, h in enumerate(top, 1):
        h["importance_rank"] = i

    return top


def extract_subsections(top_headings, record_list, word_limit):
    items_map = {}
    for pdf_path, embedded_path, _ in record_list:
        doc = os.path.basename(pdf_path)
        with open(embedded_path, "r", encoding="utf-8") as f:
            items_map[doc] = json.load(f)

    subsections = []
    for head in top_headings:
        doc = head["document"]
        level = head["level"]
        text = head["text"]
        page = head["page"]
        lines = items_map[doc]

        idxs = [i for i, l in enumerate(lines) if l["text"] == text and l["page"] == page]
        if not idxs:
            continue
        start = idxs[0] + 1

        snippet, wc = [], 0
        for item in lines[start:]:
            if item.get("label") in ("H1", "H2") or item.get("level") == level:
                break
            w = len(item["text"].split())
            if wc + w > word_limit:
                break
            snippet.append(item["text"])
            wc += w

        subsections.append({
            "document": doc,
            "refined_text": " ".join(snippet).strip(),
            "page_number": page
        })
    return subsections

def build_final_json(records, persona, job, top_k=5):
    top_heads = rank_headings(records, persona, job, top_k)
    subsec = extract_subsections(top_heads, records, WORD_LIMIT)

    return {
        "metadata": {
            "input_documents": [os.path.basename(r[0]) for r in records],
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [
            {
                "document": h["document"],
                "section_title": h["text"],
                "importance_rank": h["importance_rank"],
                "page_number": h["page"]
            }
            for h in top_heads
        ],
        "subsection_analysis": subsec
    }


def process_single_collection(PDF_DIR,PREPROC_DIR,PERSONA,JOB_TO_BE_DONE,TOP_K=5,WORD_LIMIT=50):
    os.makedirs(PREPROC_DIR, exist_ok=True)
    recs = preprocess_and_classify_all(PDF_DIR, PREPROC_DIR)
    final = build_final_json(recs, PERSONA, JOB_TO_BE_DONE, TOP_K)
    if os.path.exists(PREPROC_DIR):
        os.rmdir(PREPROC_DIR)
    return final

    # with open("final_output_1B.json", "w", encoding="utf-8") as f:
    #     json.dump(final, f, indent=2, ensure_ascii=False)
