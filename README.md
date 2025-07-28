Here’s your updated and **complete README.md** including the **Docker run instructions** and integrating all your content under a unified structure:

---

# 📄 Intelligent PDF Heading Extraction & Semantic Ranking Pipeline

---

## 🧠 Project Overview

This project implements an end-to-end **PDF analysis pipeline** that performs:

1. **Logical content extraction** from PDF documents
2. **Heading classification** (H1–H3) using a GNN (Graph Neural Network)
3. **Text embedding** using Sentence Transformers
4. **Semantic heading ranking** based on a persona/task
5. **Subsection extraction** under top-ranked headings

The final output is a **structured JSON** containing:

* 📌 Extracted headings with predicted levels (`H1`, `H2`, `H3`)
* ⭐ Top-k important headings based on a user-defined persona and task
* ✂ A concise subsection from each top heading's content

---

## 🚀 Use Cases

* 📚 Document summarization
* 🧑‍💻 Personalized content extraction
* ✈ Travel/event planning from PDFs
* 📄 Resume parsing or report structuring

---

## 📂 Project Structure

```
.
├── pdfs/                       # Input PDF files
├── preproc/                    # Intermediate embedded/heading JSONs
├── best_model.pth              # Trained GNN model
├── final_output_1B.json        # Final output JSON (headings + sections)
├── process_pdfs.py             # Batch runner
├── process_single_pdf.py       # Core extraction logic
├── Dockerfile                  # Docker container definition
└── README.md                   # This file
```

---

## 📌 Features

* 📑 **PDF Parsing** using `unstructured` and `PyMuPDF`
* 🧠 **GNN-Based Heading Classification** with `GraphSAGE`
* 🧬 **MiniLM Embeddings** for semantic richness
* 📊 **Semantic Ranking** using persona-based similarity
* ✂ **Subsection Extraction** under each top-ranked heading

---

## 🛠 Dependencies

Install all dependencies using:

```bash
pip install -r requirements.txt
```

### `requirements.txt` should contain:

```
torch
dgl
fitz  # PyMuPDF
numpy
pandas
scikit-learn
tqdm
sentence-transformers
unstructured
```

---

## 📥 Input Format

Place all **text-based** PDFs inside:

```bash
./pdfs/
```

---

## 🧪 How It Works

### 1️⃣ Preprocessing & Classification

```python
recs = preprocess_and_classify_all(PDF_DIR, PREPROC_DIR)
```

For each PDF:

* Extracts logical structure
* Builds graphs from layout & embeddings
* Classifies heading levels using the trained GNN

### 2️⃣ Semantic Ranking + Subsection Extraction

```python
final = build_final_json(recs, PERSONA, JOB_TO_BE_DONE, TOP_K)
```

* Uses embeddings to find top-k relevant headings
* Extracts \~50-word subsections under each heading

### 3️⃣ Final Output

```json
{
  "title": "Your PDF Title",
  "content": [
    {"level": "H1", "text": "Abstract", "page": 0},
    {"level": "H2", "text": "Introduction", "page": 1},
    ...
  ]
}
```

---

## 🐳 Docker Usage

Build and run the pipeline in a Docker container.

### 1️⃣ Build the Docker Image

```bash
docker build --platform linux/amd64 -t challenge1b-runner .
```

### 2️⃣ Run the Container

```bash
docker run --rm \
  -v "$(pwd)/Collection 3:/app/input:ro" \
  -v "$(pwd)/Collection 3/challenge1b_output:/app/output" \
  challenge1b-runner
```

✅ Your JSON outputs will be in: `Collection 3/challenge1b_output/`

---

## 🧾 Sample Final Output (`final_output_1B.json`)

```json
{
  "metadata": {
    "input_documents": ["example.pdf"],
    "persona": "Travel Planner",
    "job_to_be_done": "Plan a trip of 4 days for a group of 10 college friends.",
    "processing_timestamp": "2025-07-28T18:55:00"
  },
  "extracted_sections": [
    {
      "document": "example.pdf",
      "section_title": "Day 1: Arrival and Sightseeing",
      "importance_rank": 1,
      "page_number": 2
    }
  ],
  "subsection_analysis": [
    {
      "document": "example.pdf",
      "refined_text": "Arrive at the destination by noon. Check in to the hotel and rest. In the evening, visit the local market and try street food.",
      "page_number": 2
    }
  ]
}
```

---

## 🧠 Model Details

Heading classification is performed by a 2-layer **GraphSAGE** model using:

* Relative font size
* Word count
* x/y position on page
* Bold/font style
* 384-d MiniLM text embeddings

### Labels:

* `H1` – Main heading
* `H2` – Sub-heading
* `H3` – Minor heading
* `O`  – Other / Not a heading

---

## 🧠 Re-training the Model

To retrain:

1. Prepare labeled graph data
2. Use `HeadingClassifier` class (already implemented)
3. Run a typical PyTorch training loop using `DGL`

---

## ⚠️ Notes

* Works only on **text-based PDFs** (not scanned images)
* Results improve with structured documents
* Ensure `best_model.pth` is present in root before running

---

## 📚 Citations / Credits

* 🔗 [SentenceTransformers](https://www.sbert.net/)
* 🔗 [Unstructured.io](https://github.com/Unstructured-IO/unstructured)
* 🔗 [DGL.ai](https://www.dgl.ai/)
* 🔗 [PyMuPDF](https://pymupdf.readthedocs.io/)

---

Let me know if you'd like this as a downloadable `README.md` file or need badges/logos for GitHub styling.
