# 📄 Intelligent PDF Heading Extraction & Semantic Ranking Pipeline

## 🧠 Project Overview

This project implements an end-to-end *PDF analysis pipeline* that performs:

1. *Logical content extraction* from PDF documents.
2. *Heading classification* using Graph Neural Networks (GNN).
3. *Text embedding* using Sentence Transformers.
4. *Semantic heading ranking* based on a user-defined persona and task.
5. *Subsection extraction* from top-ranked headings.

The final output is a structured JSON containing:

* Extracted headings with predicted levels (H1, H2, H3)
* Top-k important headings based on a given persona and job
* A concise subsection from each top heading's content

---

## 🚀 Use Cases

* Document summarization
* Personalized content extraction
* Travel/event planning from PDFs
* Resume parsing or report structuring

---

## 📂 Project Structure


.
├── pdfs/                    # Input PDF files
├── preproc/                 # Intermediate embedded/heading JSONs
├── best_model.pth           # Trained GNN classifier
├── final_output_1B.json     # Final output with top-k sections and content
├── your_script.py           # Main pipeline (provided code)
└── README.md                # This file


---

## 📌 Features

* 📌 *PDF Parsing*: Extracts logical structure (title, paragraphs, lists, etc.) using unstructured and PyMuPDF.
* 🧠 *GNN-Based Heading Classification*: Classifies text blocks as H1, H2, H3, or Other using GraphSAGE model.
* 🔎 *MiniLM Embeddings*: Adds semantic understanding to text blocks with compact sentence embeddings.
* 📊 *Semantic Ranking: Uses a user-defined persona (e.g., *Travel Planner) and objective to rank the most relevant sections.
* ✂ *Subsection Extraction*: Retrieves a concise summary (\~50 words) of content under each top heading.

---

## 🛠 Dependencies

Install the required libraries using:

bash
pip install -r requirements.txt


**requirements.txt** (create this file with):

txt
torch
dgl
fitz  # PyMuPDF
numpy
pandas
scikit-learn
tqdm
sentence-transformers
unstructured


---

## 📥 Input

* *PDF files* stored in the ./pdfs directory.

---

## 🧪 How It Works

1. *Preprocessing & Classification*

bash
recs = preprocess_and_classify_all(PDF_DIR, PREPROC_DIR)


For each PDF:

* Logical content and layout features are extracted
* Graphs are built using layout and embedding features
* Headings are classified via a trained GNN (best_model.pth)

2. *Heading Ranking + Subsection Extraction*

bash
final = build_final_json(recs, PERSONA, JOB_TO_BE_DONE, TOP_K)


* Embeddings of headings and the user's task prompt are compared
* Top-k headings are ranked by semantic relevance
* Subsections under each heading are extracted (max 50 words)

3. *Output JSON*

Final structured JSON is saved as final_output_1B.json.

---

## 🧾 Sample Output (final_output_1B.json)

json
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


---

## 🧠 Model Details

The *heading classifier* is a 2-layer GraphSAGE model trained on node features that include:

* Relative font size
* Word count
* x/y layout position
* Font style (bold/type)
* 384-dimensional MiniLM text embedding

### Class Labels:

* H1 → Primary heading
* H2 → Sub-heading
* H3 → Minor sub-heading
* O  → Other (non-heading)

---

## 🔁 Re-training the Model

If you want to re-train the GNN model:

1. Prepare labeled graphs.
2. Define HeadingClassifier (already included).
3. Train using standard PyTorch/DGL training loop.

---


---

## 📌 Notes

* Only works on *text-based PDFs* (not scanned/image-based).
* Performance improves with higher-quality structured documents.
* Ensure best_model.pth is available in the root directory.

---

## 🧾 Citation / Credits

* [SentenceTransformer]("mixedbread-ai/mxbai-embed-large-v1"
* [Unstructured](https://github.com/Unstructured-IO/unstructured)
* [DGL](https://www.dgl.ai/)
* [PyMuPDF](https://pymupdf.readthedocs.io/)
