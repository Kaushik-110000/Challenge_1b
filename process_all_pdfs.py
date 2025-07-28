import os
import json
from pathlib import Path
from process_pdfs import process_single_collection

def main():
    # Base input directory mounted into container
    base_input = Path(os.getenv('INPUT_DIR', '/app/input'))

    # Load challenge config
    config_path = base_input / 'challenge1b_input.json'
    with open(config_path, 'r', encoding='utf-8') as cfg:
        config = json.load(cfg)

    # Directories
    pdf_dir = base_input / 'PDFs'

    # Write everything under /app/output so /app/input can stay read‑only
    output_dir = Path('/app/output')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scratch folder for preprocessing
    preproc_dir = output_dir / 'tmp_preproc'

    # Extract persona and job
    persona = config.get('persona', {})
    job = config.get('job_to_be_done', {})

    # Process collection
    final = process_single_collection(
        PDF_DIR=str(pdf_dir),
        PREPROC_DIR=str(preproc_dir),
        PERSONA=persona,
        JOB_TO_BE_DONE=job,
        TOP_K=5,
        WORD_LIMIT=50
    )

    # Save final output
    out_path = output_dir / 'final_output_1B.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"Saved final output to {out_path}")

if __name__ == '__main__':
    main()
