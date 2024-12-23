import os
from pathlib import Path
from paperscraper.scholar import get_scholar_papers
from paperscraper.pdf import save_pdf
from paperscraper.arxiv import get_and_dump_arxiv_papers, get_arxiv_papers, dump_papers

# Define the path for the output folder inside backend/documents
output_folder = Path("backend/documents/scraped_papers")
output_folder.mkdir(parents=True, exist_ok=True)  # Create the folder if it doesn't exist

# Get arXiv papers as a DataFrame
result = get_arxiv_papers(query="Classification of Human- and AI-Generated Texts: Investigating Features for ChatGPT", max_results=2)

# Save the papers metadata to a JSONL file
metadata_file = Path("backend/documents/test.jsonl")
# dump_papers(result, str(metadata_file))

# Save the papers as PDFs
for index, row in result.iterrows():
    pdf_url = row.get("pdf_url")
    doi = row.get("doi")
    if not doi:
        print(f"Skipping paper at index {index}: DOI not found.")
        continue

    # Create a sanitized file path within the output folder
    filename = doi.replace("/", "_") + ".pdf"  # Replace slashes to avoid directory issues
    filepath = output_folder / filename

    # Save the PDF
    print(f"Downloading PDF for DOI: {doi} into {filepath}")
    save_pdf({"doi": doi}, filepath=str(filepath))

print(f"Papers metadata saved to {metadata_file} and PDFs downloaded into the '{output_folder}' folder.")
