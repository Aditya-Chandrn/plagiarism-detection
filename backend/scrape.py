from pathlib import Path
from paperscraper.pdf import save_pdf
from paperscraper.arxiv import get_and_dump_arxiv_papers, get_arxiv_papers, dump_papers

output_folder = Path("backend/documents/scraped_papers")
output_folder.mkdir(parents=True, exist_ok=True) 
result = get_arxiv_papers(query="Classification of Human- and AI-Generated Texts: Investigating Features for ChatGPT", max_results=2)



scraped_papers = []


for index, row in result.iterrows():
    doi = row.get("doi")
    title = row.get("title")
    journal = row.get("journal")

    if not doi:
        print(f"Skipping paper at index {index}: DOI not found.")
        continue

    # Generate the filename using DOI
    filename = doi.replace("/", "_") + ".pdf"
    filepath = output_folder / filename

    save_pdf({"doi": doi}, filepath=str(filepath))

    # Create an object with paper details
    paper_details = {
        "doi": doi,
        "title": title,
        "journal": journal,
        "file_path": str(filepath),
    }
    scraped_papers.append(paper_details)

# Print the details of scraped papers
print("\nScraped Papers Details:")
for paper in scraped_papers:
    print(paper)

print(f"PDFs downloaded into the '{output_folder}' folder.")
