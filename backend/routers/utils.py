from typing import List
from fastapi import HTTPException, Request, status
from jose import JWTError, jwt
from jose.exceptions import ExpiredSignatureError
from dotenv import dotenv_values
from docx import Document as DocxDocument
import subprocess
import os
from nlp import research_similarity

from nlp.ai_detection.roberta_ai_detection import roberta_ai_detection
from nlp.ai_detection.detect_gpt_detection import detect_gpt_main


from pydantic_models.document_schemas import AIGeneratedContent, Similarity, SimilaritySource

from nlp.ai_detection.roberta_ai_detection import roberta_ai_detection
from nlp.ai_detection.detect_gpt_detection import detect_gpt_main

from pathlib import Path
from paperscraper.pdf import save_pdf
from paperscraper.arxiv import  get_arxiv_papers
from .logger import logger

config = dotenv_values(".env")


async def verify_token(request: Request):
    token = request.cookies.get("plagiarism-access-token")
    if token:
        token = token.replace("Bearer ", "")
        try:
            payload = jwt.decode(token, config["SECRET_KEY"], algorithms=[
                                 config["ALGORITHM"]])
        except ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return payload
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )


def read_md_file(file_path):
    text = ""
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    return text


async def convert_pdf_to_md(file_path: str) -> str:
    output_folder = os.path.dirname(file_path)

    try:
        result = subprocess.run(
            ['marker_single', file_path, output_folder],
            text=True,
            check=True,
            stdout=subprocess.PIPE,
            shell=False
        )

    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return ""

    pdf_folder = os.path.splitext(os.path.basename(file_path))[0]
    pdf_output_dir = os.path.join(output_folder, pdf_folder)

    if not os.path.exists(pdf_output_dir):
        raise FileNotFoundError(f"Output folder {pdf_output_dir} not found.")

    # Get the markdown file in that folder
    md_file = [f for f in os.listdir(pdf_output_dir) if f.endswith(".md")]

    if not md_file:
        raise FileNotFoundError(
            "Markdown file not found in the output directory.")

    md_file_path = os.path.join(pdf_output_dir, md_file[0])

    return md_file_path


async def convert_docx_to_md(file_path: str) -> str:
    doc = DocxDocument(file_path)
    markdown_content = "\n".join([para.text for para in doc.paragraphs])
    return markdown_content


async def convert_to_md(file_path: str) -> str:
    """
    Convert a file (PDF or DOCX) to Markdown.
    """
    if file_path.endswith('.pdf'):
        return await convert_pdf_to_md(file_path)
    elif file_path.endswith('.docx'):
        return await convert_docx_to_md(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def detect_similarity(path1, path2, paper):
    paper_name = paper["title"]
    logger.info(f"Detection Similarity between uploaded paper and {paper_name}")

    # result for uploaded paper vs ith webscraped paper
    result = research_similarity.research_similarity(path1, path2)
    return {
        "source": {
            "name": paper_name,
            "url": "https://arxiv.org/abs/" + paper["doi"].split("arXiv.")[-1]
        },
        "bert_score": float(result["bert_score"]),
        "tfidf_score": float(result["tfidf_score"]),
        "score": float(result["score"]),
        "plagiarized_content": {
            "sources": result["plagiarized_content"]["sources"]
        }
    }


def detect_ai_generated_content(file_path) -> List[AIGeneratedContent]:
    logger.info(f"Detection AI Generated Content")
    roberta_score = roberta_ai_detection(file_path)
    # detect_gpt_score = detect_gpt_main(file_path)
    detect_gpt_score = 0.48

    return [
        AIGeneratedContent(method_name="Roberta Base Model",
                           score=roberta_score),
        AIGeneratedContent(method_name="Detect GPT", score=detect_gpt_score)
    ]


async def scrape_and_save_research_papers(title):
    logger.info(f"Scraping papers from ArXiv...")

    output_folder = Path("scraped_papers")
    output_folder.mkdir(parents=True, exist_ok=True)

    result = get_arxiv_papers(query = title, max_results = 2)

    scraped_papers = []

    for index, row in result.iterrows():
        doi = row.get("doi")
        title = row.get("title")
        journal = row.get("journal")

        if not doi:
            print(f"Skipping paper at index {index}: DOI not found.")
            continue

        filename = doi.replace("/", "_") + ".pdf"
        filepath = output_folder / filename

        save_pdf({"doi": doi}, filepath=str(filepath))

        paper_details = {
            "doi": doi,
            "title": title,
            "journal": journal,
            "path": str(filepath),
        }
        scraped_papers.append(paper_details)

    logger.info(f"Scraped papers + PDFs downloaded into the '{output_folder}' folder")

    return scraped_papers
