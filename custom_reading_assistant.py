import random
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Literal

import pandas as pd
from pydantic import BaseModel, Field
import pymupdf
from pymupdf.utils import getColor, getColorList

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage

# Constants
#LS = 'Stockholm'
CUTOFF = 0.6
INDEX_FOLDER = Path("data","index")
MODEL_NAME = 'KBLab/sentence-bert-swedish-cased'
DATA_FOLDER = Path("data")
OUT_FOLDER = Path("annotated")
FILENAME = Path('budgetpropositionen-for-2025-hela-dokumentet-prop.-2024251 19.38.11.pdf')

def handle_colors():
    # Handle colors
    color_list = getColorList()
    # Avoid the colors that are too dark
    color_list = [color for color in color_list if sum(getColor(color)) > 1]

    custom_colors = ["pink", "green", "darkgoldenrod1", "chocolate", "palevioletred", "lightsteelblue", "mediumpurple1"]
    color_list = list(set(color_list) - set(custom_colors))
    # Shuffle color list
    random.seed(42)
    random.shuffle(color_list)
    # Extend the color list with the random colors
    custom_colors.extend(color_list)
    return custom_colors

custom_colors = handle_colors()

# Initialize embedding model
embed_model = HuggingFaceEmbedding(model_name=MODEL_NAME, trust_remote_code=True)

# Load from disk
print("Loading index from disk")
##############################################################################
# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir=INDEX_FOLDER)

# load index
index = load_index_from_storage(storage_context,embed_model=embed_model)
##############################################################################

queries = {
    'Arbetsmarknadsfrågor': {
        'Fråga 1' :'Hur går det för de unga på arbetsmarknaden?'
        },
    'Jämställdhetsfrågor': {
            'Fråga 2': 'Hur har inkomstskillnaderna mellan män och kvinnor utvecklats?'
    }
}


# pydanctic model for the results
class Result(BaseModel):
    id: str
    text: str
    source: int
    score: float
    is_duplicate: bool = False
    duplicates: List[str] = Field(default_factory=list)
    color: str
    subject: str = ""
    applicable: bool = True


retriever = index.as_retriever(similarity_top_k=5)

# Function to hash large text for efficient comparison
def hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def collect_results(data: Dict[str, str], i=0) -> List[Result]:
    results_list = []
    for subject, target in data.items():
        for target_id, target_text in target.items():
            #print(f"Retrieving for target {target_id}")
            retrieved_results = retriever.retrieve(target_text)
            #print(f"{target_id} found: {len(retrieved_results)} results")
            for r in retrieved_results:
                result = Result(
                    id=target_id,
                    text=r.text,
                    source=int(r.metadata['source']),
                    score=r.score,
                    color=custom_colors[i],
                    subject=subject,
                )
                results_list.append(result)
            i+=1
    return results_list, i

# Function to identify and mark duplicates
def mark_duplicates(models: List[Result]) -> List[Result]:
    text_hash_to_ids: Dict[str, List[int]] = {}

    # First pass: fill the dictionary with text hashes and corresponding ids
    for model in models:
        text_hash = hash_text(model.text)
        if text_hash not in text_hash_to_ids:
            text_hash_to_ids[text_hash] = []
        text_hash_to_ids[text_hash].append(model.id)

    # Second pass: mark duplicates based on the filled dictionary
    for model in models:
        text_hash = hash_text(model.text)
        id_set=set(text_hash_to_ids[text_hash])-set([model.id])
        if len(id_set) > 0:
            model.is_duplicate = True
            model.duplicates = text_hash_to_ids[text_hash]
    
    # sort by score in falling order to assert that the target_id with the highest score is annotaded
    models.sort(key=lambda x: x.score, reverse=False)

results_list, _ = collect_results(queries)


# Mark duplicates in the sample data
mark_duplicates(results_list)

# Annotation
def annotate_page(doc, result:Result, text_instances, content="Här är en annotation",failsafe=False):
    """Annotate a page with text instances."""
    page = doc.load_page(result.source - 1)  # Page numbers are 0-indexed in PyMuPDF
    for idx, inst in enumerate(text_instances):

        annot = page.add_highlight_annot(inst)
        annot.set_colors(stroke=getColor(result.color))
        if idx==0:
            score=round(result.score,2)
            if result.is_duplicate:
                annot_dup = page.add_caret_annot(inst.ur)
                annot.set_info(content=content, title=result.id, subject=result.subject)
                annot_dup.set_info(content=",".join(result.duplicates), title="Dubbletter")
                # set stroke opacity to 0.5 for duplicates
                opacity=1/(len(result.duplicates))
                annot.update(opacity=opacity)
            else:
                content=f"Score: {score}\n{content}"
                annot.set_info(content=content, title=result.id, subject=result.subject)
            if failsafe:
                annot.set_colors(stroke=getColor(result.color))
                annot.set_info(content=f"FAILSAFE\n{content}", title=result.id, subject=result.subject)
        else:
            annot.set_info(title=result.id, subject=result.subject)
            if result.is_duplicate:
                annot.update(opacity=opacity)
        annot.update()

    return doc

doc_mu = pymupdf.open(DATA_FOLDER / FILENAME) # open a document


for i, result in enumerate(results_list):
    if result.score < CUTOFF:
        print(f"Skipping result for {result.id} with score {result.score}")
        continue
    if not result.applicable:
        print(f"Skipping result for {result.id} not applicable")
        continue
    page_number = int(result.source)
    text = result.text
    page = doc_mu.load_page(page_number - 1)  # Page numbers are 0-indexed in PyMuPDF
    text_instances = page.search_for(text, quads=True)
    # if I don't find the text, try to find the first 3 words
    if not text_instances:
        text_instances = page.search_for(" ".join(text.split()[:4]), quads=True)
        failsafe=True
        if not text_instances:
            # raise an error if the text is not found
            raise ValueError(f"Text not found on page {page_number} for {result.id}")
            #print(f"Failsafe for {result.id} on page {page_number}")
    else:
        failsafe=False

    # Fetch the content text

    content_text=f"{result.id}: {result.subject}\n\n{queries[result.subject][result.id]}"
    
    doc_mu = annotate_page(doc_mu, result, text_instances, content=content_text, failsafe=failsafe)

# Save reults_list to disk Excel
df = pd.DataFrame([r.dict() for r in results_list])
# Sort by source
df = df.sort_values(by=["subject","source","score"], ascending=[True, True, False])
# Drop if score is below cutoff
df = df[df["score"] >= CUTOFF]

df.to_excel(OUT_FOLDER / f"{FILENAME.stem}.xlsx", index=False)

doc_mu.save(OUT_FOLDER / FILENAME)

print("Document saved with annotations")