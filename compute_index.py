from pathlib import Path
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.ingestion import IngestionPipeline

#from llama_index.core.node_parser import SentenceSplitter

from llama_index.readers.file import PyMuPDFReader

# Traverse one parent folder up
data_folder=Path('data')

out_folder=Path("annoterat")
index_folder=Path("bp_index")
filename=Path('budgetpropositionen-for-2025-hela-dokumentet-prop.-2024251 19.38.11.pdf')
loader = PyMuPDFReader()
docs = loader.load(file_path=data_folder / filename)

index_folder.mkdir(parents=True, exist_ok=True)

print("Number of documents:", len(docs))

from llama_index.core.node_parser import SentenceSplitter
model_name ='KBLab/sentence-bert-swedish-cased'

from torch import backends
if backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
embed_model = HuggingFaceEmbedding(model_name=model_name, trust_remote_code=True, device=device)

# create the pipeline with transformations
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=256, chunk_overlap=20),
        embed_model,
    ]
)

# run the pipeline
nodes = pipeline.run(documents=docs, show_progress=True)

# Building index from nodes
index = VectorStoreIndex(nodes, embed_model=embed_model)

# Save index
index.storage_context.persist(persist_dir=index_folder)