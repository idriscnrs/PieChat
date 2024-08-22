from pathlib import Path

import pandas as pd
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from transformers import AutoTokenizer

from .config import GlobalConfig, EmbeddingConfig


def split_markdown(
    meta_df: pd.DataFrame, embedding_path: Path, data_path: Path
):
    print("Splitting markdown files")
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    tokenizer = AutoTokenizer.from_pretrained(embedding_path)
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    r_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", r"(?<=\. )", " ", ""]
    )

    docs = []
    for _, row in meta_df.iterrows():
        if row.isnull()["markdown_path"]:
            continue

        md_str = (data_path / row.markdown_path).read_text()
        if md_str != "None":
            splitted_md = md_splitter.split_text(md_str)
            for doc in splitted_md:
                doc.metadata["source"] = row.source
                doc.metadata["language"] = row.language
                docs.append(doc)
    print(f"Split Markdown parts into {len(docs)} documents")

    docs = r_splitter.split_documents(docs)
    print(f"Split documents into {len(docs)} parts of maximum 500 tokens")
    return docs


def create_vdb(
    docs: list, emb_config: EmbeddingConfig, vdb_path: Path
):
    """Create a vector database from the documents"""
    embedding = HuggingFaceEmbeddings(
        model_name=str(emb_config.embedding_path),  # Does not accept Path
        model_kwargs={
            "device": f"cuda:{emb_config.device_id}",
            "trust_remote_code": not emb_config.no_trust_remote_code,
            # "model_kwargs": {
            #     "attn_implementation": emb_config.attn_implementation,
            #     "torch_dtype": torch.float16
            # } 
        },
    )

    if vdb_path.exists():
        if any(vdb_path.iterdir()):
            raise FileExistsError(
                f"Vector database directory {vdb_path} is not empty"
            )
    else:
        vdb_path.mkdir(parents=True)

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=str(vdb_path),  # Does not accept Path
    )

    print(f"vector database created in {vdb_path}")
    return vectordb


def make_vdb(config: GlobalConfig):
    meta_df = pd.read_csv(config.data_path / "metadata.csv")

    docs = split_markdown(
        meta_df=meta_df,
        embedding_path=config.emb_config.embedding_path,
        data_path=config.data_path
    )

    create_vdb(
        docs=docs,
        emb_config=config.emb_config,
        vdb_path=config.vdb_path
    )
