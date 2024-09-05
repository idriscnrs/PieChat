FROM vllm/vllm-openai:v0.5.5

# Set the working directory
WORKDIR /PieChat

# Copy the current directory contents into the container
COPY . /PieChat

# Install any needed packages
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    git-lfs wget

RUN pip install --no-cache-dir \
    pandas \
    transformers \
    gradio \
    chromadb \
    langchain \
    langchain_chroma \
    langchain_huggingface \
    langchain_text_splitters \
    sentence_transformers \
    gradio_rag_sources \
    pysqlite3-binary

# Download IDRIS documentation
RUN cd / \
    && git lfs install \
    && git clone https://huggingface.co/datasets/CNRS-IDRIS/idris_doc_dataset

# Set port and server name for gradio
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Set the entrypoint
RUN chmod +x /PieChat/entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
