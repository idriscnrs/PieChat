FROM vllm/vllm-openai:v0.5.5

# Copy the current directory contents into the container
COPY . .

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
RUN git lfs install
RUN git clone https://huggingface.co/datasets/CNRS-IDRIS/idris_doc_dataset

# Create the vector database
RUN NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python3 -m piechat --config_file ./config_sofi.ini \
    --make_vdb --data_path ./idris_doc_dataset

# Run the application
RUN NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python3 -m piechat --config_file ./config_sofi.ini --run_server
