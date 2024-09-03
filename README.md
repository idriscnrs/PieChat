# PieChat

## How to setup PieChat

**1. Download the preprocessed IDRIS documentation**
```bash
git lfs install
git clone https://huggingface.co/datasets/CNRS-IDRIS/idris_doc_dataset
```

**2. Clone the PieChat repository**
```bash
git clone https://github.com/idriscnrs/PieChat.git
cd PieChat
```

**3. Make your configuration file <br>**
Set the wright pathes, choose the devices, the models configurations...
`config.ini` and `config_sofi.ini` are examples of configuration files.


**4. Create the Vector Database <br>**
Warning: A Vector Database is associated to a specific Embedding model. If you change the model, you need to recreate the VDB.
```bash
python -m piechat --config_file ./config.ini --make_vdb
```

**5. Launch the PieChat server**
```bash
python -m piechat --config_file ./config.ini
```


## With Sofi
You need to use the [following image](https://hub.docker.com/layers/cnrsidris/spellm/new/images/sha256-6930be29304dfb64518527ef6dc429781d1395685f13ce22b0d8c9781a5ea715?context=repo) to run PieChat with Sofi: `cnrsidris/spellm:new` <br>

Then install the python libraries:
```bash
pip install langchain_huggingface
pip install langchain_chroma
pip install gradio_rag_sources
```

At this time, the P2P communication between GPUs is not set on Sofi. You need to disable it with `NCCL_P2P_DISABLE=1`. <br>
Also, if you want to use more than one GPU, you need to specify the `CUDA_VISIBLE_DEVICES` environment variable. <br>

Here is an example of command to launch PieChat with Sofi on 8 GPUs:
```bash
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m piechat --config_file ./config_sofi.ini
```
