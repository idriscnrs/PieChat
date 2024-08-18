import gradio as gr

from .config import GlobalConfig
from .engine import PieChat
from .make_vdb import make_vdb


def run():
    config = GlobalConfig()

    if config.make_vdb:
        make_vdb(config)
    else:
        pie_chat = PieChat(**config.export(), llm_config=config.llm_config)
        gr.ChatInterface(pie_chat.chat).launch(share=True)
