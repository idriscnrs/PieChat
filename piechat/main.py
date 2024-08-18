from .config import GlobalConfig
from .engine import PieChat
from .gradio import launch_gradio
from .make_vdb import make_vdb


def run():
    config = GlobalConfig()

    if config.make_vdb:
        make_vdb(config)
    else:
        pie_chat = PieChat(**config.export(), llm_config=config.llm_config)
        config.like_data_path.mkdir(exist_ok=True)
        launch_gradio(pie_chat, config)
