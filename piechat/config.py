import re
from argparse import ArgumentParser, Namespace
from configparser import ConfigParser
from dataclasses import dataclass, field, fields
from pathlib import Path


def add_spaces_to_string(s):
    lines = s.splitlines()
    return lines[0] + "\n" + '\n'.join(' ' * 4 + line for line in lines[1:])


@dataclass(kw_only=True)
class Config:
    config_name: str = "DEFAULT"
    sub_config: bool = False
    config_file: Path = field(
        default=Path.cwd() / "config.ini", metadata={"converter": Path, "export": False}
    )

    def __post_init__(self) -> None:
        self.fields_names = [field.name for field in fields(self)]

        args = self.get_args()
        if args.config_file is not None:
            self.config_file = args.config_file

        self.from_file(Path(self.config_file))
        self.from_args(args)
        self.correct_type()

    def from_file(self, config_path) -> None:
        confparser = ConfigParser()
        if not config_path.exists():
            if not self.sub_config:
                print(f"Config file not found: {config_path}")
                print("Using default values and command line arguments only.")
            return None

        confparser.read(config_path)
        for key, val in confparser[self.config_name].items():
            if key in self.fields_names:
                setattr(self, key, val)
            else:
                print(f"Unknown key from config file, ignoring: {key}")

    def get_args(self) -> Namespace:
        argparser = ArgumentParser()
        for dataclass_field in fields(self):
            if not isinstance(getattr(self, dataclass_field.name), Config):
                argparser.add_argument(
                    f"--{dataclass_field.name}",
                    action="store_true" if dataclass_field.type is bool else "store",
                    default=None,
                )
        args, _ = argparser.parse_known_args()
        return args

    def from_args(self, args) -> None:
        for key, val in vars(args).items():
            if val is not None and key in self.fields_names:
                setattr(self, key, val)

    def correct_type(self) -> None:
        # Convert the values of the config attributes
        for dataclass_field in fields(self):
            converter = dataclass_field.metadata.get("converter", None)
            if converter is not None:
                value = getattr(self, dataclass_field.name)
                if value is not None:
                    self.__setattr__(dataclass_field.name, converter(value))

    def export(self, meta_filter="export") -> dict:
        config_dict = {}
        for data_field in fields(self):
            if data_field.metadata.get(meta_filter) or meta_filter == "all":
                config_value = getattr(self, data_field.name)
                if isinstance(config_value, Config):
                    sub_config_dict = {
                        f"{config_value.config_name.lower()}.{k}": v
                        for k, v in config_value.export(meta_filter).items()
                    }
                    config_dict.update(sub_config_dict)
                else:
                    config_dict[data_field.name] = config_value
        return config_dict

    def export_config(self) -> dict:
        config_dict = self.export("all")
        # Transform Path to str
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        return config_dict

    def __str__(self) -> str:
        string = f"{self.__class__.__qualname__}(\n"
        for f in fields(self):
            config_value = getattr(self, f.name)
            if isinstance(config_value, Config):
                string += " " * 4 + f"{f.name}="
                string += f"{add_spaces_to_string(config_value.__str__())},\n"
            else:
                string += " " * 4 + f"{f.name}={config_value},\n"
        string += ")"
        return string


@dataclass(kw_only=True)
class LLMConfig(Config):
    config_name: str = "LLM"

    llm_path: Path = field(
        default="", metadata={"converter": Path, "export": True}
    )
    max_model_length: int = field(
        default=4096, metadata={"converter": int, "export": True}
    )
    dtype: str = field(
        default="bfloat16", metadata={"converter": str, "export": True}
    )
    gpu_memory_utilization: float = field(
        default=0.80, metadata={"converter": float, "export": True}
    )
    temperature: float = field(
        default=0.75, metadata={"converter": float, "export": True}
    )
    max_tokens: int = field(
        default=1024, metadata={"converter": int, "export": True}
    )
    stop_tokens: list = field(
        default_factory=lambda: "eos,bos",
        metadata={"converter": lambda x: x.split(","), "export": True}
    )
    tensor_parallel_size: int = field(
        default=1, metadata={"converter": int, "export": True}
    )
    llm_header: str = field(
        default="Answer the question with the context.",
        metadata={"converter": str, "export": True}
    )


@dataclass(kw_only=True)
class EmbeddingConfig(Config):
    config_name: str = "EMBEDDING"

    embedding_path: Path = field(
        default="", metadata={"converter": Path, "export": True}
    )
    emb_device: str = field(
        default="cuda", metadata={"converter": str, "export": True}
    )
    no_trust_remote_code: bool = field(
        default=False, metadata={"converter": bool, "export": True}
    )
    attn_implementation: str = field(
        default="flash_attention_2", metadata={"converter": str, "export": True}
    )
    emb_precision: str = field(
        default="float16", metadata={"converter": str, "export": True}
    )


@dataclass(kw_only=True)
class RerankerConfig(Config):
    config_name: str = "RERANKER"

    reranker_path: Path = field(
        default=Path.cwd() / "reranker", metadata={"converter": Path, "export": True}
    )
    reranker_device: str = field(
        default="cuda", metadata={"converter": str, "export": True}
    )
    reranker_guide: str = field(
        default="Retrieve relevant passages that help to answer the query.",
        metadata={"converter": str, "export": True}
    )
    reranker_precision: str = field(
        default="float16", metadata={"converter": str, "export": True}
    )
    no_rerank: bool = field(
        default=False, metadata={"converter": bool, "export": True}
    )


@dataclass(kw_only=True)
class VDBConfig(Config):
    config_name: str = "VDB"

    vdb_path: Path = field(
        default=Path.cwd() / "vdb", metadata={"converter": Path, "export": True}
    )
    data_path: Path = field(
        default=Path.cwd() / "data", metadata={"converter": Path, "export": False}
    )

    max_tokens_per_chunk: int = field(
        default=512, metadata={"converter": int, "export": True}
    )
    nb_tokens_chunk_overlap: int = field(
        default=128, metadata={"converter": int, "export": True}
    )


@dataclass(kw_only=True)
class GlobalConfig(Config):
    config_name: str = "GLOBAL"

    system_prompts_path: Path = field(
        default=Path.cwd() / "system_prompts.txt",
        metadata={"converter": Path, "export": False}
    )

    retrieval_threshold: float = field(
        default=0.2, metadata={"converter": float, "export": False}
    )
    n_retrieved_docs: int = field(
        default=8, metadata={"converter": int, "export": False}
    )
    coef_rerank_retrieve_docs: float = field(
        default=4.0, metadata={"converter": float, "export": True}
    )

    make_vdb: bool = field(
        default=False, metadata={"converter": bool, "export": False}
    )

    saved_data_path: Path = field(
        default=Path.cwd() / "saved_data", metadata={"converter": Path, "export": False}
    )

    llm_config: LLMConfig = field(
        default_factory=lambda: LLMConfig(sub_config=True),
        metadata={"export": False}
    )
    emb_config: EmbeddingConfig = field(
        default_factory=lambda: EmbeddingConfig(sub_config=True),
        metadata={"export": False}
    )
    reranker_config: RerankerConfig = field(
        default_factory=lambda: RerankerConfig(sub_config=True),
        metadata={"export": False}
    )
    vdb_config: VDBConfig = field(
        default_factory=lambda: VDBConfig(sub_config=True),
        metadata={"export": False}
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.system_prompts_path.exists():
            system_prompts_lines = self.system_prompts_path.read_text().splitlines()
            attribute_updates = {}

            for line in system_prompts_lines:
                if re.search(r"\[(.*)\]", line):
                    current_attribute = re.search(r"\[(.*)\]", line).group(1)
                    attribute_updates[current_attribute] = ""
                else:
                    attribute_updates[current_attribute] += line + "\n"

            for attribute, value in attribute_updates.items():
                if attribute == "reranker_guide":
                    self.reranker_config.reranker_guide = value.strip()
                    print(f"Updated {attribute} from system prompts file")

                if attribute == "llm_header":
                    self.llm_config.llm_header = value.strip()
                    print(f"Updated {attribute} from system prompts file")
