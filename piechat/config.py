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
            if data_field.metadata.get(meta_filter):
                config_value = getattr(self, data_field.name)
                if isinstance(config_value, Config):
                    sub_config_dict = {
                        f"{config_value.config_name.lower()}.{k}": v
                        for k, v in config_value.export().items()
                    }
                    config_dict.update(sub_config_dict)
                else:
                    config_dict[data_field.name] = config_value
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
class GlobalConfig(Config):
    config_name: str = "GLOBAL"

    models_dir: Path = field(
        default=Path.cwd() / "models",
        metadata={"converter": Path, "export": False}
    )

    llm_name: str = field(
        default="model", metadata={"converter": str, "export": False}
    )
    llm_path: Path = field(
        default="", metadata={"converter": Path, "export": True}
    )

    embedding_name: str = field(
        default="embedding", metadata={"converter": str, "export": False}
    )
    embedding_path: Path = field(
        default="", metadata={"converter": Path, "export": True}
    )

    vdb_path: Path = field(
        default=Path.cwd() / "vdb", metadata={"converter": Path, "export": True}
    )
    data_path: Path = field(
        default=Path.cwd() / "data", metadata={"converter": Path, "export": False}
    )

    stop_tokens: list = field(
        default_factory=lambda: "eos,bos",
        metadata={"converter": lambda x: x.split(","), "export": True}
    )

    make_vdb: bool = field(
        default=False, metadata={"converter": bool, "export": False}
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.llm_path == Path(""):
            self.llm_path = self.models_dir / self.llm_name

        if self.embedding_path == Path(""):
            self.embedding_path = self.models_dir / self.embedding_name
