import yaml


class Config:
    huggingface_hub_token: str
    device: str
    split_on_sentences: bool
    log_prompts: bool
    log_llm_responses: bool

    def __init__(
            self,
            token: str,
            device: str,
            split_on_sentences: bool,
            log_prompts: bool,
            log_llm_responses: bool
    ):
        self.huggingface_hub_token = token
        self.device = device
        self.split_on_sentences = split_on_sentences
        self.log_prompts = log_prompts
        self.log_llm_responses = log_llm_responses


def load_config(file_path: str) -> Config:
    with open(file_path, 'r') as stream:
        config_dict = yaml.safe_load(stream)['app-config']

    return Config(
        config_dict['huggingface-hub-token'],
        config_dict['device'],
        config_dict['split-on-sentences'],
        config_dict['log-prompts'],
        config_dict['log-llm-responses']
    )
