import yaml


class Config:
    huggingface_hub_token: str
    device: str
    log_prompts: bool
    log_llm_responses: bool

    def __init__(self, token: str, device: str, log_prompts: bool, log_llm_responses: bool):
        self.huggingface_hub_token = token
        self.device = device
        self.log_prompts = log_prompts
        self.log_llm_responses = log_llm_responses


def load_config(file_path: str) -> Config:
    with open(file_path, 'r') as stream:
        config_dict = yaml.safe_load(stream)['app-config']

    return Config(
        config_dict['huggingface-hub-token'],
        config_dict['device'],
        config_dict['log-prompts'],
        config_dict['log-llm-responses']
    )
