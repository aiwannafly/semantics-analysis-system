import yaml


class Config:
    show_class_predictions: bool
    llm: str
    huggingface_hub_token: str
    device: str
    show_explanation: bool
    split_on_sentences: bool
    log_prompts: bool
    log_llm_responses: bool

    def __init__(
            self,
            show_class_predictions: bool,
            llm: str,
            token: str,
            device: str,
            show_explanation: bool,
            split_on_sentences: bool,
            log_prompts: bool,
            log_llm_responses: bool
    ):
        self.show_class_predictions = show_class_predictions
        self.llm = llm
        self.huggingface_hub_token = token
        self.device = device
        self.show_explanation = show_explanation
        self.split_on_sentences = split_on_sentences
        self.log_prompts = log_prompts
        self.log_llm_responses = log_llm_responses

        if self.show_explanation:
            self.log_llm_responses = True


def load_config(file_path: str) -> Config:
    with open(file_path, 'r') as stream:
        config_dict = yaml.safe_load(stream)['app-config']

    return Config(
        config_dict['show-class-predictions'],
        config_dict['llm'],
        config_dict['huggingface-hub-token'],
        config_dict['device'],
        config_dict['show-explanation'],
        config_dict['split-on-sentences'],
        config_dict['log-prompts'],
        config_dict['log-llm-responses']
    )
