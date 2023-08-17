DEFAULT_VOCAB_SIZE = 20000
DEFAULT_MAX_LENGTH = 250

def load_config(path_to_config="config.toml"):
    try:
        with open("config.toml", "rb") as f:
            config_data = tomllib.load(f)
            VARIABLES = config_data.get("HYPERPARAMETERS", {})
            VOCAB_SIZE = VARIABLES.get("VOCAB_SIZE", DEFAULT_VOCAB_SIZE)
            MAX_LENGTH = VARIABLES.get("MAX_LENGTH", DEFAULT_MAX_LENGTH)    
    except Exception:
        VOCAB_SIZE = DEFAULT_VOCAB_SIZE
        MAX_LENGTH = DEFAULT_MAX_LENGTH
    return VOCAB_SIZE, MAX_LENGTH