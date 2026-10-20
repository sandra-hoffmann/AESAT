import os

# Mixture of Experts Mode
class Config:
    # ChatGPT Configuration
    OPENAI_API_KEY = os.environ.get("KICONNECT_KEY", "")
    OPENAI_API_BASE = "https://chat.kiconnect.nrw/api/v1"
    OPENAI_MODEL = "OpenAI OSS 120B"
    # Claude Configuration
    CLAUDE_API_KEY = os.environ.get("KICONNECT_KEY", "")
    CLAUDE_API_BASE = "https://chat.kiconnect.nrw/api/v1"
    CLAUDE_MODEL = "Mistral-small-4-119b-2603"
    
    # DeepSeek Configuration
    DEEPSEEK_API_KEY = os.environ.get("KICONNECT_KEY", "")
    DEEPSEEK_API_BASE = "https://chat.kiconnect.nrw/api/v1"
    DEEPSEEK_MODEL = "Qwen3.8 MA"
    
    # Code Rule Configuration
    CODE_RULE = "void restart_mab(kissat * solver) {// new code}"
    
    # Solution Generation Configuration
    SOLUTION_COUNT = 20  # Number of solutions to generate
    VARIANT_COUNT = 3   # Number of variants per solution
    CODE_SOURCE_PATH = "prompt/keycode.txt"  # Code source file path
    DATA_PARALLEL_SIZE = 60  # Data parallel size
    MAX_ITERATIONS = 3  # Maximum iterations per solution
    MAX_RETRY_COUNT = 10  # Maximum retry count when parsing response fails

class LLMConfig:
    def __init__(self, api_base, api_key, model_name, stream=False):
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name
        self.stream = stream
