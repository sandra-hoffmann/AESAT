import requests
import time
from abc import ABC, abstractmethod
from config import Config, LLMConfig

# server-side or load problems that can pass; anything else (400, 401, 404)
# will not get better by waiting
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class BaseLLMAPI(ABC):
    def __init__(self, api_base, api_key, model_name, stream=False, max_retries=8,
                 base_delay=30, max_delay=600, timeout=600):
        self.config = LLMConfig(api_base, api_key, model_name, stream)
        self.max_retries = max_retries
        self.base_delay = base_delay      # first wait in seconds, doubled each retry
        self.max_delay = max_delay        # cap per wait
        self.timeout = timeout            # a hanging request counts as a failure

    def _delay(self, retries, response):
        """Seconds to wait before the next attempt: Retry-After if the server
        sends one, otherwise exponential backoff."""
        if response is not None:
            retry_after = response.headers.get("Retry-After", "")
            if retry_after.isdigit():
                return min(int(retry_after), self.max_delay)
        return min(self.base_delay * 2 ** (retries - 1), self.max_delay)

    def _make_request(self, url, headers, data):
        retries = 0
        while True:
            response = None
            try:
                response = requests.post(url, headers=headers, json=data,
                                         timeout=self.timeout)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except (requests.exceptions.RequestException, KeyError, ValueError) as e:
                retries += 1
                print(f"Request error: {type(e).__name__}: {e}")
                if response is not None:
                    print(f"  Status: {response.status_code}")
                    print(f"  Body: {response.text[:500]}")
                print(f"  Prompt length: {len(data['messages'][0]['content'])} chars")

                status = response.status_code if response is not None else None
                if (isinstance(e, requests.exceptions.HTTPError)
                        and status not in RETRYABLE_STATUS):
                    raise Exception(f"Request failed, not retryable: {e}")
                if retries >= self.max_retries:
                    raise Exception(f"Request failed after {self.max_retries} retries: {e}")

                delay = self._delay(retries, response)
                print(f"Request failed, retrying in {delay} seconds "
                      f"(attempt {retries}/{self.max_retries})")
                time.sleep(delay)

    @abstractmethod
    def generate(self, prompt):
        pass


class GPTCallAPI(BaseLLMAPI):
    def generate(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}]
        }
        return self._make_request(
            f"{self.config.api_base}/chat/completions",
            headers,
            data
        )


class ClaudeCallAPI(BaseLLMAPI):
    def generate(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}]
        }
        return self._make_request(
            f"{self.config.api_base}/chat/completions",
            headers,
            data
        )


class DeepSeekCallAPI(BaseLLMAPI):
    def generate(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}]
        }
        return self._make_request(
            f"{self.config.api_base}/chat/completions",
            headers,
            data
        )