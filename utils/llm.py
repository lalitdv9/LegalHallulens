import os
import time
from typing import Optional

import certifi
import dotenv

try:
    import httpx
except ImportError:
    httpx = None

try:
    from google import genai
except ImportError:
    genai = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


dotenv.load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


SUPPORTED_PROVIDERS = {"openrouter", "gemini", "ollama"}
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
openrouter_client = None


def _get_env_provider() -> Optional[str]:
    provider = os.getenv("LLM_PROVIDER") or os.getenv("DEFAULT_LLM_PROVIDER")
    if provider:
        normalized = provider.strip().lower()
        if normalized in SUPPORTED_PROVIDERS:
            return normalized
    return None


def _resolve_provider(model: str, provider: Optional[str] = None) -> str:
    if not model:
        raise ValueError("Model parameter is required.")

    if provider:
        normalized = provider.strip().lower()
        if normalized not in SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}")
        return normalized

    model_lower = model.strip().lower()
    env_provider = _get_env_provider()

    if model_lower.startswith("ollama/"):
        return "ollama"

    if model_lower.startswith("gemini"):
        return "gemini"

    if "/" not in model_lower and ":" in model_lower:
        return "ollama"

    if env_provider == "ollama" and "/" not in model_lower:
        return "ollama"

    if env_provider == "gemini" and (
        model_lower.startswith("google/") or "/" not in model_lower
    ):
        return "gemini"

    return "openrouter"


def _normalize_model(model: str, provider: str) -> str:
    normalized = model.strip()

    if provider == "ollama" and normalized.lower().startswith("ollama/"):
        return normalized.split("/", 1)[1]

    if provider == "gemini" and normalized.lower().startswith("google/"):
        return normalized.split("/", 1)[1]

    return normalized


def _initialize_openrouter_client():
    global openrouter_client

    if openrouter_client is not None:
        return openrouter_client

    if OpenAI is None:
        raise ImportError("openai package is not installed. Install with: pip install openai")

    if httpx is None:
        raise ImportError("httpx package is not installed. Install with: pip install httpx")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set in environment variables.")

    openrouter_client = OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        http_client=httpx.Client(verify=False),
    )
    return openrouter_client


def _call_openrouter(prompt, model, temperature=0.0, max_retries=5, max_tokens=None):
    client = _initialize_openrouter_client()

    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            }

            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as error:
            error_str = str(error)

            if "429" in error_str or "rate_limit" in error_str.lower():
                if attempt < max_retries - 1:
                    wait_time = min(120, 10 * (2 ** attempt))
                    print(
                        f"   Rate limit (429). Waiting {wait_time}s before retry {attempt + 2}/{max_retries}..."
                    )
                    time.sleep(wait_time)
                else:
                    print("   Max retries exhausted. Try again in a few minutes.")
                    raise
            else:
                if attempt < max_retries - 1:
                    wait_time = 3 * (attempt + 1)
                    print(f"   Error: {error_str[:100]}... Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise


def _call_gemini_native(prompt, model, temperature=0.0, max_retries=5, max_tokens=None):
    if genai is None:
        raise ImportError(
            "google-genai package is not installed. Install with: pip install google-genai"
        )

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set in environment variables.")

    client = genai.Client(api_key=api_key)

    for attempt in range(max_retries):
        try:
            config = {"temperature": temperature}
            if max_tokens is not None:
                config["max_output_tokens"] = max_tokens

            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            return response.text
        except Exception as error:
            error_str = str(error)
            if "429" in error_str or "rate_limit" in error_str.lower() or "quota" in error_str.lower():
                if attempt < max_retries - 1:
                    wait_time = min(120, 10 * (2 ** attempt))
                    print(
                        f"   Gemini rate limit/quota. Waiting {wait_time}s before retry {attempt + 2}/{max_retries}..."
                    )
                    time.sleep(wait_time)
                else:
                    print("   Max retries exhausted for Gemini. Try again later.")
                    raise
            else:
                if attempt < max_retries - 1:
                    wait_time = 3 * (attempt + 1)
                    print(f"   Gemini error: {error_str[:100]}... Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise


def _call_ollama(
    prompt,
    model,
    temperature=0.0,
    max_retries=5,
    max_tokens=None,
    json_mode=False,
    system_prompt=None,
):
    try:
        from ollama import chat
    except ImportError as error:
        raise ImportError("ollama package is not installed. Install with: pip install ollama") from error

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    options = {"temperature": temperature}
    if max_tokens is not None:
        options["num_predict"] = max_tokens

    for attempt in range(max_retries):
        try:
            response = chat(
                model=model,
                messages=messages,
                options=options,
                format="json" if json_mode else None,
            )
            return response.message.content
        except Exception as error:
            if attempt < max_retries - 1:
                wait_time = 2 * (attempt + 1)
                print(f"   Ollama error: {str(error)[:100]}... Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise


def call_llm(
    prompt,
    model,
    temperature=0.0,
    max_retries=5,
    max_tokens=None,
    provider=None,
    json_mode=False,
    system_prompt=None,
):
    if not model:
        raise ValueError("Model parameter is required.")

    resolved_provider = _resolve_provider(model, provider)
    resolved_model = _normalize_model(model, resolved_provider)

    if resolved_provider == "gemini":
        return _call_gemini_native(
            prompt,
            model=resolved_model,
            temperature=temperature,
            max_retries=max_retries,
            max_tokens=max_tokens,
        )

    if resolved_provider == "ollama":
        return _call_ollama(
            prompt,
            model=resolved_model,
            temperature=temperature,
            max_retries=max_retries,
            max_tokens=max_tokens,
            json_mode=json_mode,
            system_prompt=system_prompt,
        )

    return _call_openrouter(
        prompt,
        model=resolved_model,
        temperature=temperature,
        max_retries=max_retries,
        max_tokens=max_tokens,
    )




def count_tokens(prompt, model, verbose=True, provider=None):
    if not model:
        raise ValueError("Model parameter is required for token counting.")

    resolved_provider = _resolve_provider(model, provider)
    resolved_model = _normalize_model(model, resolved_provider)
    model_lower = resolved_model.lower()
    token_count = None
    method = None

    if resolved_provider == "ollama":
        token_count = len(prompt) // 4
        method = "approximate (4 chars/token)"
    elif resolved_provider == "gemini" or model_lower.startswith("gemini"):
        if genai is None:
            raise ImportError(
                "google-genai is required for Gemini token counting. Install with: pip install google-genai"
            )
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        result = client.models.count_tokens(model=resolved_model, contents=prompt)
        token_count = result.total_tokens
        method = "google genai (exact)"
    elif "openai/" in model_lower or model_lower.startswith("gpt") or model_lower.startswith("o1"):
        try:
            import tiktoken
        except ImportError as error:
            raise ImportError("tiktoken is required for OpenAI models. Install with: pip install tiktoken") from error

        if "gpt-4" in model_lower or "gpt-4o" in model_lower:
            encoding = tiktoken.encoding_for_model("gpt-4")
        elif "gpt-3.5" in model_lower:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        else:
            encoding = tiktoken.get_encoding("cl100k_base")

        token_count = len(encoding.encode(prompt))
        method = "tiktoken (exact)"
    elif "anthropic/" in model_lower or "claude" in model_lower:
        try:
            import anthropic
        except ImportError as error:
            raise ImportError(
                "anthropic is required for Claude models. Install with: pip install anthropic"
            ) from error

        client = anthropic.Anthropic()
        token_count = client.count_tokens(prompt)
        method = "anthropic (exact)"
    elif "llama" in model_lower or "meta-llama/" in model_lower:
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "transformers is required for Llama models. Install with: pip install transformers"
            ) from error

        try:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
            token_count = len(tokenizer.encode(prompt))
            method = "transformers (exact)"
        except Exception:
            token_count = len(prompt) // 4
            method = "approximate (4 chars/token)"
            if verbose:
                print("Warning: Using approximate token counting (gated model access unavailable)")
    else:
        raise ValueError(
            f"Token counting not supported for model: {model}. Supported providers: openrouter, gemini, ollama"
        )

    if verbose:
        print(f"Token Count: {token_count:,} tokens")
        print(f"   Method: {method}")
        print(f"   Model: {get_model_name(model, provider=resolved_provider)}")

    return token_count


def get_model_name(model, provider=None):
    if not model:
        raise ValueError("Model parameter is required. You must specify which model you're using.")

    resolved_provider = _resolve_provider(model, provider)
    resolved_model = _normalize_model(model, resolved_provider)

    if resolved_provider == "ollama" and not model.lower().startswith("ollama/"):
        return f"ollama/{resolved_model}"

    if resolved_provider == "gemini" and "/" not in model:
        return resolved_model

    return model


def call_gemini(prompt, model=DEFAULT_GEMINI_MODEL, temperature=0, max_tokens=None):
    return call_llm(
        prompt,
        model=model,
        temperature=temperature,
        max_retries=5,
        max_tokens=max_tokens,
        provider="gemini" if model.startswith("gemini") else None,
    )


print("LLM helper function defined (provider routing: OpenRouter, Gemini, Ollama)")
