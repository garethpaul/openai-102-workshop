from components import common
import streamlit as st

st.markdown("""
# Prerequisites 📚

Here is some reading to get you started.

## OpenAI
[OpenAI](https://openai.com) is an AI research and deployment company. They are focused on building safe and beneficial general-purpose artificial intelligence (AGI). They are also focused on ensuring that AGI is deployed safely and securely.

- [OpenAI API](https://openai.com/blog/openai-api/)
- [OpenAI API Documentation](https://beta.openai.com/docs/introduction)
- [OpenAI API Python SDK](https://github.com/openai/openai-python)
- [OpenAI API Python SDK Documentation](https://openai-python.readthedocs.io/en/latest/)
- [OpenAI Cookbooks](https://github.com/openai/openai-cookbook)
- [OpenAI API Examples](https://platform.openai.com/examples)
- [OpenAI API Playground](https://beta.openai.com/playground)

# What are OpenAI models?
OpenAI models are pre-trained models that can be used to generate text, images, and more. They are trained on a large corpus of data and can be fine-tuned to generate text for a specific domain.

Examples of OpenAI models include:

- [GPT-4](https://openai.com/blog/openai-api/)
- [GPT-3](https://openai.com/blog/openai-api/)
- [CLIP](https://openai.com/blog/clip/)
- [DALL-E](https://openai.com/blog/dall-e/)
- [Codex](https://openai.com/blog/openai-codex/)
- [Jukebox](https://openai.com/blog/jukebox/)

# Getting started with the OpenAI API
The OpenAI API is a powerful tool that can be used to generate text, images, and more. It can be used to generate text for a variety of use cases, including:
- Chatbots
- Content generation
- Creative writing
- Customer support
- Data augmentation
- Data labeling
- Data synthesis
- Data translation
- Data validation
- Data verification

# Getting started with the OpenAI Python SDK
The OpenAI Python SDK is a Python library that can be installed using pip. 

```bash
pip install openai
```

To make a request to the OpenAI API, you need to create an instance of the OpenAI class. You can then use the instance to make requests to the API.

```python
import openai

response = openai.Completion.create(
    engine="davinci",
    prompt="This is a test",
    temperature=0.3,
    max_tokens=60,
    top_p=1,
    frequency_penalty=0.5,
    presence_penalty=0.5,
    stop=["\n", "testing"]
)
```

The api_key can be found in your OpenAI account settings.

The parameters for the request are:
- engine: The engine to use for the request.
- prompt: The prompt to use for the request.
- temperature: The temperature to use for the request.
- max_tokens: The maximum number of tokens to use for the request.
- top_p: The top p to use for the request.
- frequency_penalty: The frequency penalty to use for the request.
- presence_penalty: The presence penalty to use for the request.
- stop: The stop to use for the request.

More information about the params can be found in the [OpenAI API Documentation](https://beta.openai.com/docs/introduction).

The response from the API is a JSON object.

```json

{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "text": " of the Emergency Broadcast System. This is only a test.\""
    }
  ],
  "created": 1685500360,
  "id": "cmpl-7M5vc1wPrHOsHCNsCT8NHGgtPM8BP",
  "model": "davinci",
  "object": "text_completion",
  "usage": {
    "completion_tokens": 12,
    "prompt_tokens": 4,
    "total_tokens": 16
  }
}
```

To learn more about the API visit the [OpenAI API Documentation](https://beta.openai.com/docs/introduction).

""")
common.add_logo()
