# OpenAI API Compatibility

Review date: 2026-06-13

Compatibility status: historical workshop examples; not current integration guidance

## Preserved SDK Boundary

The workshop intentionally pins `openai==0.28.1` and uses the older module-level
Python surface, including `openai.Completion.create`,
`openai.ChatCompletion.create`, and `openai.Embedding.create`. The fine-tuning
lesson also shows the historical `openai api fine_tunes.create` CLI shape.

These calls are preserved so the existing workshop remains internally
consistent. They have not been credentialed or live-tested against the current
OpenAI API and should not be copied into a new integration without migration.

## Affected Examples

- `pages/1_🧐_Getting_Started.py` describes the historical GPT-3
  `davinci`/`ada` capability hierarchy.
- `pages/2_⚡️_API.py` shows Completion, ChatCompletion,
  `gpt-3.5-turbo`, `gpt-3.5-turbo-0301`, and old documentation links.
- `pages/2_📝_Embeddings.py`, `pages/3_🔍_Text_Search.py`,
  `generate_industry_embeddings.py`, `embeddings_demo_step4.py`, and
  `utils/generate.py` use `text-embedding-ada-002` through
  `openai.Embedding.create`.
- `pages/8_🦾_FineTuning.py`, `customer_cluster.py`, `demo.py`, and
  `utils/generate.py` use the legacy `text-davinci-003` Completions model or
  fine-tuning commands.
- `embeddings_demo_step4.py` and `utils/generate.py` also use
  `openai.ChatCompletion.create` with `gpt-3.5-turbo`.

## Current Official References

OpenAI's current documentation should be rechecked when a migration begins:

- [Models](https://developers.openai.com/api/docs/models) describes current
  model selection and states that current models are available through the
  Responses API and client SDKs.
- [All models](https://developers.openai.com/api/docs/models/all) distinguishes
  current, older, and deprecated model families.
- [`text-embedding-3-small`](https://developers.openai.com/api/docs/models/text-embedding-3-small)
  is documented as a newer, improved ada-class embedding model.
- [`davinci-002`](https://developers.openai.com/api/docs/models/davinci-002)
  is documented on the legacy Completions API rather than as current general
  integration guidance.

These links establish that the workshop surface is historical. They do not
select a replacement model or guarantee that a one-line model substitution is
compatible with the workshop's prompts, vector dimensions, response parsing,
fine-tuning data, token counting, or cost examples.

## Migration Boundary

A dedicated migration must review SDK construction, Responses API request and
response shapes, model availability, embedding dimensions, token accounting,
fine-tuning commands and data, exception handling, retries, and every
learner-visible code sample together. It must use credentialed tests in an
explicit environment and preserve the default no-network `make check` gate.

Until then:

- keep the compatibility warning visible on affected learner pages
- keep API keys local and out of source, logs, caches, and screenshots
- do not present historical prices, model rankings, or endpoint shapes as
  current facts
- do not silently mix current documentation with code that still requires the
  0.28.1 SDK surface
