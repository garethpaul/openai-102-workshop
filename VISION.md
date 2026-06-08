## OpenAI 102 Workshop Vision

OpenAI 102 Workshop is a Streamlit learning app for API concepts, embeddings,
search, recommendations, clustering, fine-tuning exercises, and retrieval-style
question answering.

The repository is useful as a hands-on workshop: learners can run the app,
inspect small scripts, and compare generic model responses with responses
grounded in prepared embeddings and metadata.

The goal is to keep the workshop runnable, teachable, and honest about API,
model, and dependency assumptions.

The current focus is:

Priority:

- Preserve the Streamlit lesson flow and local Docker path
- Keep generated caches and prepared data clearly separated from source logic
- Make API-token handling explicit and local to the learner
- Document model, SDK, and Pinecone assumptions when examples depend on them

Next priorities:

- Add a single quickstart that verifies the app boots without paid API calls
- Mark stale API examples before updating them
- Add lightweight tests for embedding-cache loading and nearest-neighbor lookup
- Document which files are workshop fixtures versus generated output

Contribution rules:

- One PR = one focused lesson, dependency, data, or documentation change.
- Do not commit real API keys, customer data, or private workshop material.
- Keep code examples small enough for learners to trace.
- Explain model or API migrations in the README.

## Security And Responsible Use

Workshop users provide their own API credentials. The app should not persist,
print, or transmit those credentials except to the APIs that the user
explicitly enables while running the lesson.

## What We Will Not Merge (For Now)

- Checked-in secrets or private data
- Hidden network calls outside the lesson being demonstrated
- Model upgrades without compatibility notes
- Large cache refreshes that are not reproducible
