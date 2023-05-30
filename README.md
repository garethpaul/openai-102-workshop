# Level 2 - API 101? What’s an embedding? How do you use APIs?

# Code Sample
<https://github.com/garethpaul/gpt-docs-api>

## Local Install
To get started you can run the following commands:

```make build run```

You should then be able to access streamlit in your browser with
http://localhost:8502

For more information see the `Makefile`

## Docker
If you'd prefer to use Docker to build and run. 

```docker build -t streamlit-app .```
```docker run -p 8502:8502 streamlit-app```


# App Information

The app.py file provides a Streamlit application for generating responses to 
user queries using OpenAI's GPT-3 models. The application uses embeddings and
metadata to create augmented queries for generating detailed and context-aware
answers.

The application offers two modes of operation:
1. Use Embeddings: The application retrieves embeddings for the query, finds
   the top-k closest metadata entries, creates an augmented query, and uses
   OpenAI's ChatCompletion API to generate a response. The URLs for the top-k
   metadata are also displayed as sources.
2. Generic Response: If "Use Embeddings" is not selected, the application uses
   OpenAI's Completion API to generate a generic response to the query.

Functions in this module include:
- load_embeddings_and_train_model(pickle_file_path): Load embeddings from a
  pickle file and train a nearest neighbors model.
- get_embeddings(query): Retrieve embeddings for a query using OpenAI's
  Embedding API.
- get_top_k_metadata(embedding, nn_model, metadata): Retrieve the top-k
  metadata entries corresponding to the nearest neighbors.
- create_augmented_query(top_k_metadata, query): Construct an augmented query
  by combining top-k metadata contexts with the query.
- get_model_response(augmented_query): Generate a response to the augmented
  query using OpenAI's ChatCompletion API.
- get_generic_response(query): Generate a generic response to the query using
  OpenAI's Completion API.
- main(): The main function for the Streamlit app, responsible for handling
  user input and generating responses.


## Example usage:
To run the Streamlit application, execute the following command:
`streamlit run app.py`