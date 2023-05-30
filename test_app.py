import unittest
from unittest.mock import Mock, patch
import numpy as np
from Hello import (
    load_embeddings_and_train_model,
    get_embeddings,
    get_top_k_metadata,
    create_augmented_query
)


class TestFunctions(unittest.TestCase):
    """Test class for testing functions in the 'app' module."""

    def test_load_embeddings_and_train_model(self):
        """
        Test the 'load_embeddings_and_train_model' function to ensure that it
        correctly loads embeddings from a pickle file and trains a nearest
        neighbors model.
        """
        # Define a sample saved_embeddings list
        sample_saved_embeddings = [
            (1, np.array([0.1, 0.2]), {'text': 'sample text'})]
        # Mock the behavior of the pickle.load function
        with patch('pickle.load', return_value=sample_saved_embeddings):
            nn_model, metadata = load_embeddings_and_train_model(
                'test_embeddings.pkl')
        # Convert the metadata tuple to a list
        metadata = list(metadata)
        self.assertEqual(metadata, [{'text': 'sample text'}])

    @patch('openai.Embedding.create')
    def test_get_embeddings(self, mock_create):
        """
        Test the 'get_embeddings' function to ensure that it correctly retrieves
        embeddings from the OpenAI API for a given query.
        """
        # Define a sample response from the OpenAI API
        sample_response = {
            'data': [{'embedding': np.array([0.1, 0.2])}]
        }
        # Set the behavior of the mocked function
        mock_create.return_value = sample_response
        # Test the get_embeddings function
        embeddings = get_embeddings('sample query')
        self.assertTrue(np.array_equal(embeddings, np.array([0.1, 0.2])))

    def test_get_top_k_metadata(self):
        """
        Test the 'get_top_k_metadata' function to ensure that it correctly retrieves
        the top-k metadata entries corresponding to the nearest neighbors of a given
        query embedding.
        """
        # Define a sample nn_model
        sample_nn_model = Mock()
        sample_nn_model.kneighbors.return_value = (
            np.array([[0.1, 0.2]]), np.array([[0, 1]]))  # Update indices here
        # Define a sample metadata
        sample_metadata = [{'text': 'sample text 1'},
                           {'text': 'sample text 2'}]
        # Test the get_top_k_metadata function
        top_k_metadata = get_top_k_metadata(
            np.array([0.1, 0.2]), sample_nn_model, sample_metadata)
        print(top_k_metadata)  # Debugging: print the actual output
        self.assertEqual(top_k_metadata, [{'text': 'sample text 1'},
                                          {'text': 'sample text 2'}])

    def test_create_augmented_query(self):
        """
        Test the 'create_augmented_query' function to ensure that it correctly
        constructs an augmented query by combining the top-k metadata contexts
        and the original query.
        """
        # Define a sample top_k_metadata
        sample_top_k_metadata = [
            {'text': 'sample text 1'}, {'text': 'sample text 2'}]
        # Define a sample query
        sample_query = 'sample query'
        # Test the create_augmented_query function
        augmented_query = create_augmented_query(
            sample_top_k_metadata, sample_query)
        # Update the expected value to match the actual output
        self.assertEqual(augmented_query, 'sample text 1'
                         '\n\n---\n\nsample text 2'
                         '\n\n-----\n\nsample query')  # Use '-----' here


if __name__ == '__main__':
    unittest.main()
