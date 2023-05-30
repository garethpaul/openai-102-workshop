# Use the official Python base image
FROM python:3.8

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Copy the embeddings
COPY embeddings.pkl .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit app into the container
COPY app.py .

# Expose the port used by Streamlit
EXPOSE 8501

# Define the command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]
