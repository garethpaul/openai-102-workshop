# Use the official Python base image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy only the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Get the embeddings file
RUN apt-get update && apt-get install -y wget
RUN wget -O embeddings.pkl https://storage.googleapis.com/artifacts.gjones-webinar.appspot.com/embeddings.pkl

# Copy all the remaining files into the container
COPY . .

# Expose the port used by Streamlit
EXPOSE 8501

# Define the command to run the Streamlit app
CMD ["streamlit", "run", "👋_Hello.py"]