# Use the official Python base image
FROM python:3.8

# Set the working directory
WORKDIR /app

# Copy only the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the remaining files into the container
COPY . .

# Expose the port used by Streamlit
EXPOSE 8501

# Define the command to run the Streamlit app
CMD ["streamlit", "run", "👋_Hello.py"]