# Step 1: Use an official light Python runtime as a parent image
FROM python:3.11-slim

# Step 2: Set the working directory inside the container to /app
WORKDIR /app

# Step 3: Install system dependencies needed for libraries like FAISS or soundfile
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy the requirements file into the container first
# (This leverages Docker caching so rebuilding is faster if requirements don't change)
COPY requirements.txt .

# Step 5: Install your Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Copy the rest of your local backend directory files into the container
COPY . .

# Step 7: Expose port 5000 so the container can talk to your React frontend
EXPOSE 5000

# Step 8: Define the command to run your Flask application
CMD ["python", "app.py"]