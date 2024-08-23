# Use the official Python image from the Docker Hub
FROM python:3.12.2

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose port 5000
EXPOSE 5000

# Define the command to run the application
CMD ["python", "app.py"]
