FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Set environment variable
ENV AIPROXY_TOKEN="eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6ImhhcmloYXJhbi5jaGFuZHJhbkBzdHJhaXZlLmNvbSJ9.viQ4bynZwvWld8CWCAsq2GmqJUNvK3ERXK12P5FSUJc"

# Copy the requirements file
COPY requirements.txt .

# Install virtualenv
RUN pip install --no-cache-dir virtualenv

# Create a virtual environment
RUN virtualenv venv

# Activate the virtual environment and install dependencies
RUN . venv/bin/activate && pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -U UV && \
    apt-get update && apt-get install -y ffmpeg && apt-get clean && rm -rf /var/lib/apt/lists/* && \
    npm install -g prettier

# Copy the application code
COPY . .

# Download the datagen.py script
ADD https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py /app/datagen.py

# Run the datagen.py script with the specified argument
RUN . venv/bin/activate && python datagen.py --data /data hariharan.chandran@straive.com

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["/app/venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]