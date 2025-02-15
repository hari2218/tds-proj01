FROM python:3.9-slim

LABEL maintainer="Hariharan C <hariharan.chandran@straive.com>"

ENV PYTHONHTTPSVERIFY=0
ENV CURLOPT_SSL_VERIFYHOST=0
ENV CURLOPT_SSL_VERIFYPEER=0

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg nodejs npm && \
    pip install --no-cache-dir virtualenv uv && \
    virtualenv venv

# Install certificates
RUN apt-get update && apt-get install -y ca-certificates && update-ca-certificates

# Activate the virtual environment and install Python packages
RUN . venv/bin/activate && uv pip install --no-cache-dir -r requirements.txt

# Install prettier globally using npm
RUN npm install -g prettier

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the application code
COPY . .

# Download the datagen.py script
# ADD https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py /app/datagen.py

# Run the datagen.py script with the specified argument
RUN . venv/bin/activate && uv run https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py --root /data hariharan.chandran@straive.com

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["/app/venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
