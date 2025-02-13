FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies and the UV module
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -U UV

# Copy the application code
COPY . .

# Download the datagen.py script
ADD https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py /app/datagen.py

# Run the datagen.py script with the specified argument
RUN python datagen.py --data /data hariharan.chandran@straive.com

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]