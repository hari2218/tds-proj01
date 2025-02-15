# tds-proj01 FastAPI Application

This project is a FastAPI application that provides various endpoints for handling tasks, reading files, and processing data. It utilizes SQLite for database operations and includes several utility functions for data manipulation.

## Assumptions/Notes

- The `AIPROXY_TOKEN` environment variable is required.
- **A1:** The `DATA_DIR` is configured to '/data'. Assuming we need to download the data.
- **A3:** Monday is designated as the first day of the week.
- **A8:** Unable to use LLM because of fllowing error: It looks like *OpenAI's* API is blocking the request due to sensitive content policies (likely because it detects a credit card number). OpenAI has strict rules against processing personally identifiable information (PII), including credit card numbers.
- **B10:** Used GET method with path parameters.
- `ffmpeg` and `prettier` installed through `Docker`.

## Project Structure

```
tds-proj01
├── main.py          # Contains the FastAPI application and endpoint definitions
├── Dockerfile           # Instructions to build the Docker image for the application
├── requirements.txt     # Lists the Python dependencies required for the project
└── README.md            # Documentation for the project
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd tds-proj01
   ```

2. **Create a virtual environment (optional but recommended):**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```
   uvicorn src.main:app --reload
   ```

## Docker Instructions

To build and run the Docker image for this application, follow these steps:

1. **Build the Docker image:**
   ```
   docker build -t fastapi-app .
   ```

2. **Run the Docker container:**
   ```
   docker run -d -p 8000:8000 fastapi-app
   ```

The application will be accessible at `http://localhost:8000`.

## Usage

Once the application is running, you can access the API endpoints defined in `main.py`. You can use tools like Postman or curl to interact with the API.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.