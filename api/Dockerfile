FROM python:3.11-slim

WORKDIR /app

# Install API dependencies
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code
COPY api/ ./api/

# Expose port
EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
