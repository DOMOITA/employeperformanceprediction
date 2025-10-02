FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements_streamlit.txt .
RUN pip install --no-cache-dir -r requirements_streamlit.txt

# Copy the application
COPY . .

# Expose the port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
