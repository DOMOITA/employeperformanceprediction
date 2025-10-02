#!/bin/bash
echo "Starting INX Employee Performance Streamlit App..."
echo

# Navigate to app directory
cd "$(dirname "$0")"

# Activate virtual environment (if exists)
if [ -f "../.venv/bin/activate" ]; then
    source ../.venv/bin/activate
    echo "Virtual environment activated"
fi

# Install requirements if needed
pip install -r requirements_streamlit.txt

# Run the Streamlit app
streamlit run streamlit_app.py --server.port 8501
