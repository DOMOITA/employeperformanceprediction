# Streamlit App Deployment Guide

## 1. Streamlit Community Cloud (Recommended - FREE)

### Prerequisites:
- GitHub account
- Your code pushed to a GitHub repository

### Steps:
1. **Push your code to GitHub:**
   ```bash
   git add .
   git commit -m "Add Streamlit app"
   git push origin main
   ```

2. **Go to [share.streamlit.io](https://share.streamlit.io)**

3. **Connect your GitHub account and select:**
   - Repository: `DOMOITA/employeperformanceprediction`
   - Branch: `main`
   - Main file path: `employee_performance_app/streamlit_app.py`

4. **Click Deploy!**

### Requirements file:
Make sure you have `requirements_streamlit.txt` in your repo root or app folder.

---

## 2. Heroku Deployment

### Prerequisites:
- Heroku account
- Heroku CLI installed

### Files needed:

#### Procfile (create in app root):
```
web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

#### setup.sh (create in app root):
```bash
mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
```

### Deployment steps:
```bash
# Install Heroku CLI
# Login to Heroku
heroku login

# Create Heroku app
heroku create your-app-name

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

---

## 3. Railway Deployment

### Steps:
1. Go to [railway.app](https://railway.app)
2. Connect your GitHub account
3. Select your repository
4. Railway will auto-detect Streamlit and deploy

### Configuration:
- Start command: `streamlit run employee_performance_app/streamlit_app.py --server.port $PORT`
- Port: 8080 (or whatever Railway assigns)

---

## 4. Render Deployment

### Steps:
1. Go to [render.com](https://render.com)
2. Connect your GitHub account
3. Create a new Web Service
4. Select your repository

### Configuration:
- Build Command: `pip install -r requirements_streamlit.txt`
- Start Command: `streamlit run employee_performance_app/streamlit_app.py --server.port $PORT --server.address 0.0.0.0`

---

## 5. Local Network Deployment

### For testing on local network:
```bash
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Then access via: `http://YOUR_LOCAL_IP:8501`

---

## 6. Docker Deployment

### Dockerfile (create in app root):
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements_streamlit.txt .
RUN pip install -r requirements_streamlit.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and run:
```bash
docker build -t employee-performance-app .
docker run -p 8501:8501 employee-performance-app
```

---

## Troubleshooting

### Common Issues:

1. **Missing requirements.txt**: Make sure all dependencies are listed
2. **File paths**: Use relative paths in your code
3. **Port configuration**: Different platforms use different port configurations
4. **Model files**: Ensure model files are included in your repository

### Testing locally:
```bash
# Activate virtual environment
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate   # Linux/Mac

# Install requirements
pip install -r requirements_streamlit.txt

# Run the app
streamlit run streamlit_app.py
```

---

## Recommended Deployment Order:

1. **Start with Streamlit Community Cloud** (easiest and free)
2. **Try Railway or Render** (simple and reliable)
3. **Use Heroku** (if you need more control)
4. **Docker deployment** (for containerized environments)

Choose the platform that best fits your needs and technical requirements!
