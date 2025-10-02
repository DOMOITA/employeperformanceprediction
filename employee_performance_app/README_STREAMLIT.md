# INX Employee Performance Prediction - Streamlit App

A modern web application for predicting employee performance using machine learning, built with Streamlit.

## 🚀 Quick Start

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements_streamlit.txt
   ```

2. **Run the application:**
   ```bash
   streamlit run streamlit_app.py
   ```

   Or use the convenience scripts:
   - Windows: Double-click `run_app.bat`
   - Linux/Mac: `./run_app.sh`

3. **Open your browser to:** `http://localhost:8501`

## 🌐 Deployment Options

### Option 1: Streamlit Community Cloud (Recommended - FREE)

1. **Push your code to GitHub**
2. **Go to [share.streamlit.io](https://share.streamlit.io)**
3. **Connect GitHub and deploy with these settings:**
   - Repository: `DOMOITA/employeperformanceprediction`
   - Branch: `main`
   - Main file: `employee_performance_app/streamlit_app.py`

### Option 2: Railway (Easy deployment)

1. **Go to [railway.app](https://railway.app)**
2. **Connect your GitHub repository**
3. **Railway auto-detects Streamlit and deploys**

### Option 3: Render

1. **Go to [render.com](https://render.com)**
2. **Create new Web Service from GitHub**
3. **Configure:**
   - Build Command: `pip install -r requirements_streamlit.txt`
   - Start Command: `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`

### Option 4: Docker

1. **Build the image:**
   ```bash
   docker build -t employee-performance-app .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8501:8501 employee-performance-app
   ```

## 🔧 Features

- **🏠 Home:** Welcome page with app overview
- **🔮 Predict:** Interactive employee performance prediction
- **💡 Recommendations:** Performance improvement suggestions
- **ℹ️ About:** Project information and model metrics

## 🛠️ Technical Stack

- **Frontend:** Streamlit with custom CSS
- **Backend:** Python, scikit-learn
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly, Matplotlib

## 📞 Support

For detailed deployment instructions, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

---

**Ready to deploy? Choose your preferred platform from the options above!** 🚀
