# Project Setup and Run Guide

Follow these commands to easily set up your virtual environment, install dependencies, and run the applications. You can copy and paste these commands directly into your terminal (PowerShell/Command Prompt).

### 1. Create the Virtual Environment
Create a new virtual environment named `venv`:
```bash
python -m venv venv_exam
```

### 2. Activate the Virtual Environment
Activate the environment so that packages are installed locally:
```bash
.\venv_exam\Scripts\activate
```
*(Note: If you are using Git Bash or a Linux/macOS terminal, use `source venv_exam/Scripts/activate` or `source venv_exam/bin/activate` instead)*

### 3. Install/Update Dependencies
Install all the required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Run the Backend App
To start the main application script:
```bash
python app.py
```

### 5. Run the UI App
To start the user interface application:
```bash
python app_ui.py
```
*(Note: If `app_ui.py` is a Streamlit app, you might need to run `streamlit run app_ui.py` instead)*
