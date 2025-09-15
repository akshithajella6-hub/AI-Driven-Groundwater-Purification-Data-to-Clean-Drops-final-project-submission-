This project predicts the purity level of groundwater (Safe / Unsafe) using a machine-learning model built in Python.The backend is a Flask API that receives water-quality measurements 
in JSON format and returns the predicted water status.

Features:
- Pre-trained machine-learning model (Random Forest & Logistic Regression).
- REST API endpoint predict that accepts water-quality parameters and
  responds with `Safe` or `Unsafe`.
- Built entirely with Python 3.12.

To Run the API:
 Open a terminal / PowerShell inside the project folder.
   Start the Flask server:  python app.py
    The server will run locally at: http://127.0.0.1:5000
