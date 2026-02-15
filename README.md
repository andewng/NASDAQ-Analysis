# NASDAQ-Analysis Project

An interactive stock analysis and valuation project combining financial simulation and machine learning models to estimate stock valuation over time.

Repository Contents:

- ```NASDAQML.py``` A Streamlit-based Stock Analysis & Price Simulation App  
- ```valuation.ipynd``` A Machine Learning Valuation Notebook (Work in Progress)

The long-term objective is to integrate both components into a unified valuation engine that allows switching between different valuation methodologies and generating automated insights and reports.

---

##  Structure
├── NASDAQML.py # Streamlit Monte Carlo simulation app
├── valuation.ipynb # LSTM & ML valuation modeling (WIP)
└── README.md

**File:** `NASDAQML.py`
This Streamlit application performs stochastic stock price simulation using Monte Carlo simulation.

### How to Run
pip install -r requirements.txt
streamlit run NASDAQML.py

**File:** `valuation.ipynd`
This notebook explores data analysis & machine learning to predict stock valuation over time.
***This component is still under active development**

- Python
- Streamlit
- NumPy
- Pandas
- Matplotlib / Seaborn
- TensorFlow / Keras 
- scikit-learn
- yfinance

This project is for educational purposes only & does not constitute financial advice.
