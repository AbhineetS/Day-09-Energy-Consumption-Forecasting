**Day 09 — Energy Consumption Forecasting (LSTM)**  

This project explores **short-term energy consumption forecasting** using a **Long Short-Term Memory (LSTM)** neural network.  
The model analyzes time-series electricity usage data and predicts upcoming consumption trends, helping identify daily usage patterns and potential demand spikes.  

---

**Project Overview**  
• Implemented an LSTM-based forecasting model for time-series energy data  
• Preprocessed raw data and created supervised learning sequences (24-hour lookback)  
• Trained the model with dropout regularization for generalization  
• Evaluated using **Root Mean Squared Error (RMSE)** and visualized predictions  

---

**Model Architecture**  
• Input: 24-hour historical consumption window  
• Hidden Layers: LSTM(64) → Dense(32) → Dense(1)  
• Optimizer: Adam  
• Loss: Mean Squared Error  
• Test RMSE: **0.2029**  

---

**Tech Stack**  
TensorFlow / Keras | Pandas | NumPy | Matplotlib | Scikit-learn  

---

**Output Files**  
| File | Description |  
|------|--------------|  
| `energy_lstm.h5` | Trained model file |  
| `training_loss.png` | Training loss visualization |  
| `forecast_plot.png` | Forecasted vs actual values |  

---

**Run the Project**  
```
source ../Day-01-Titanic/venv/bin/activate
pip install tensorflow pandas scikit-learn matplotlib
python3 run_energy_forecast.py
```

---

**Dataset Source**  
The dataset (`energy.csv`) is not included due to GitHub size limits.  
You can download it here →  
[Individual Household Electric Power Consumption Dataset](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)

---

**Result**  
The LSTM model efficiently predicts short-term energy usage patterns with low error, demonstrating how deep learning can optimize resource forecasting and energy management systems.  

---

**Part of my 64-Day AI/ML Challenge — building one real-world project every day to strengthen applied machine learning skills.**  