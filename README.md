# Bitcoin Price Prediction with LSTM and HMM Regime AI

This project predicts Bitcoin prices with a deep learning **LSTM** model and upgrades the analysis with a **Hidden Markov Model (HMM)** that detects hidden market regimes such as bull, bear, sideways, and high-volatility phases.

The result is a more advanced AI project: instead of showing only a future price line, the app now also explains **what type of market the model believes Bitcoin is in** and adapts forecast uncertainty using the detected regime.

## Project Highlights

- Forecasts Bitcoin closing prices with an LSTM model.
- Detects hidden market states using a Gaussian Hidden Markov Model.
- Uses multi-signal regime features:
  - log returns
  - 7-day rolling volatility
  - intraday range
  - volume change
- Visualizes market regimes directly on the BTC price chart.
- Shows current regime probabilities and the HMM transition matrix.
- Builds a regime-aware uncertainty band around the future forecast.

## Why This Is More Advanced

Traditional price prediction apps usually output a single forecast curve.

This project now combines:

- **Sequence modeling** with LSTM for forecasting
- **Probabilistic state modeling** with HMM for regime detection
- **Explainable market context** through state summaries and transition probabilities

That makes the project feel closer to a real-world financial AI dashboard instead of a basic single-model demo.

## Tech Stack

- Python 3.10+
- TensorFlow / Keras
- hmmlearn
- NumPy / Pandas / scikit-learn
- Plotly
- Streamlit
- yfinance

## App Features

### 1. Dashboard & Analysis
- Candlestick chart
- 50-day and 200-day moving averages
- Live BTC price metrics

### 2. AI Predictions
- Future Bitcoin price forecast using LSTM
- Downloadable forecast table
- Regime-aware uncertainty band scaled by HMM volatility

### 3. HMM Regime AI
- Hidden regime detection across historical BTC data
- Current regime probability view
- Regime transition matrix
- Regime summary table with returns, volatility, and confidence

### 4. Hybrid Model Info
- Shows how LSTM forecasting and HMM regime detection work together

## How the HMM Works

The HMM is trained on engineered BTC market features and learns hidden states without manual labels.

After training, each hidden state is interpreted as one of the following:

- Bull Trend
- Bear Trend
- High Volatility
- Sideways / Accumulation

These inferred states are then used to:

- explain the current market condition
- visualize when the market changed state
- widen or tighten the forecast uncertainty band

## How to Run Locally

1. Clone the repository
   ```bash
   git clone https://github.com/desaiyash21/bitcoin-price-prediction.git
   cd bitcoin-price-prediction
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app
   ```bash
   streamlit run app.py
   ```

## Future Improvements

- Train regime-specific LSTM models
- Add sentiment and macroeconomic features
- Add backtesting for each detected regime
- Compare HMM with GMM and Bayesian forecasting
- Add anomaly detection with a VAE
