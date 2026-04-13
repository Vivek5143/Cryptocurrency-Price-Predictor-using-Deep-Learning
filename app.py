import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import load_model
from hmmlearn.hmm import GaussianHMM
from datetime import datetime

st.set_page_config(
    page_title="Bitcoin Price Prediction",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

TIME_STEP = 60
DEFAULT_HMM_STATES = 4
REGIME_PALETTE = {
    "Bull Trend": "#43AA8B",
    "Bear Trend": "#F94144",
    "High Volatility": "#F8961E",
    "Sideways / Accumulation": "#577590",
}
REGIME_DESCRIPTIONS = {
    "Bull Trend": "Momentum and returns are positive, suggesting trend continuation conditions.",
    "Bear Trend": "Returns are weak and the market is behaving defensively.",
    "High Volatility": "Price swings are elevated and forecasts should be treated with extra caution.",
    "Sideways / Accumulation": "Price action is relatively balanced and the market is consolidating.",
}

# --- CUSTOM CSS FOR ADVANCED LOOK ---
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at top left, rgba(88, 166, 255, 0.08), transparent 32%),
            radial-gradient(circle at bottom right, rgba(67, 170, 139, 0.08), transparent 30%),
            #0E1117;
        color: #C9D1D9;
    }
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    .metric-container {
        background-color: rgba(22, 27, 34, 0.88);
        border: 1px solid #30363D;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.18);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #58A6FF;
    }
    .metric-label {
        font-size: 14px;
        color: #8B949E;
    }
    .insight-card {
        background: linear-gradient(135deg, rgba(88, 166, 255, 0.12), rgba(67, 170, 139, 0.12));
        border: 1px solid rgba(88, 166, 255, 0.24);
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 18px;
    }
    .insight-title {
        font-size: 16px;
        font-weight: 600;
        color: #F0F6FC;
        margin-bottom: 8px;
    }
    .regime-chip {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(88, 166, 255, 0.16);
        border: 1px solid rgba(88, 166, 255, 0.35);
        color: #F0F6FC;
        font-size: 13px;
        margin-top: 8px;
    }
    h1, h2, h3 {
        color: #F0F6FC;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(88, 166, 255, 0.1);
        border-bottom: 2px solid #58A6FF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_keras_model():
    return load_model("crypto_model.keras")


model = load_keras_model()


@st.cache_data(ttl=3600)
def fetch_data():
    ticker = "BTC-USD"
    end = datetime.now()
    start = datetime(end.year - 15, end.month, end.day)
    data = yf.download(ticker, start=start, end=end)
    if data.empty:
        raise ValueError("No data fetched. Check internet connection or Yahoo Finance status.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    data = data.dropna()
    return data


def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[["Close"]])
    return scaler, data_scaled


def predict_future(model, last_data, scaler, future_days=30):
    future_predictions = []
    input_seq = last_data[-TIME_STEP:].reshape(1, TIME_STEP, 1)

    for _ in range(future_days):
        pred = model.predict(input_seq, verbose=0)[0][0]
        future_predictions.append(pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))


def filter_recent_days(frame, days):
    if frame.empty:
        return frame
    cutoff = frame.index.max() - pd.Timedelta(days=days)
    return frame.loc[frame.index >= cutoff]


def label_hidden_states(state_stats):
    ordered_by_return = state_stats["avg_return"].sort_values()
    bear_state = ordered_by_return.index[0]
    bull_state = ordered_by_return.index[-1]

    labels = {
        bear_state: "Bear Trend",
        bull_state: "Bull Trend",
    }

    remaining_states = [state for state in state_stats.index if state not in labels]
    if remaining_states:
        highest_volatility_state = (
            state_stats.loc[remaining_states, "avg_volatility"].sort_values().index[-1]
        )
        labels[highest_volatility_state] = "High Volatility"
        remaining_states = [state for state in remaining_states if state != highest_volatility_state]

    for state in remaining_states:
        labels[state] = "Sideways / Accumulation"

    return labels


@st.cache_data(ttl=3600, show_spinner=False)
def detect_market_regimes(data, n_states=DEFAULT_HMM_STATES):
    if len(data) < 180:
        raise ValueError("Not enough historical data to train the HMM regime detector.")

    regime_frame = data.copy()
    regime_frame["Return"] = regime_frame["Close"].pct_change()
    regime_frame["LogReturn"] = np.log(regime_frame["Close"]).diff()
    regime_frame["Volatility7D"] = regime_frame["LogReturn"].rolling(7).std()
    regime_frame["IntradayRange"] = (
        (regime_frame["High"] - regime_frame["Low"]) / regime_frame["Close"]
    )
    regime_frame["VolumeChange"] = regime_frame["Volume"].replace(0, np.nan).pct_change()
    regime_frame = regime_frame.replace([np.inf, -np.inf], np.nan).dropna().copy()

    feature_columns = ["LogReturn", "Volatility7D", "IntradayRange", "VolumeChange"]
    scaled_features = StandardScaler().fit_transform(regime_frame[feature_columns])

    hmm_model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=400,
        random_state=42,
    )
    hmm_model.fit(scaled_features)
    hidden_states = hmm_model.predict(scaled_features)
    state_probabilities = hmm_model.predict_proba(scaled_features)

    regime_frame["State"] = hidden_states
    regime_frame["RegimeConfidence"] = state_probabilities.max(axis=1)

    state_stats = (
        regime_frame.groupby("State")
        .agg(
            avg_return=("Return", "mean"),
            avg_volatility=("Volatility7D", "mean"),
            avg_range=("IntradayRange", "mean"),
            avg_volume_change=("VolumeChange", "mean"),
            observations=("State", "size"),
            avg_confidence=("RegimeConfidence", "mean"),
        )
        .sort_index()
    )

    regime_labels = label_hidden_states(state_stats)
    state_stats["Regime"] = state_stats.index.map(regime_labels)
    state_stats = state_stats.sort_values("avg_return").reset_index()
    state_stats["Color"] = state_stats["Regime"].map(REGIME_PALETTE).fillna("#58A6FF")

    state_to_regime = state_stats.set_index("State")["Regime"].to_dict()
    state_to_color = state_stats.set_index("State")["Color"].to_dict()
    regime_frame["Regime"] = regime_frame["State"].map(state_to_regime)
    regime_frame["RegimeColor"] = regime_frame["State"].map(state_to_color)

    display_order = state_stats["State"].tolist()
    transition_matrix = hmm_model.transmat_[np.ix_(display_order, display_order)]
    ordered_names = [state_to_regime[state] for state in display_order]
    transition_df = pd.DataFrame(
        transition_matrix,
        index=ordered_names,
        columns=ordered_names,
    )

    latest_probabilities = pd.DataFrame(
        {
            "State": range(n_states),
            "Probability": state_probabilities[-1],
        }
    )
    latest_probabilities["Regime"] = latest_probabilities["State"].map(state_to_regime)
    latest_probabilities["Color"] = latest_probabilities["State"].map(state_to_color)
    latest_probabilities = latest_probabilities.sort_values("Probability", ascending=False)

    state_stats = state_stats.rename(
        columns={
            "avg_return": "Average Return",
            "avg_volatility": "Average Volatility",
            "avg_range": "Average Intraday Range",
            "avg_volume_change": "Average Volume Change",
            "observations": "Observations",
            "avg_confidence": "Average Confidence",
        }
    )

    return regime_frame, state_stats, transition_df, latest_probabilities


def compute_regime_streak(regimes):
    current_regime = regimes.iloc[-1]
    streak = 0
    for regime in reversed(regimes.tolist()):
        if regime != current_regime:
            break
        streak += 1
    return current_regime, streak


def main():
    st.title("Bitcoin Price Prediction & HMM Regime AI")
    st.markdown(
        "LSTM forecasting paired with a Hidden Markov Model to detect hidden Bitcoin market regimes."
    )

    st.sidebar.title("AI Settings")
    future_days = st.sidebar.slider("Future Forecast Days", 1, 60, 30)
    hmm_states = st.sidebar.slider("HMM Regime States", 3, 5, DEFAULT_HMM_STATES)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Historical View Range")
    history_range = st.sidebar.selectbox(
        "Show data for:",
        ["1 Month", "6 Months", "1 Year", "3 Years", "All Time"],
        index=2,
    )

    data = fetch_data()

    if history_range == "1 Month":
        view_data = filter_recent_days(data, 30)
    elif history_range == "6 Months":
        view_data = filter_recent_days(data, 180)
    elif history_range == "1 Year":
        view_data = filter_recent_days(data, 365)
    elif history_range == "3 Years":
        view_data = filter_recent_days(data, 1095)
    else:
        view_data = data

    regime_frame, state_stats, transition_df, latest_probabilities = detect_market_regimes(
        data, hmm_states
    )
    current_regime, regime_streak = compute_regime_streak(regime_frame["Regime"])
    current_regime_row = state_stats.loc[state_stats["Regime"] == current_regime].iloc[0]
    current_regime_confidence = latest_probabilities.iloc[0]["Probability"]
    current_regime_description = REGIME_DESCRIPTIONS.get(
        current_regime, "The HMM is describing the current market state from live BTC behaviour."
    )

    current_price = data["Close"].iloc[-1]
    prev_price = data["Close"].iloc[-2]
    change = current_price - prev_price
    pct_change = (change / prev_price) * 100
    high_24h = data["High"].iloc[-1]
    low_24h = data["Low"].iloc[-1]

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            f'<div class="metric-container"><div class="metric-label">Current BTC Price</div><div class="metric-value">${current_price:,.2f}</div><div style="color: {"#3FB950" if change > 0 else "#F85149"}">{change:+,.2f} ({pct_change:+.2f}%)</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-container"><div class="metric-label">24H High</div><div class="metric-value">${high_24h:,.2f}</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="metric-container"><div class="metric-label">24H Low</div><div class="metric-value">${low_24h:,.2f}</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="metric-container"><div class="metric-label">Current AI Regime</div><div class="metric-value" style="font-size: 18px;">{current_regime}</div><div>{current_regime_confidence:.1%} confidence</div></div>',
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f'<div class="metric-container"><div class="metric-label">Regime Streak</div><div class="metric-value">{regime_streak} days</div><div>HMM state persistence</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">Advanced AI Insight</div>
            <div>{current_regime_description}</div>
            <div class="regime-chip">Detected Regime: {current_regime}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Dashboard & Analysis", "AI Predictions", "HMM Regime AI", "Model Info"]
    )

    with tab1:
        st.subheader(f"Bitcoin Price Action ({history_range})")

        data_full_ma = data.copy()
        data_full_ma["MA50"] = data_full_ma["Close"].rolling(window=50).mean()
        data_full_ma["MA200"] = data_full_ma["Close"].rolling(window=200).mean()

        if history_range == "1 Month":
            view_data_ma = filter_recent_days(data_full_ma, 30)
        elif history_range == "6 Months":
            view_data_ma = filter_recent_days(data_full_ma, 180)
        elif history_range == "1 Year":
            view_data_ma = filter_recent_days(data_full_ma, 365)
        elif history_range == "3 Years":
            view_data_ma = filter_recent_days(data_full_ma, 1095)
        else:
            view_data_ma = data_full_ma

        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=view_data_ma.index,
                open=view_data_ma["Open"],
                high=view_data_ma["High"],
                low=view_data_ma["Low"],
                close=view_data_ma["Close"],
                name="BTC/USD",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=view_data_ma.index,
                y=view_data_ma["MA50"],
                line=dict(color="orange", width=1.5),
                name="50-Day MA",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=view_data_ma.index,
                y=view_data_ma["MA200"],
                line=dict(color="cyan", width=1.5),
                name="200-Day MA",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_rangeslider_visible=False,
            height=600,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Future Predicted Trend")
        st.caption(
            f"LSTM forecast with HMM context: current regime is {current_regime} at {current_regime_confidence:.1%} confidence."
        )

        if st.button("Run AI Prediction"):
            with st.spinner(f"Computing {future_days} days into the future..."):
                try:
                    scaler, data_scaled = preprocess_data(data)
                    future_predictions = predict_future(model, data_scaled, scaler, future_days)

                    future_dates = pd.date_range(
                        start=data.index[-1], periods=future_days + 1, freq="D"
                    )[1:]
                    predictions_df = pd.DataFrame(
                        {"Date": future_dates, "Predicted Close": future_predictions.flatten()}
                    )

                    fig_pred = go.Figure()
                    recent_hist = filter_recent_days(data, 180)
                    fig_pred.add_trace(
                        go.Scatter(
                            x=recent_hist.index,
                            y=recent_hist["Close"],
                            mode="lines",
                            name="Historical Close",
                            line=dict(color="#58A6FF", width=2),
                        )
                    )

                    connect_date = recent_hist.index[-1]
                    connect_price = recent_hist["Close"].iloc[-1]
                    pred_dates = [connect_date] + list(future_dates)
                    pred_prices = np.array([connect_price] + list(future_predictions.flatten()))

                    fig_pred.add_trace(
                        go.Scatter(
                            x=pred_dates,
                            y=pred_prices,
                            mode="lines",
                            name="AI Prediction",
                            line=dict(color="#FF7B72", width=2, dash="dash"),
                        )
                    )

                    regime_volatility = max(float(current_regime_row["Average Volatility"]), 0.005)
                    horizon = np.arange(len(pred_prices))
                    uncertainty_band = pred_prices * regime_volatility * 1.96 * np.sqrt(horizon)
                    uncertainty_band[0] = 0

                    upper_bound = pred_prices + uncertainty_band
                    lower_bound = np.maximum(pred_prices - uncertainty_band, 0)

                    fig_pred.add_trace(
                        go.Scatter(
                            name="Regime-Aware Uncertainty",
                            x=pred_dates + pred_dates[::-1],
                            y=list(upper_bound) + list(lower_bound[::-1]),
                            fill="toself",
                            fillcolor="rgba(255, 123, 114, 0.18)",
                            line=dict(color="rgba(255,255,255,0)"),
                            hoverinfo="skip",
                            showlegend=True,
                        )
                    )

                    fig_pred.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=500,
                        hovermode="x unified",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                        ),
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)

                    c1, c2, c3 = st.columns(3)
                    c1.metric(
                        f"Price on {future_dates[0].strftime('%Y-%m-%d')}",
                        f"${predictions_df.iloc[0]['Predicted Close']:.2f}",
                    )
                    c2.metric(
                        f"Price on {future_dates[-1].strftime('%Y-%m-%d')}",
                        f"${predictions_df.iloc[-1]['Predicted Close']:.2f}",
                    )
                    c3.metric(
                        "Forecast Regime",
                        current_regime,
                        f"{current_regime_confidence:.1%} confidence",
                    )

                    st.caption(
                        "The shaded band scales with the volatility of the HMM-inferred market regime, so uncertainty expands faster during unstable conditions."
                    )
                    st.markdown("### Prediction Data Extract")
                    st.dataframe(predictions_df, use_container_width=True)

                    csv = predictions_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name="btc_predictions.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        else:
            st.info("Click the 'Run AI Prediction' button above to generate future price points.")

    with tab3:
        st.subheader("Hidden Markov Model Regime Detection")
        st.caption(
            "The HMM learns hidden market states from log returns, rolling volatility, intraday range, and volume change."
        )

        if history_range == "1 Month":
            regime_view = filter_recent_days(regime_frame, 30)
        elif history_range == "6 Months":
            regime_view = filter_recent_days(regime_frame, 180)
        elif history_range == "1 Year":
            regime_view = filter_recent_days(regime_frame, 365)
        elif history_range == "3 Years":
            regime_view = filter_recent_days(regime_frame, 1095)
        else:
            regime_view = regime_frame

        fig_regime = go.Figure()
        fig_regime.add_trace(
            go.Scatter(
                x=regime_view.index,
                y=regime_view["Close"],
                mode="lines",
                name="BTC Close",
                line=dict(color="#94A3B8", width=1.3),
                opacity=0.45,
            )
        )

        for _, state_row in state_stats.iterrows():
            regime_name = state_row["Regime"]
            regime_points = regime_view[regime_view["Regime"] == regime_name]
            if regime_points.empty:
                continue
            fig_regime.add_trace(
                go.Scatter(
                    x=regime_points.index,
                    y=regime_points["Close"],
                    mode="markers",
                    name=regime_name,
                    marker=dict(color=state_row["Color"], size=7, opacity=0.9),
                    text=[
                        f"Confidence: {confidence:.1%}"
                        for confidence in regime_points["RegimeConfidence"]
                    ],
                    hovertemplate="%{x}<br>$%{y:,.2f}<br>%{text}<extra>%{fullData.name}</extra>",
                )
            )

        fig_regime.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=520,
            hovermode="closest",
        )
        st.plotly_chart(fig_regime, use_container_width=True)

        col_left, col_right = st.columns(2)
        with col_left:
            prob_chart = go.Figure(
                go.Bar(
                    x=latest_probabilities["Regime"],
                    y=latest_probabilities["Probability"],
                    marker_color=latest_probabilities["Color"],
                )
            )
            prob_chart.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=360,
                title="Current Regime Probabilities",
                yaxis_tickformat=".0%",
            )
            st.plotly_chart(prob_chart, use_container_width=True)

        with col_right:
            heatmap = go.Figure(
                data=go.Heatmap(
                    z=transition_df.values,
                    x=transition_df.columns,
                    y=transition_df.index,
                    colorscale="Viridis",
                    text=np.round(transition_df.values, 2),
                    texttemplate="%{text}",
                )
            )
            heatmap.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=360,
                title="HMM Transition Matrix",
            )
            st.plotly_chart(heatmap, use_container_width=True)

        st.markdown("### Regime Summary Table")
        summary_table = state_stats.copy()
        summary_table["Average Return"] = summary_table["Average Return"].map(lambda x: f"{x:.3%}")
        summary_table["Average Volatility"] = summary_table["Average Volatility"].map(
            lambda x: f"{x:.3%}"
        )
        summary_table["Average Intraday Range"] = summary_table[
            "Average Intraday Range"
        ].map(lambda x: f"{x:.3%}")
        summary_table["Average Volume Change"] = summary_table["Average Volume Change"].map(
            lambda x: f"{x:.3%}"
        )
        summary_table["Average Confidence"] = summary_table["Average Confidence"].map(
            lambda x: f"{x:.1%}"
        )
        st.dataframe(
            summary_table[
                [
                    "State",
                    "Regime",
                    "Average Return",
                    "Average Volatility",
                    "Average Intraday Range",
                    "Average Volume Change",
                    "Average Confidence",
                    "Observations",
                ]
            ],
            use_container_width=True,
        )

    with tab4:
        st.subheader("Hybrid Model Architecture")
        st.markdown(
            f"""
            **Forecasting Model:** Long Short-Term Memory (LSTM)  
            **Regime Detection Model:** Gaussian Hidden Markov Model ({hmm_states} states)  
            **Training Data Window:** 15 Years of Historical BTC-USD Prices  
            **Forecast Feature:** Close Price  
            **HMM Features:** Log Return, 7-Day Volatility, Intraday Range, Volume Change  
            **Time Step:** {TIME_STEP} Days  
            """
        )
        st.markdown(
            """
            **Why this is more advanced now:**  
            The project no longer produces only a point forecast. It also learns latent market states,
            visualizes regime transitions, and adapts forecast uncertainty based on the detected regime.
            """
        )


if __name__ == "__main__":
    main()
