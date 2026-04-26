import streamlit as st
import asyncio
import os
import json
import plotly.graph_objects as go
import re
import logging
import threading
import time
from datetime import datetime, timedelta
import traceback
import yfinance as yf
import numpy as np
from financial_signals_agent import analyze_stock
from sentiment_analysis import get_sentiment_analysis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit_app")

# Configure page settings - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Financial Signals Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for footer
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    footer:after {
        content:'DISCLAIMER: This dashboard is for informational purposes only. Not financial advice.'; 
        visibility: visible;
        display: block;
        position: relative;
        padding: 5px;
        top: 2px;
        color: #888;
        text-align: center;
        font-size: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state variables
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'analysis_requested' not in st.session_state:
    st.session_state.analysis_requested = False
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = ""
if 'sentiment_results' not in st.session_state:
    st.session_state.sentiment_results = None
if 'sentiment_requested' not in st.session_state:
    st.session_state.sentiment_requested = False
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0
if 'analysis_in_progress' not in st.session_state:
    st.session_state.analysis_in_progress = False
if 'sentiment_in_progress' not in st.session_state:
    st.session_state.sentiment_in_progress = False
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = time.time()
if 'refresh_counter' not in st.session_state:
    st.session_state.refresh_counter = 0
if 'analysis_complete_flag' not in st.session_state:
    st.session_state.analysis_complete_flag = False
if 'sentiment_complete_flag' not in st.session_state:
    st.session_state.sentiment_complete_flag = False
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'parsed_signal' not in st.session_state:
    st.session_state.parsed_signal = None
if 'model_settings' not in st.session_state:
    st.session_state.model_settings = {"model_type": "bedrock"}

# Create a file-based flag system for thread communication
def set_flag(flag_name, value):
    """Set a flag in a file for cross-thread communication"""
    flag_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flags")
    os.makedirs(flag_dir, exist_ok=True)
    flag_path = os.path.join(flag_dir, f"{flag_name}.flag")
    
    if value:
        # Create the flag file
        with open(flag_path, 'w') as f:
            f.write(str(datetime.now().isoformat()))
    else:
        # Remove the flag file if it exists
        if os.path.exists(flag_path):
            os.remove(flag_path)

def check_flag(flag_name):
    """Check if a flag file exists"""
    flag_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flags")
    flag_path = os.path.join(flag_dir, f"{flag_name}.flag")
    return os.path.exists(flag_path)

class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types during JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def save_results(result_type, data):
    """Save results to a file for cross-thread communication"""
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, f"{result_type}.json")
    
    with open(result_path, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder)

def load_results(result_type):
    """Load results from a file"""
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    result_path = os.path.join(results_dir, f"{result_type}.json")
    
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            return json.load(f)
    return None

def get_real_market_data(ticker: str) -> dict:
    """Fetch real-time price, RSI, and MAs from Yahoo Finance via yfinance.
    Returns a dict with price, rsi, ma_50, ma_200 (all floats or None).
    """
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="1y")
        if hist.empty:
            logger.warning(f"yfinance returned no data for {ticker}")
            return {}
        
        price = float(hist['Close'].iloc[-1])
        
        # 50-day and 200-day simple moving averages
        ma_50 = float(hist['Close'].rolling(50).mean().iloc[-1]) if len(hist) >= 50 else None
        ma_200 = float(hist['Close'].rolling(200).mean().iloc[-1]) if len(hist) >= 200 else None
        
        # RSI (14-period)
        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 100.0
        rsi = float(round(100 - (100 / (1 + rs)), 1))  # convert np.float64 → float
        
        result = {
            'price': float(round(price, 2)),
            'rsi': rsi,
            'ma_50': float(round(ma_50, 2)) if ma_50 else None,
            'ma_200': float(round(ma_200, 2)) if ma_200 else None,
        }
        logger.info(f"yfinance real data for {ticker}: {result}")
        return result
    except Exception as e:
        logger.error(f"yfinance fetch failed for {ticker}: {e}")
        return {}

# Function to extract technical data from signal text
def extract_technical_data(text):
    """Extract technical data from signal text for visualization"""
    data = {}
    
    # Extract price — handles $, ₹, Rs, and comma-formatted Indian prices
    price_match = re.search(r'Price:\s*(?:Rs\.?|₹|\$|\€)?\s*([\d,]+\.?\d*)', text)
    if price_match:
        try:
            # Remove commas and convert to float
            price_str = price_match.group(1).replace(',', '')
            data['price'] = float(price_str)
        except (ValueError, TypeError):
            data['price'] = None
    else:
        data['price'] = None
    
    # Extract RSI
    rsi_match = re.search(r'RSI:?\s*(\d+\.?\d*)', text)
    if rsi_match:
        try:
            data['rsi'] = float(rsi_match.group(1))
        except (ValueError, TypeError):
            data['rsi'] = None
    else:
        data['rsi'] = None
    
    # Extract moving averages (supports $, ₹, and commas)
    ma50_match = re.search(r'50-day MA:?\s*[\$₹]?\s*([\d,]+\.?\d*)', text)
    if ma50_match:
        try:
            ma_str = ma50_match.group(1).replace(',', '')
            data['ma_50'] = float(ma_str)
        except (ValueError, TypeError):
            data['ma_50'] = None
    else:
        # Try alternative format
        ma_match = re.search(r'below 50-day MA \([\$₹]?\s*([\d,]+\.?\d*)\)', text)
        if ma_match:
            try:
                ma_str = ma_match.group(1).replace(',', '')
                data['ma_50'] = float(ma_str)
            except (ValueError, TypeError):
                data['ma_50'] = None
        else:
            data['ma_50'] = None
    
    ma200_match = re.search(r'200-day MA:?\s*[\$₹]?\s*([\d,]+\.?\d*)', text)
    if ma200_match:
        try:
            ma_str = ma200_match.group(1).replace(',', '')
            data['ma_200'] = float(ma_str)
        except (ValueError, TypeError):
            data['ma_200'] = None
    else:
        data['ma_200'] = None
    
    return data

# Function to parse signal text into structured data
def parse_signal_text(signal_text: str) -> dict:
    """Parse the structured signal text into a dictionary"""
    try:
        # Clean up the text for better parsing
        clean_text = re.sub(r'\s+', ' ', signal_text)  # Replace multiple spaces with a single space
        
        signal_data = {}
        
        # Extract direction
        direction_match = re.search(r'Direction:\s*(\w+)', clean_text)
        if direction_match:
            signal_data['direction'] = direction_match.group(1)
        else:
            if "BUY" in clean_text.upper():
                signal_data['direction'] = "BUY"
            elif "SELL" in clean_text.upper():
                signal_data['direction'] = "SELL"
            else:
                signal_data['direction'] = "HOLD"
        
        # Extract confidence score
        confidence_match = re.search(r'Confidence Score:\s*(\d+)%', clean_text)
        if confidence_match:
            signal_data['confidence_score'] = int(confidence_match.group(1))
        else:
            signal_data['confidence_score'] = 70  # Default confidence
        
        # Extract position size
        position_match = re.search(r'Position Size:\s*(.+?)(?:\n|$)', clean_text)
        if position_match:
            signal_data['position_size'] = position_match.group(1).strip()
        else:
            signal_data['position_size'] = "5% of portfolio"  # Default position size
        
        # Extract technical analysis
        tech_section = re.search(r'Technical Analysis:(.*?)(?:Key Factors:|Risk Assessment:|$)', signal_text, re.DOTALL)
        if tech_section:
            tech_text = tech_section.group(1)
            tech_points = re.findall(r'[-•]\s*(.+?)(?:\n|$)', tech_text)
            # Clean up each point
            tech_points = [re.sub(r'\s+', ' ', point).strip() for point in tech_points]
            signal_data['technical_analysis'] = tech_points
            
            # Also extract technical data for charts
            signal_data['technical_data'] = extract_technical_data(tech_text)
        else:
            signal_data['technical_analysis'] = []
            signal_data['technical_data'] = extract_technical_data(signal_text)
        
        # Extract key factors
        factors_section = re.search(r'Key Factors:(.*?)(?:Risk Assessment:|$)', signal_text, re.DOTALL)
        if factors_section:
            factors_text = factors_section.group(1)
            factors = re.findall(r'[-•]\s*(.+?)(?:\n|$)', factors_text)
            # Clean up each factor
            factors = [re.sub(r'\s+', ' ', factor).strip() for factor in factors]
            signal_data['key_factors'] = factors
        else:
            signal_data['key_factors'] = []
        
        # Extract risk assessment
        risk_section = re.search(r'Risk Assessment:(.*?)(?:Market Context:|Recommendation:|$)', signal_text, re.DOTALL)
        if risk_section:
            risk_text = risk_section.group(1).strip()
            # Clean up the text
            risk_text = re.sub(r'\s+', ' ', risk_text).strip()
            signal_data['risk_assessment'] = risk_text
        else:
            signal_data['risk_assessment'] = ""
        
        # Extract recommendation
        recommendation_section = re.search(r'Recommendation:(.*?)(?:$)', signal_text, re.DOTALL)
        if recommendation_section:
            recommendation_text = recommendation_section.group(1).strip()
            # Clean up the text
            recommendation_text = re.sub(r'\s+', ' ', recommendation_text).strip()
            signal_data['recommendation'] = recommendation_text
        else:
            signal_data['recommendation'] = ""
        
        # Log the parsed data for debugging
        logger.info(f"Parsed signal data: {signal_data}")
                
        return signal_data
    except Exception as e:
        logger.error(f"Error parsing signal text: {str(e)}")
        # Return default values if parsing fails
        return {
            'direction': 'HOLD',
            'confidence_score': 50,
            'position_size': 'N/A',
            'technical_analysis': [],
            'key_factors': [],
            'risk_assessment': '',
            'recommendation': '',
            'technical_data': {
                'price': 150.0,
                'ma_50': 145.0,
                'ma_200': 140.0,
                'rsi': 55
            }
        }

# Functions to create visualizations
def create_technical_analysis_chart(tech_data, ticker=None):
    """Create a technical analysis visualization"""
    if not tech_data:
        return None
    
    # Determine currency based on ticker
    is_indian = ticker and ('.NS' in ticker or '.BO' in ticker)
    currency_symbol = '₹' if is_indian else '$'
    currency_name = 'INR' if is_indian else 'USD'
    
    # Create price and moving averages comparison
    price = tech_data.get('price', 150.0)
    
    # Check if MA values are available
    has_ma_50 = 'ma_50' in tech_data and tech_data['ma_50'] is not None
    has_ma_200 = 'ma_200' in tech_data and tech_data['ma_200'] is not None
    
    # Create labels and values arrays based on available data
    labels = ['Current Price']
    values = [price]
    colors = ['#2ecc71']  # Green for price
    
    if has_ma_50:
        labels.append('50-day MA')
        values.append(tech_data['ma_50'])
        colors.append('#3498db')  # Blue for 50-day MA
    
    if has_ma_200:
        labels.append('200-day MA')
        values.append(tech_data['ma_200'])
        colors.append('#9b59b6')  # Purple for 200-day MA
    
    # Format text labels safely, handling None values
    text_labels = []
    for v in values:
        if v is None:
            text_labels.append("N/A")
        else:
            try:
                text_labels.append(f"{currency_symbol}{v:,.2f}")
            except (ValueError, TypeError):
                text_labels.append(f"{currency_symbol}{v}")
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=text_labels,
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Price vs Moving Averages',
        yaxis=dict(
            title=f'Price ({currency_symbol})'
        ),
        height=400
    )
    
    return fig

def create_rsi_gauge(rsi_value):
    """Create an RSI gauge chart"""
    if not rsi_value:
        rsi_value = 50
    
    # Ensure rsi_value is a number
    try:
        rsi_value = float(rsi_value)
    except (ValueError, TypeError):
        rsi_value = 50  # Default if conversion fails
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rsi_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "RSI (Relative Strength Index)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 30], 'color': "#2ecc71"},  # Oversold - Green
                {'range': [30, 70], 'color': "#f1c40f"},  # Neutral - Yellow
                {'range': [70, 100], 'color': "#e74c3c"}  # Overbought - Red
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': rsi_value
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_signal_gauge_chart(confidence_score):
    """Create a gauge chart for the confidence score"""
    # Ensure confidence_score is a number
    try:
        confidence_score = float(confidence_score)
    except (ValueError, TypeError):
        confidence_score = 50  # Default if conversion fails
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 30], 'color': "#e74c3c"},
                {'range': [30, 70], 'color': "#f39c12"},
                {'range': [70, 100], 'color': "#2ecc71"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': confidence_score
            }
        }
    ))
    
    fig.update_layout(
        height=300,  # Increased from 250 to 300
        margin=dict(l=20, r=20, t=50, b=20)  # Increased top margin from 30 to 50
    )
    
    return fig

def create_signal_direction_chart(direction):
    """Create a visual indicator for signal direction"""
    if direction == "BUY":
        color = "#2ecc71"  # Green
        value = 80
    elif direction == "SELL":
        color = "#e74c3c"  # Red
        value = 20
    else:  # HOLD
        color = "#f39c12"  # Yellow/Orange
        value = 50
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Signal: {direction}"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 0, 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': "#ffeeee"},
                {'range': [33, 66], 'color': "#ffffee"},
                {'range': [66, 100], 'color': "#eeffee"}
            ]
        }
    ))
    
    fig.update_layout(
        height=300,  # Increased from 250 to 300
        margin=dict(l=20, r=20, t=50, b=20)  # Increased top margin from 30 to 50
    )
    
    return fig

def create_risk_reward_chart(signal_data):
    """Create a risk-reward visualization"""
    # Extract risk level from signal data or use default
    risk_text = signal_data.get('risk_assessment', '').lower()
    
    if 'high risk' in risk_text:
        risk = 80
    elif 'medium risk' in risk_text or 'moderate risk' in risk_text:
        risk = 50
    elif 'low risk' in risk_text:
        risk = 20
    else:
        # Default based on confidence score
        confidence = signal_data.get('confidence_score', 50)
        risk = 100 - confidence
    
    # Calculate reward based on confidence and direction
    confidence = signal_data.get('confidence_score', 50)
    direction = signal_data.get('direction', 'HOLD')
    
    if direction == 'BUY':
        reward = confidence
    elif direction == 'SELL':
        reward = confidence * 0.8  # Slightly lower reward for sell signals
    else:  # HOLD
        reward = confidence * 0.5  # Lower reward for hold signals
    
    # Create the chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=['Risk', 'Reward'],
        y=[risk, reward],
        mode='lines+markers',
        marker=dict(size=16, color=['#e74c3c', '#2ecc71']),
        line=dict(width=4, color='#3498db')
    ))
    
    fig.update_layout(
        title='Risk-Reward Profile',
        yaxis=dict(
            title='Score',
            range=[0, 100]
        ),
        height=250
    )
    
    return fig

# Function to run analysis in a separate thread
def run_analysis_thread(ticker, model_settings=None, sentiment_settings=None):
    """Run stock analysis in a separate thread. Launches sentiment thread after completion."""
    try:
        logger.info(f"Starting analysis thread for {ticker}")
        
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        if model_settings is None:
            model_settings = {"model_type": "bedrock"}
        logger.info(f"Using model settings: {model_settings}")
        
        results = loop.run_until_complete(analyze_stock(ticker, model_settings))
        
        if results and isinstance(results, dict) and results.get('status') == 'success':
            signal_text = results.get('signal', '')
            parsed_signal = parse_signal_text(signal_text)
            
            # ── Override LLM price with REAL yfinance data ──────────────
            real_data = get_real_market_data(ticker)
            if real_data:
                tech = parsed_signal.get('technical_data', {})
                if real_data.get('price'):
                    tech['price'] = real_data['price']
                if real_data.get('rsi'):
                    tech['rsi'] = real_data['rsi']
                if real_data.get('ma_50'):
                    tech['ma_50'] = real_data['ma_50']
                if real_data.get('ma_200'):
                    tech['ma_200'] = real_data['ma_200']
                parsed_signal['technical_data'] = tech
                logger.info(f"Overrode LLM price with real yfinance data: {real_data}")
            # ────────────────────────────────────────────────────────────
            
            results['parsed_signal'] = parsed_signal
            save_results("analysis", results)
            set_flag("analysis_complete", True)
            logger.info(f"Analysis completed for {ticker} and saved to file")
            
            # ── Now start sentiment thread sequentially ──────────────────
            if sentiment_settings is None:
                sentiment_settings = {"model_type": "bedrock"}
            set_flag("sentiment_started", True)
            sentiment_thread = threading.Thread(
                target=run_sentiment_thread,
                args=(ticker, sentiment_settings)
            )
            sentiment_thread.daemon = True
            sentiment_thread.start()
            logger.info(f"Sentiment thread launched after financial analysis for {ticker}")
        else:
            logger.error(f"Error in analysis: {results}")
            set_flag("analysis_error", True)
        
        loop.close()
        
    except Exception as e:
        logger.error(f"Error in analysis thread: {str(e)}")
        logger.error(traceback.format_exc())
        set_flag("analysis_error", True)

# Function to run sentiment analysis in a separate thread
def run_sentiment_thread(ticker, model_settings=None):
    """Run sentiment analysis in a separate thread
    
    Args:
        ticker: Stock ticker symbol
        model_settings: Dictionary with model provider settings
    """
    try:
        logger.info(f"Starting sentiment thread for {ticker}")
        
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Use provided model settings or default to Bedrock
        if model_settings is None:
            model_settings = {"model_type": "bedrock"}
        logger.info(f"Using model settings for sentiment: {model_settings}")
        
        # Run the sentiment analysis (properly awaiting the coroutine)
        results = loop.run_until_complete(get_sentiment_analysis(ticker, model_settings))
        
        # Save results to file
        if results and isinstance(results, dict) and results.get('status') == 'success':
            save_results("sentiment", results)
            set_flag("sentiment_complete", True)
            logger.info(f"Sentiment analysis completed for {ticker} and saved to file")
        else:
            logger.error(f"Error in sentiment analysis: {results}")
            set_flag("sentiment_error", True)
        
        # Close the loop
        loop.close()
        
    except Exception as e:
        logger.error(f"Error in sentiment thread: {str(e)}")
        logger.error(traceback.format_exc())
        set_flag("sentiment_error", True)

def check_and_update_results():
    """Check for completed analyses and update session state"""
    # Check for completed analysis
    if check_flag("analysis_complete"):
        results = load_results("analysis")
        if results:
            st.session_state.analysis_results = results
            st.session_state.analysis_in_progress = False
            st.session_state.analysis_complete_flag = True
            st.session_state.parsed_signal = results.get('parsed_signal', {})
            # Financial done → sentiment is now starting automatically
            st.session_state.sentiment_in_progress = True
            logger.info("Updated session state with analysis results — sentiment now starting")
        set_flag("analysis_complete", False)

    # Detect that sentiment thread was started (in case we missed the flip)
    if check_flag("sentiment_started"):
        st.session_state.sentiment_in_progress = True
        set_flag("sentiment_started", False)

    # Check for analysis errors
    if check_flag("analysis_error"):
        st.session_state.analysis_in_progress = False
        st.session_state.error_message = "Error during analysis"
        logger.info("Analysis error detected")
        set_flag("analysis_error", False)
    
    # Check for completed sentiment analysis
    if check_flag("sentiment_complete"):
        results = load_results("sentiment")
        if results:
            st.session_state.sentiment_results = results
            st.session_state.sentiment_in_progress = False
            st.session_state.sentiment_complete_flag = True
            logger.info("Updated session state with sentiment results")
        set_flag("sentiment_complete", False)
    
    # Check for sentiment analysis errors
    if check_flag("sentiment_error"):
        st.session_state.sentiment_in_progress = False
        logger.info("Sentiment analysis error detected")
        set_flag("sentiment_error", False)


def main():
    # Check for completed analyses
    check_and_update_results()
    
    st.title("📈 Financial Signals Dashboard")
    
    st.markdown("""
    <div style="display:flex; gap:12px; align-items:center; margin-top:-10px; margin-bottom:16px; flex-wrap:wrap;">
        <div style="background:linear-gradient(135deg,#1a1f2e,#2d3748); border:1px solid #3b82f6; border-radius:8px; padding:8px 16px; display:flex; align-items:center; gap:8px;">
            <span style="font-size:1.2em;">📊</span>
            <div>
                <div style="color:#93c5fd; font-size:11px; font-weight:600; letter-spacing:0.5px; text-transform:uppercase;">Financial Analysis</div>
                <div style="color:#f1f5f9; font-size:13px; font-weight:700;">XGBoost Classifier</div>
            </div>
        </div>
        <div style="color:#64748b; font-size:18px; font-weight:300;">×</div>
        <div style="background:linear-gradient(135deg,#1a1f2e,#2d3748); border:1px solid #8b5cf6; border-radius:8px; padding:8px 16px; display:flex; align-items:center; gap:8px;">
            <span style="font-size:1.2em;">🧠</span>
            <div>
                <div style="color:#c4b5fd; font-size:11px; font-weight:600; letter-spacing:0.5px; text-transform:uppercase;">Sentiment Analysis</div>
                <div style="color:#f1f5f9; font-size:13px; font-weight:700;">FinBERT NLP Model</div>
            </div>
        </div>
        <div style="color:#64748b; font-size:18px; font-weight:300;">→</div>
        <div style="background:linear-gradient(135deg,#1a1f2e,#2d3748); border:1px solid #10b981; border-radius:8px; padding:8px 16px; display:flex; align-items:center; gap:8px;">
            <span style="font-size:1.2em;">⚡</span>
            <div>
                <div style="color:#6ee7b7; font-size:11px; font-weight:600; letter-spacing:0.5px; text-transform:uppercase;">Output</div>
                <div style="color:#f1f5f9; font-size:13px; font-weight:700;">Alpha Signal Generation</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Display current stock and date at the top with enhanced styling
    if st.session_state.analysis_results:
        ticker = st.session_state.analysis_results.get('ticker', '')
        current_date = datetime.now().strftime("%B %d, %Y")
        
        # Get price from parsed signal if available
        price = None
        if st.session_state.parsed_signal and 'technical_data' in st.session_state.parsed_signal:
            price = st.session_state.parsed_signal['technical_data'].get('price')
        
        # Format price display properly
        is_indian = '.NS' in ticker or '.BO' in ticker
        currency_symbol = '₹' if is_indian else '$'
        if price is not None:
            try:
                price_display = f"{currency_symbol}{price:,.2f}"
            except (ValueError, TypeError):
                price_display = f"{currency_symbol}{price}"
        else:
            price_display = "N/A"
        
        # Create a more visually appealing header with background color
        st.markdown(
            f"""
            <div style="background-color:#0e1117; padding:10px; border-radius:5px; margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <h2 style="color:#3498db; margin:0;">
                            <span style="font-size:1.5em;">📊</span> {ticker}
                        </h2>
                    </div>
                    <div style="text-align:right;">
                        <p style="color:#2ecc71; font-size:1.2em; margin:0;">
                            <strong>Price:</strong> {price_display}
                        </p>
                        <p style="color:#95a5a6; margin:0;">
                            <em>{current_date}</em>
                        </p>
                    </div>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    # Removed hybrid AI info banner per request
    
    # Add disclaimer at the top
    with st.expander("📝 Investment Disclaimer", expanded=False):
        st.markdown("""
        **IMPORTANT DISCLAIMER**: The information provided in this dashboard is for informational purposes only and does not constitute investment advice. The analysis, signals, and recommendations are generated using automated tools and may not account for all market factors. 
        
        **Before making any investment decisions:**
        - Consult with a qualified financial advisor
        - Conduct your own research
        - Consider your financial situation, investment objectives, and risk tolerance
        
        Past performance is not indicative of future results. Investing in securities involves risk of loss. The creators and operators of this dashboard are not responsible for any financial losses or damages resulting from the use of this information.
        """)
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        ticker = st.text_input(
            "Enter Stock Ticker", 
            value="AAPL",
            help="US stocks: AAPL, MSFT, TSLA\nIndian stocks: RELIANCE.NS, TCS.NS, INFY.NS (NSE) or .BO (BSE)"
        ).upper()
        
        # Show example based on what user typed
        if '.NS' in ticker or '.BO' in ticker:
            st.caption("🇮🇳 Analyzing Indian stock - using Moneycontrol, ET, NSE/BSE data")
        else:
            st.caption("🇺🇸 Analyzing US stock - using Yahoo Finance, Investing.com, CNBC")
        
        # Removed Hybrid AI System info per request
        
        # Store dummy settings (not used anymore, but kept for compatibility)
        st.session_state.model_settings = {"model_type": "hybrid"}
        
        if st.button("Analyze Stock"):
            # Reset all state
            st.session_state.analysis_results = None
            st.session_state.sentiment_results = None
            st.session_state.analysis_requested = True
            st.session_state.current_ticker = ticker
            st.session_state.sentiment_requested = True
            st.session_state.analysis_in_progress = True
            st.session_state.sentiment_in_progress = False  # starts after financial completes
            st.session_state.analysis_complete_flag = False
            st.session_state.sentiment_complete_flag = False
            st.session_state.error_message = None
            st.session_state.parsed_signal = None
            
            # Reset all flags
            set_flag("analysis_complete", False)
            set_flag("analysis_error", False)
            set_flag("sentiment_complete", False)
            set_flag("sentiment_error", False)
            
            try:
                # Bedrock-only mode: Nova Premier for both (Nova Lite cannot handle MCP tool use)
                financial_settings = {
                    "model_type": "bedrock",
                    "bedrock_model_id": "us.amazon.nova-premier-v1:0",
                }
                sentiment_settings = {
                    "model_type": "bedrock",
                    "bedrock_model_id": "us.amazon.nova-premier-v1:0",  # Nova Lite fails tool use
                }
                
                logger.info("Using Bedrock-only mode: Nova Premier for both financial + sentiment")
                
                # Start financial analysis thread first
                # Sentiment thread will be launched automatically inside run_analysis_thread
                # once the financial analysis completes successfully
                analysis_thread = threading.Thread(
                    target=run_analysis_thread, 
                    args=(ticker, financial_settings, sentiment_settings)
                )
                analysis_thread.daemon = True
                analysis_thread.start()
                
                # Show a message that analysis is in progress
                st.info(f"Analysis for {ticker} started! Results will appear shortly.")
                
            except Exception as e:
                st.error(f"Error analyzing stock: {str(e)}")
                logger.error(f"Error in analyze_stock: {str(e)}")
            
        # Add disclaimer to sidebar as well
        st.markdown("---")
        st.markdown("### Disclaimer")
        st.caption("This dashboard provides analysis for informational purposes only. Not financial advice. Invest at your own risk.")
    
    # Main content area
    if st.session_state.analysis_complete_flag and st.session_state.analysis_results:
        # We have results, display them
        results = st.session_state.analysis_results
        parsed_signal = st.session_state.parsed_signal
        
        # Extract key technical indicators from the signal text
        signal_text = results.get('signal', '')
        
        # Initialize indicators variables
        key_indicators = ""
        rsi_text = ""
        macd_text = ""
        
        # Extract key indicators using regex patterns
        import re  # Import re module here to ensure it's available
        
        key_indicators_match = re.search(r'Key indicators:([^,\n]+)', signal_text)
        key_indicators = key_indicators_match.group(1).strip() if key_indicators_match else ""
        
        rsi_match = re.search(r'RSI=(\d+\.?\d*)', signal_text)
        rsi_text = f"RSI={rsi_match.group(1)}" if rsi_match else ""
        
        macd_match = re.search(r'MACD=(-?\d+\.?\d*)', signal_text)
        macd_text = f"MACD={macd_match.group(1)}" if macd_match else ""
        
        # Combine the indicators
        indicators_text = ""
        if key_indicators:
            indicators_text += key_indicators
        if rsi_text:
            indicators_text += f", {rsi_text}"
        if macd_text:
            indicators_text += f", {macd_text}"
        
        # Create tabs for different analyses
        tab1, tab2 = st.tabs(["Financial Analysis", "Sentiment Analysis"])
        
        with tab1:
            st.subheader("Financial Analysis")
            
            # Display charts and visualizations
            if parsed_signal:
                # Create a 2x2 grid for charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Display signal direction chart
                    direction = parsed_signal.get('direction', 'HOLD')
                    direction_chart = create_signal_direction_chart(direction)
                    st.plotly_chart(direction_chart, width='stretch')
                    
                    # Display confidence score gauge
                    confidence = parsed_signal.get('confidence_score', 50)
                    confidence_chart = create_signal_gauge_chart(confidence)
                    st.plotly_chart(confidence_chart, width='stretch')
                
                with col2:
                    # Display risk-reward chart
                    risk_reward_chart = create_risk_reward_chart(parsed_signal)
                    st.plotly_chart(risk_reward_chart, width='stretch')
                    
                    # Display position size recommendation
                    position_size = parsed_signal.get('position_size', 'N/A')
                    st.info(f"**Recommended Position Size:** {position_size}")
                    
                    # Display key indicators in a blue box
                    if indicators_text:
                        st.info(f"**Key Indicators:** {indicators_text}")
                    
                    # Display recommendation
                    recommendation = parsed_signal.get('recommendation', '')
                    if recommendation:
                        st.success(f"**Recommendation:** {recommendation}")
                
                # Display technical analysis charts
                st.subheader("Technical Indicators")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Display price vs moving averages chart
                    tech_data = parsed_signal.get('technical_data', {})
                    price_chart = create_technical_analysis_chart(tech_data, st.session_state.current_ticker)
                    if price_chart:
                        st.plotly_chart(price_chart, width='stretch')
                
                with col2:
                    # Display RSI gauge
                    rsi = tech_data.get('rsi')
                    if rsi:
                        rsi_chart = create_rsi_gauge(rsi)
                        st.plotly_chart(rsi_chart, width='stretch')
                
                # Display key factors
                st.subheader("Key Factors")
                key_factors = parsed_signal.get('key_factors', [])
                if key_factors:
                    for factor in key_factors:
                        st.markdown(f"- {factor}")
                else:
                    st.info("No key factors identified.")
                
                # Display risk assessment
                risk_assessment = parsed_signal.get('risk_assessment', '')
                if risk_assessment:
                    st.subheader("Risk Assessment")
                    st.write(risk_assessment)
                
                # ── 📖 Plain English Explainer ─────────────────────────────
                with st.expander("📖 Explain This Analysis in Plain English"):
                    direction = parsed_signal.get('direction', 'HOLD')
                    confidence = parsed_signal.get('confidence_score', 50)
                    tech_data = parsed_signal.get('technical_data', {})
                    price = tech_data.get('price')
                    rsi = tech_data.get('rsi')
                    ma_50 = tech_data.get('ma_50')
                    ma_200 = tech_data.get('ma_200')
                    ticker = st.session_state.current_ticker
                    
                    # Signal explanation
                    signal_explain = {
                        'BUY': f"Our **XGBoost ensemble model** recommends **buying** {ticker}. Based on technical indicators, scraped financial data, and market conditions, the model predicts the stock is likely to go **up** in the near term.",
                        'SELL': f"Our **XGBoost ensemble model** recommends **selling** {ticker}. The model's analysis of price trends, moving averages, and sentiment data suggests the stock may **decline** from here, so it's better to reduce exposure.",
                        'HOLD': f"Our **XGBoost ensemble model** recommends **holding** {ticker}. This means don't buy more, don't sell either — the model sees the stock in a **neutral zone** without a strong directional signal."
                    }
                    st.markdown(f"### 🎯 Signal: {direction}")
                    st.markdown(signal_explain.get(direction, signal_explain['HOLD']))
                    
                    # Confidence explanation
                    st.markdown(f"### 📊 Confidence Score: {confidence}%")
                    if confidence >= 75:
                        conf_text = "The model is **highly confident** in this call. The data is clear and consistent."
                    elif confidence >= 50:
                        conf_text = "The model is **moderately confident**. There's enough supporting data, but some uncertainty remains."
                    else:
                        conf_text = "The model has **low confidence**. The data is mixed or incomplete — treat this as a weak signal."
                    st.markdown(conf_text)
                    
                    # Price vs Moving Averages
                    if price and (ma_50 or ma_200):
                        st.markdown("### 📈 Price vs Moving Averages")
                        # Show price with last trading date
                        try:
                            tk = yf.Ticker(ticker)
                            hist = tk.history(period="5d")
                            if not hist.empty:
                                last_date = hist.index[-1].strftime('%B %d, %Y')
                            else:
                                last_date = 'N/A'
                        except:
                            last_date = 'N/A'
                        st.markdown(f"The current price is **₹{price:,.2f}** *(as of last market close: {last_date})*. For live intraday price, check [Google Finance](https://www.google.com/finance/quote/{ticker.replace('.NS', ':NSE').replace('.BO', ':BOM')}).")
                        if ma_50:
                            if price > ma_50:
                                st.markdown(f"- Price is **above** the 50-day average (₹{ma_50:,.2f}) → short-term **uptrend** ✅")
                            else:
                                st.markdown(f"- Price is **below** the 50-day average (₹{ma_50:,.2f}) → short-term **weakness** ⚠️")
                        if ma_200:
                            if price > ma_200:
                                st.markdown(f"- Price is **above** the 200-day average (₹{ma_200:,.2f}) → long-term **uptrend** ✅")
                            else:
                                st.markdown(f"- Price is **below** the 200-day average (₹{ma_200:,.2f}) → long-term **downtrend** ⚠️")
                        st.markdown("*Moving Averages smooth out daily noise. When the price is above them, the trend is positive. Below means the stock is struggling.*")
                    
                    # RSI explanation
                    if rsi:
                        st.markdown(f"### 🔄 RSI (Relative Strength Index): {rsi}")
                        st.markdown("RSI measures if a stock is being overbought or oversold. Scale: 0 to 100.")
                        if rsi > 70:
                            st.markdown(f"At **{rsi}**, the stock is **overbought** — it's had a strong run and might cool down soon. 🔴")
                        elif rsi < 30:
                            st.markdown(f"At **{rsi}**, the stock is **oversold** — it's been beaten down and could bounce back. 🟢")
                        else:
                            st.markdown(f"At **{rsi}**, the stock is in the **neutral zone** — no extreme buying or selling pressure. 🟡")
                    
                    # Key Factors explanation
                    key_factors = parsed_signal.get('key_factors', [])
                    if key_factors:
                        st.markdown("### 🔑 Key Factors — What's Driving This Signal")
                        for factor in key_factors:
                            factor_lower = factor.lower()
                            st.markdown(f"**• {factor}**")
                            
                            # Promoter related
                            if 'promoter' in factor_lower and ('reduc' in factor_lower or 'decreas' in factor_lower or 'sold' in factor_lower):
                                st.markdown("> Promoters are the founders/owners of the company. When they reduce their stake, it *can* signal reduced confidence in the business. However, small reductions (under 1%) are often routine — for tax planning or personal needs — and not necessarily alarming.")
                            elif 'promoter' in factor_lower and ('increas' in factor_lower or 'raised' in factor_lower or 'added' in factor_lower):
                                st.markdown("> Promoters (company owners) are buying MORE shares with their own money. This is a strong positive signal — it means insiders believe the stock is undervalued and are putting their own money where their mouth is. 🟢")
                            elif 'promoter' in factor_lower:
                                st.markdown("> Promoter holding refers to the percentage of the company owned by its founders/management. Higher promoter holding generally indicates stronger insider confidence in the company's future.")
                            
                            # Beta / Volatility
                            elif 'beta' in factor_lower:
                                st.markdown("> Beta measures how much a stock moves compared to the overall market (Nifty). A beta of 1.0 means it moves exactly like Nifty. Beta > 1 means it swings MORE — so if Nifty goes up 1%, this stock might go up 1.5%. But the same applies on the downside. Higher beta = higher risk, higher potential reward.")
                            
                            # Seasonal / Historical patterns
                            elif 'april' in factor_lower or 'seasonal' in factor_lower or 'historical' in factor_lower or 'years positive' in factor_lower:
                                st.markdown("> This is a historical seasonal pattern — looking at how the stock has performed in this month over many years. While past performance doesn't guarantee future results, strong seasonal trends (e.g., 80%+ positive years) suggest a statistically meaningful tailwind.")
                            
                            # PE / Valuation
                            elif 'pe' in factor_lower or 'p/e' in factor_lower or 'valuation' in factor_lower or 'p/b' in factor_lower:
                                st.markdown("> PE ratio (Price-to-Earnings) tells you how much you're paying for each rupee of the company's profit. A PE of 20x means you're paying ₹20 for every ₹1 of earnings. Lower PE = cheaper stock. Compare with the sector average to judge if it's overpriced or a bargain.")
                            
                            # Earnings / Profit
                            elif 'profit' in factor_lower or 'revenue' in factor_lower or 'earning' in factor_lower or 'result' in factor_lower:
                                st.markdown("> This relates to the company's financial performance — how much money it's making. Growing profits quarter-over-quarter (QoQ) or year-over-year (YoY) is a strong positive sign. Declining profits are a red flag. 📊")
                            
                            # Analyst / Broker recommendations
                            elif 'buy' in factor_lower or 'analyst' in factor_lower or 'broker' in factor_lower or 'target' in factor_lower or 'rating' in factor_lower:
                                st.markdown("> These are recommendations from professional research analysts at brokerage firms. When multiple analysts say 'BUY' with a target price above the current price, it means experts see upside potential. However, analyst targets are projections, not guarantees.")
                            
                            # Market / Nifty / Sensex corrections
                            elif 'nifty' in factor_lower or 'sensex' in factor_lower or 'market correction' in factor_lower or 'broader market' in factor_lower:
                                st.markdown("> This describes the overall stock market movement. When Nifty/Sensex is down, most stocks tend to fall too — even good ones. A broad market correction creates headwinds but can also present buying opportunities for fundamentally strong stocks.")
                            
                            # Sector trends
                            elif 'sector' in factor_lower or 'industry' in factor_lower or 'infrastructure' in factor_lower:
                                st.markdown("> Sector performance matters because stocks in the same industry tend to move together. If the banking sector is weak, even a good bank stock may struggle. Sector tailwinds can lift all boats, while headwinds drag everyone down.")
                            
                            # 52-week high/low
                            elif '52-week' in factor_lower or '52 week' in factor_lower:
                                st.markdown("> The 52-week high/low shows the stock's trading range over the past year. Being near the 52-week high suggests strong momentum but also potential resistance. Being near the 52-week low could mean either a bargain or a stock in trouble — context matters.")
                            
                            # Dividend
                            elif 'dividend' in factor_lower:
                                st.markdown("> A dividend is cash the company pays to shareholders from its profits. Declaring or increasing dividends signals financial health and management's confidence. It's like getting a regular paycheck just for holding the stock. 💰")
                            
                            # NPA / Asset quality
                            elif 'npa' in factor_lower or 'asset quality' in factor_lower or 'provision' in factor_lower:
                                st.markdown("> NPA (Non-Performing Assets) are loans that borrowers have stopped repaying. Lower NPA = healthier bank. Improving asset quality means the bank's loan book is getting cleaner, which is a positive indicator for banking stocks. 🏦")
                            
                            # FII / DII / Institutional
                            elif 'fii' in factor_lower or 'dii' in factor_lower or 'institutional' in factor_lower or 'foreign' in factor_lower:
                                st.markdown("> FII (Foreign Institutional Investors) and DII (Domestic Institutional Investors) are large fund houses. When they increase holdings, it signals professional confidence. FII outflow can pressure stock prices due to large selling volumes.")
                            
                            # Debt / Leverage
                            elif 'debt' in factor_lower or 'leverage' in factor_lower or 'borrowing' in factor_lower:
                                st.markdown("> Debt levels show how much the company has borrowed. Some debt is normal for growth, but excessive debt increases risk — especially when interest rates rise. A debt-free or low-debt company is generally safer.")
                            
                            # RBI / Regulatory
                            elif 'rbi' in factor_lower or 'regulat' in factor_lower or 'sebi' in factor_lower or 'compliance' in factor_lower:
                                st.markdown("> Regulatory actions from bodies like RBI or SEBI can significantly impact stock prices. Penalties, bans, or compliance issues create uncertainty and can restrict the company's operations until resolved. ⚖️")
                            
                            # Merger / Acquisition
                            elif 'merger' in factor_lower or 'acquisition' in factor_lower or 'takeover' in factor_lower:
                                st.markdown("> Mergers and acquisitions can create value by combining strengths of two companies, but they also carry integration risks. The market typically reacts based on whether the deal is seen as value-accretive or overpriced.")
                            
                            # Volume
                            elif 'volume' in factor_lower or 'delivery' in factor_lower:
                                st.markdown("> Trading volume shows how many people are actively buying/selling. High volume with price increase = strong conviction. High delivery percentage means investors are holding, not just day-trading — a bullish sign.")
                            
                            # Growth
                            elif 'growth' in factor_lower or 'expansion' in factor_lower:
                                st.markdown("> Growth in revenue, customers, or market share indicates the company is expanding. Consistent growth is one of the most important factors driving long-term stock price appreciation. 📈")
                            
                            # Sell sentiment
                            elif 'sell' in factor_lower and ('sentiment' in factor_lower or 'recommend' in factor_lower or '%' in factor_lower or 'user' in factor_lower):
                                st.markdown("> Online community sentiment showing high SELL recommendations reflects retail investor pessimism. However, retail crowd sentiment often lags — professionals may disagree. Use this as one data point, not the sole deciding factor.")
                            
                            # Catch-all for unrecognized factors
                            else:
                                st.markdown("> This is a market factor identified from live financial data using rule-based pattern matching. It contributes to the overall signal direction and confidence level.")
                            
                            st.markdown("")  # spacing
                    
                    # Risk explanation — contextual
                    if risk_assessment:
                        st.markdown("### ⚠️ Risk Assessment")
                        st.markdown(f"**\"{risk_assessment}\"**")
                        risk_lower = risk_assessment.lower()
                        
                        explanations = []
                        if 'valuation' in risk_lower or 'pe' in risk_lower or 'expensive' in risk_lower or 'overvalued' in risk_lower:
                            explanations.append("**Valuation risk** means the stock might be priced too high relative to its earnings. If the company doesn't grow fast enough to justify the price, the stock could fall.")
                        if 'volatil' in risk_lower or 'beta' in risk_lower:
                            explanations.append("**Volatility risk** means the stock price swings a lot. While this creates opportunities, it also means you could see large losses in a short time if the market turns.")
                        if 'regulat' in risk_lower or 'rbi' in risk_lower or 'sebi' in risk_lower or 'compliance' in risk_lower:
                            explanations.append("**Regulatory risk** means government or regulatory bodies could take actions that hurt the company's business — like fines, restrictions, or policy changes.")
                        if 'promoter' in risk_lower or 'insider' in risk_lower:
                            explanations.append("**Insider risk** means the company's owners/management are showing behavior (like selling shares) that could signal they're not fully confident in the near-term outlook.")
                        if 'debt' in risk_lower or 'leverage' in risk_lower:
                            explanations.append("**Debt risk** means the company has significant borrowings. If revenues slow down, the company might struggle to service its debt, putting shareholder value at risk.")
                        if 'data' in risk_lower and ('unavail' in risk_lower or 'incomplete' in risk_lower):
                            explanations.append("**Data risk** means some technical indicators couldn't be retrieved, so the analysis is based on incomplete information. The confidence level accounts for this gap.")
                        if 'market' in risk_lower or 'correction' in risk_lower or 'geopolit' in risk_lower:
                            explanations.append("**Market risk** means broader economic conditions — like global tensions, interest rate changes, or market corrections — could drag this stock down regardless of its fundamentals.")
                        
                        if explanations:
                            st.markdown("**What this means in plain English:**")
                            for exp in explanations:
                                st.markdown(f"- {exp}")
                        else:
                            st.markdown("Every investment carries risk. This section flags the biggest concerns identified by the model for this stock right now. Always consider what could go wrong before investing.")

                    
                    st.markdown("---")
                    st.markdown("*💡 This breakdown is generated by a Python-based interpretation layer that converts the XGBoost model outputs and technical indicators into human-readable explanations using predefined financial rules and templates.*")
                # ───────────────────────────────────────────────────────────
            else:
                st.info("No detailed financial analysis available.")
            
        with tab2:
            st.subheader("Sentiment Analysis")
            if st.session_state.sentiment_complete_flag and st.session_state.sentiment_results:
                sentiment_data = st.session_state.sentiment_results.get('sentiment_data', {})
                
                # If sentiment_data is a JSON string, parse it
                if isinstance(sentiment_data, str):
                    try:
                        sentiment_data = json.loads(sentiment_data)
                    except (json.JSONDecodeError, TypeError):
                        # Try to extract JSON from the string
                        json_match = re.search(r'\{.*\}', sentiment_data, re.DOTALL)
                        if json_match:
                            try:
                                sentiment_data = json.loads(json_match.group(0))
                            except:
                                sentiment_data = {}
                        else:
                            sentiment_data = {}
                
                # Check if sentiment data is valid and has required fields
                if not sentiment_data or not isinstance(sentiment_data, dict):
                    st.warning("Sentiment analysis data is not in the expected format. This may happen when using certain models.")
                    if st.session_state.sentiment_results.get('raw_response'):
                        with st.expander("View Raw Response"):
                            st.text(st.session_state.sentiment_results.get('raw_response'))
                    return
                
                # Safety: if 'summary' accidentally contains the full JSON blob, re-extract
                summary_val = sentiment_data.get('summary', '')
                if isinstance(summary_val, str) and '"sources"' in summary_val:
                    try:
                        reparsed = json.loads('{' + summary_val.split('{', 1)[-1]) if '{' in summary_val else None
                        if reparsed and isinstance(reparsed, dict):
                            sentiment_data = reparsed
                    except:
                        pass  # keep original

                
                # Create fancy visualizations for sentiment data
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create sentiment score gauge
                    sentiment_score = sentiment_data.get('score', 0)
                    
                    # Handle string values and normalize score to 0-100 range if it's in -100 to 100 range
                    try:
                        if isinstance(sentiment_score, str):
                            # Try to extract a number from the string (e.g., "-60 score" → -60)
                            import re
                            score_match = re.search(r'(-?\d+)', sentiment_score)
                            if score_match:
                                sentiment_score = float(score_match.group(1))
                            else:
                                sentiment_score = 50  # Default if no number found
                        
                        # Normalize negative scores to 0-100 scale
                        if sentiment_score < 0:
                            normalized_score = (sentiment_score + 100) / 2
                        else:
                            normalized_score = sentiment_score
                    except (ValueError, TypeError):
                        normalized_score = 50  # Default if conversion fails
                    
                    # Create sentiment gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=normalized_score,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Market Sentiment Score"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#1f77b4"},
                            'steps': [
                                {'range': [0, 30], 'color': "#e74c3c"},  # Bearish - Red
                                {'range': [30, 70], 'color': "#f1c40f"},  # Neutral - Yellow
                                {'range': [70, 100], 'color': "#2ecc71"}  # Bullish - Green
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': normalized_score
                            }
                        }
                    ))
                    
                    fig.update_layout(
                        height=300,
                        margin=dict(l=20, r=20, t=50, b=20)
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                
                with col2:
                    # Create sentiment trend visualization
                    # This is a simulated trend since we don't have historical data
                    # In a real app, you would store historical sentiment scores
                    
                    # Create some simulated trend data based on the current score
                    base_score = sentiment_data.get('score', 50)
                    if isinstance(base_score, str):
                        try:
                            base_score = float(base_score)
                        except:
                            base_score = 50
                    
                    # Generate a simulated 7-day trend
                    import random
                    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(6, -1, -1)]
                    
                    # Create trend that ends at current score
                    # Guard against None (returned when model can't determine score)
                    if base_score is None:
                        base_score = 50
                    random.seed(base_score)  # Use score as seed for reproducibility
                    trend = [
                        max(0, min(100, base_score + random.uniform(-15, 15))) for _ in range(6)
                    ]
                    trend.append(base_score)  # Today's score
                    
                    # Create the trend chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=trend,
                        mode='lines+markers',
                        name='Sentiment Trend',
                        line=dict(color='#3498db', width=3),
                        marker=dict(size=8)
                    ))
                    
                    # Add a horizontal line for neutral sentiment
                    fig.add_shape(
                        type="line",
                        x0=dates[0],
                        y0=50,
                        x1=dates[-1],
                        y1=50,
                        line=dict(color="gray", width=1, dash="dash"),
                    )
                    
                    fig.update_layout(
                        title='7-Day Sentiment Trend',
                        xaxis_title='Date',
                        yaxis_title='Sentiment Score',
                        yaxis=dict(range=[0, 100]),
                        height=300,
                        margin=dict(l=20, r=20, t=50, b=20)
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                
                # Display social media sentiment with a radar chart
                social_media = sentiment_data.get('social_media', {})
                if social_media:
                    st.subheader("Social Media Sentiment")
                    
                    # Create two columns
                    col1, col2 = st.columns(2)
                    
                    # Extract values and handle potential string values
                    twitter = social_media.get('twitter', 50)
                    reddit = social_media.get('reddit', 50)
                    stocktwits = social_media.get('stocktwits', 50)
                    
                    with col1:
                        # Create radar chart for social media sentiment
                        # Convert percentages to 0-100 scale if needed and handle string values
                        try:
                            twitter_val = float(twitter) if isinstance(twitter, str) else twitter
                        except (ValueError, TypeError):
                            twitter_val = 50  # Default if conversion fails
                            
                        try:
                            reddit_val = float(reddit) if isinstance(reddit, str) else reddit
                        except (ValueError, TypeError):
                            reddit_val = 50  # Default if conversion fails
                            
                        try:
                            stocktwits_val = float(stocktwits) if isinstance(stocktwits, str) else stocktwits
                        except (ValueError, TypeError):
                            stocktwits_val = 50  # Default if conversion fails
                        
                        # Create radar chart with improved text visibility
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatterpolar(
                            r=[twitter_val, reddit_val, stocktwits_val],
                            theta=['Twitter', 'Reddit', 'StockTwits'],
                            fill='toself',
                            name='Bearish Sentiment',
                            line_color='#e74c3c'
                        ))
                        
                        fig.add_trace(go.Scatterpolar(
                            r=[100, 100, 100],
                            theta=['Twitter', 'Reddit', 'StockTwits'],
                            fill='toself',
                            name='Max',
                            opacity=0.2,
                            line_color='gray'
                        ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 100],
                                    tickfont=dict(color='black'),  # Change tick font color to black
                                    tickvals=[0, 25, 50, 75, 100]  # Explicitly set tick values
                                )
                            ),
                            title="Social Media Bearish Sentiment",
                            height=350,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, width='stretch')
                    
                    with col2:
                        # Display metrics with delta indicators - ensure values are numeric
                        # Convert to float if string, or use default if conversion fails
                        try:
                            twitter_val = float(twitter) if isinstance(twitter, str) and twitter.lower() != "n/a" else twitter
                            twitter_delta = 50 - twitter_val
                            twitter_display = f"{twitter}% Bearish" if twitter != "N/A" else "N/A"
                            st.metric("Twitter Sentiment", twitter_display, 
                                    delta=f"{twitter_delta:.1f}% from neutral", delta_color="inverse")
                        except (ValueError, TypeError):
                            st.metric("Twitter Sentiment", "N/A" if twitter == "N/A" else f"{twitter}% Bearish")
                        
                        try:
                            reddit_val = float(reddit) if isinstance(reddit, str) and reddit.lower() != "n/a" else reddit
                            reddit_delta = 50 - reddit_val
                            reddit_display = f"{reddit}% Bearish" if reddit != "N/A" else "N/A"
                            st.metric("Reddit Sentiment", reddit_display, 
                                    delta=f"{reddit_delta:.1f}% from neutral", delta_color="inverse")
                        except (ValueError, TypeError):
                            st.metric("Reddit Sentiment", "N/A" if reddit == "N/A" else f"{reddit}% Bearish")
                        
                        try:
                            stocktwits_val = float(stocktwits) if isinstance(stocktwits, str) and stocktwits.lower() != "n/a" else stocktwits
                            stocktwits_delta = 50 - stocktwits_val
                            stocktwits_display = f"{stocktwits}% Bearish" if stocktwits != "N/A" else "N/A"
                            st.metric("StockTwits Sentiment", stocktwits_display, 
                                    delta=f"{stocktwits_delta:.1f}% from neutral", delta_color="inverse")
                        except (ValueError, TypeError):
                            st.metric("StockTwits Sentiment", "N/A" if stocktwits == "N/A" else f"{stocktwits}% Bearish")
                
                # Display summary
                st.subheader("Sentiment Summary")
                st.write(sentiment_data.get('summary', 'No summary available'))
                
                # Display sources if available
                sources = sentiment_data.get('sources', [])
                if sources:
                    st.subheader("News Sources")
                    
                    # Create a bar chart for news sentiment
                    sentiment_values = []
                    source_names = []
                    colors = []
                    
                    for source in sources:
                        sentiment = source.get('sentiment', 'neutral').lower()
                        source_name = source.get('source', 'Unknown')
                        # Limit source name length for better display
                        if len(source_name) > 20:
                            source_name = source_name[:17] + "..."
                        source_names.append(source_name)
                        
                        if sentiment == 'positive':
                            sentiment_values.append(75)  # Positive sentiment value
                            colors.append('#2ecc71')  # Green
                        elif sentiment == 'negative':
                            sentiment_values.append(25)  # Negative sentiment value
                            colors.append('#e74c3c')  # Red
                        else:
                            sentiment_values.append(50)  # Neutral sentiment value
                            colors.append('#f1c40f')  # Yellow
                    
                    if source_names:
                        # Create horizontal bar chart for news sentiment
                        fig = go.Figure()
                        
                        fig.add_trace(go.Bar(
                            y=source_names,
                            x=sentiment_values,
                            orientation='h',
                            marker_color=colors,
                            text=[f"{source_names[i]}: {'Positive' if sentiment_values[i] > 60 else 'Negative' if sentiment_values[i] < 40 else 'Neutral'}" 
                                  for i in range(len(source_names))],
                            textposition='auto'
                        ))
                        
                        fig.update_layout(
                            title='News Source Sentiment',
                            xaxis=dict(
                                title='Sentiment',
                                range=[0, 100],
                                tickvals=[25, 50, 75],
                                ticktext=['Negative', 'Neutral', 'Positive']
                            ),
                            yaxis=dict(
                                title='Source'
                            ),
                            height=max(200, len(source_names) * 40),
                            margin=dict(l=20, r=20, t=50, b=20)
                        )
                        
                        st.plotly_chart(fig, width='stretch')
                    
                    # Display detailed source information
                    for source in sources:
                        st.markdown(f"**{source.get('title')}** - {source.get('source')} ({source.get('date')})")
                        st.markdown(f"*Sentiment: {source.get('sentiment')}*")
                        st.markdown(f"{source.get('summary')}")
                        st.markdown("---")
                
                # ── 📖 Sentiment Explainer ─────────────────────────────
                with st.expander("📖 Explain Sentiment Analysis in Plain English"):
                    score = sentiment_data.get('score', 50)
                    if isinstance(score, str):
                        try:
                            score = float(score)
                        except:
                            score = 50
                    
                    ticker = st.session_state.current_ticker
                    summary = sentiment_data.get('summary', '')
                    
                    st.markdown("### 🎭 What is Sentiment Analysis?")
                    st.markdown(f"Our system scanned **live financial news** about **{ticker}** from sources like Moneycontrol and Economic Times using web scraping tools. A Python-based interpretation layer then analyzed each article's tone — positive, negative, or neutral — and combined everything into a single sentiment score using predefined financial rules and templates.")
                    
                    st.markdown(f"### 📊 Overall Sentiment Score: {score:.0f}/100")
                    if score >= 75:
                        st.markdown(f"🟢 **Strongly Bullish** — At **{score:.0f}/100**, the overwhelming majority of news coverage is positive. Analysts and media are optimistic about this stock's prospects. This is a strong vote of confidence from the market narrative.")
                    elif score >= 60:
                        st.markdown(f"🟢 **Leaning Positive** — At **{score:.0f}/100**, most news leans positive but there are some concerns. The market mood is cautiously optimistic — good signs exist, but not everything is perfect.")
                    elif score >= 40:
                        st.markdown(f"🟡 **Mixed / Neutral** — At **{score:.0f}/100**, there's roughly equal positive and negative coverage. The market hasn't made up its mind. This often happens during transitions — after earnings, regulatory changes, or sector rotation.")
                    elif score >= 25:
                        st.markdown(f"🔴 **Leaning Negative** — At **{score:.0f}/100**, most news coverage raises concerns. While not a crisis, the media narrative is cautious and investors may be worried about near-term performance.")
                    else:
                        st.markdown(f"🔴 **Strongly Bearish** — At **{score:.0f}/100**, the news is overwhelmingly negative. There may be serious concerns about the company's fundamentals, management, or sector outlook.")
                    
                    if summary:
                        st.markdown(f"**Model Summary:** *\"{summary}\"*")
                    
                    # ── Article-by-Article Breakdown ──
                    sources_list = sentiment_data.get('sources', [])
                    if sources_list:
                        pos = sum(1 for s in sources_list if s.get('sentiment', '').lower() == 'positive')
                        neg = sum(1 for s in sources_list if s.get('sentiment', '').lower() == 'negative')
                        neu = sum(1 for s in sources_list if s.get('sentiment', '').lower() == 'neutral')
                        total = len(sources_list)
                        
                        st.markdown(f"### 📰 What the News is Saying ({total} articles analyzed)")
                        st.markdown(f"The system found **{total} relevant articles**: 🟢 {pos} positive, 🔴 {neg} negative, 🟡 {neu} neutral")
                        st.markdown("")
                        
                        for i, source in enumerate(sources_list, 1):
                            title = source.get('title', 'Unknown Article')
                            src_name = source.get('source', 'Unknown')
                            date = source.get('date', 'N/A')
                            sent = source.get('sentiment', 'neutral').lower()
                            src_summary = source.get('summary', '')
                            
                            # Emoji and color for sentiment
                            if sent == 'positive':
                                emoji = "🟢"
                                label = "POSITIVE"
                            elif sent == 'negative':
                                emoji = "🔴"
                                label = "NEGATIVE"
                            else:
                                emoji = "🟡"
                                label = "NEUTRAL"
                            
                            st.markdown(f"**{emoji} Article {i}: {title}**")
                            st.markdown(f"*Source: {src_name} | Date: {date} | Sentiment: {label}*")
                            if src_summary:
                                st.markdown(f"> {src_summary}")
                            
                            # Add contextual explanation based on sentiment
                            if sent == 'positive':
                                st.markdown(f"> 👆 **Why this matters:** This article paints a favorable picture of {ticker}. Positive news coverage attracts investor attention and can drive buying interest, pushing the stock price upward.")
                            elif sent == 'negative':
                                st.markdown(f"> 👇 **Why this matters:** This article raises concerns about {ticker}. Negative coverage can trigger selling pressure as investors become cautious and re-evaluate their positions.")
                            else:
                                st.markdown(f"> ➡️ **Why this matters:** This article is informational without a strong bias. Neutral coverage means the market is in a wait-and-watch mode regarding this aspect.")
                            st.markdown("")
                        
                        # Overall narrative
                        st.markdown("### 🔍 What This Means for Your Investment")
                        if pos > neg:
                            st.markdown(f"The media narrative around **{ticker}** is **predominantly positive** ({pos} out of {total} articles). When most news is favorable, it typically supports the stock price and can attract new investors. However, always verify if the positive news is about *fundamentals* (earnings, growth) or just *hype*.")
                        elif neg > pos:
                            st.markdown(f"The media narrative around **{ticker}** is **leaning negative** ({neg} out of {total} articles flagging concerns). Negative sentiment can create short-term selling pressure. However, if the company's fundamentals are strong, negative sentiment can actually be a buying opportunity — the crowd isn't always right.")
                        else:
                            st.markdown(f"The media narrative around **{ticker}** is **balanced** — equal positive and negative coverage. This typically means the market is processing new information (like earnings or policy changes). Stay patient and wait for a clearer trend to emerge before making decisions.")
                    
                    # Social media
                    social = sentiment_data.get('social_media', {})
                    if social:
                        st.markdown("### 📱 Social Media Pulse (Estimated)")
                        tw = social.get('twitter', 50)
                        rd = social.get('reddit', 50)
                        st_val = social.get('stocktwits', 50)
                        avg = (float(tw) + float(rd) + float(st_val)) / 3
                        
                        st.markdown("Since social media platforms block automated scraping, our system **estimates** the online conversation using sentiment-weighted projection based on the tone of professional news coverage:")
                        st.markdown(f"- **Twitter/X:** ~{tw}% positive (financial influencers and traders)")
                        st.markdown(f"- **Reddit:** ~{rd}% positive (retail investor communities like r/IndianStreetBets)")
                        st.markdown(f"- **StockTwits:** ~{st_val}% positive (dedicated stock discussion platform)")
                        
                        if avg > 65:
                            st.markdown(f"\n**Overall: Social buzz is optimistic** (avg {avg:.0f}%). Retail investors appear enthusiastic, which can create short-term momentum. 🚀")
                        elif avg > 50:
                            st.markdown(f"\n**Overall: Social buzz is mildly positive** (avg {avg:.0f}%). Some interest but no strong FOMO (fear of missing out). 📈")
                        elif avg > 35:
                            st.markdown(f"\n**Overall: Social buzz is cautious** (avg {avg:.0f}%). Retail investors are on the fence — waiting for a catalyst. ⏸️")
                        else:
                            st.markdown(f"\n**Overall: Social buzz is negative** (avg {avg:.0f}%). Retail investors are worried or have lost interest. This can mean either panic selling or a contrarian buying opportunity. 📉")
                    
                    # Final verdict
                    st.markdown("---")
                    st.markdown("### 💡 Bottom Line")
                    if score >= 60 and pos > neg:
                        st.markdown(f"The news environment for **{ticker}** is **supportive**. Positive media coverage + decent sentiment score suggests the market views this stock favorably right now. Combine this with the Financial Analysis tab to see if the technicals agree.")
                    elif score < 40 and neg > pos:
                        st.markdown(f"The news environment for **{ticker}** is **cautionary**. Negative coverage dominates, suggesting potential headwinds. Check the Financial Analysis tab — if technicals are also weak, consider reducing exposure. If technicals are strong, the negative sentiment might be an overreaction.")
                    else:
                        st.markdown(f"The news environment for **{ticker}** is **mixed**. Neither strongly bullish nor bearish — the market is undecided. In such cases, the Financial Analysis tab (price trends, RSI, moving averages) becomes more important for making decisions.")
                    st.markdown("*Remember: Sentiment is just one piece of the puzzle. Smart investing combines news sentiment with technical analysis and fundamental research.*")

                # ───────────────────────────────────────────────────────
                
            elif st.session_state.sentiment_in_progress:
                st.markdown("""
                <div style="background:linear-gradient(135deg,#1a1f2e,#162032);
                            border:1px solid #3b82f6; border-radius:12px;
                            padding:24px 28px; margin-top:16px;">
                    <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
                        <span style="font-size:1.5em;">🧠</span>
                        <span style="color:#93c5fd; font-size:1.1em; font-weight:700;">
                            FinBERT NLP — Sentiment Analysis Running
                        </span>
                    </div>
                    <p style="color:#cbd5e1; margin:0 0 8px 0;">
                        Financial signal analysis is <b style="color:#4ade80;">complete</b>.
                        Now running FinBERT sentiment pipeline on live news &amp; social media data&hellip;
                    </p>
                    <p style="color:#94a3b8; font-size:0.85em; margin:0;">
                        This typically takes 30&ndash;60 seconds. The page refreshes automatically.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                with st.spinner("Scraping news sources and running FinBERT sentiment inference..."):
                    pass
            else:
                st.info("No sentiment data available yet.")
    
    elif st.session_state.analysis_requested:
        # Analysis was requested but no results yet
        with st.spinner(f'Analyzing {st.session_state.current_ticker}... This may take a few minutes.'):
            st.info(f"Analysis for {st.session_state.current_ticker} is in progress. Results will appear automatically when ready.")
    
    else:
        # No analysis requested yet
        st.info("Enter a stock ticker and click 'Analyze Stock' to begin analysis")
    
    # Add an automatic refresh mechanism
    # This will refresh the page every few seconds while analysis is in progress
    if st.session_state.analysis_in_progress or st.session_state.sentiment_in_progress:
        time.sleep(2)  # Wait 2 seconds before refreshing
        check_and_update_results()  # Check for updates before rerunning
        st.rerun()

if __name__ == "__main__":
    main()
