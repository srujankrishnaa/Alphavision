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

def save_results(result_type, data):
    """Save results to a file for cross-thread communication"""
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, f"{result_type}.json")
    
    with open(result_path, 'w') as f:
        json.dump(data, f)

def load_results(result_type):
    """Load results from a file"""
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    result_path = os.path.join(results_dir, f"{result_type}.json")
    
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            return json.load(f)
    return None

# Function to extract technical data from signal text
def extract_technical_data(text):
    """Extract technical data from signal text for visualization"""
    data = {}
    
    # Extract price (supports $, ₹, and commas)
    price_match = re.search(r'Price:\s*[\$₹]?\s*([\d,]+\.?\d*)', text)
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
def run_analysis_thread(ticker, model_settings=None):
    """Run stock analysis in a separate thread
    
    Args:
        ticker: Stock ticker symbol
        model_settings: Dictionary with model provider settings
    """
    try:
        logger.info(f"Starting analysis thread for {ticker}")
        
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Use provided model settings or default to Bedrock
        if model_settings is None:
            model_settings = {"model_type": "bedrock"}
        logger.info(f"Using model settings: {model_settings}")
        
        # Run the analysis (properly awaiting the coroutine)
        results = loop.run_until_complete(analyze_stock(ticker, model_settings))
        
        # Save results to file
        if results and isinstance(results, dict) and results.get('status') == 'success':
            # Parse the signal text
            signal_text = results.get('signal', '')
            parsed_signal = parse_signal_text(signal_text)
            
            # Add the parsed signal to the results
            results['parsed_signal'] = parsed_signal
            
            # Save to file
            save_results("analysis", results)
            set_flag("analysis_complete", True)
            logger.info(f"Analysis completed for {ticker} and saved to file")
        else:
            logger.error(f"Error in analysis: {results}")
            set_flag("analysis_error", True)
        
        # Close the loop
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
            logger.info("Updated session state with analysis results")
        set_flag("analysis_complete", False)  # Reset the flag
    
    # Check for analysis errors
    if check_flag("analysis_error"):
        st.session_state.analysis_in_progress = False
        st.session_state.error_message = "Error during analysis"
        logger.info("Analysis error detected")
        set_flag("analysis_error", False)  # Reset the flag
    
    # Check for completed sentiment analysis
    if check_flag("sentiment_complete"):
        results = load_results("sentiment")
        if results:
            st.session_state.sentiment_results = results
            st.session_state.sentiment_in_progress = False
            st.session_state.sentiment_complete_flag = True
            logger.info("Updated session state with sentiment results")
        set_flag("sentiment_complete", False)  # Reset the flag
    
    # Check for sentiment analysis errors
    if check_flag("sentiment_error"):
        st.session_state.sentiment_in_progress = False
        logger.info("Sentiment analysis error detected")
        set_flag("sentiment_error", False)  # Reset the flag

def main():
    # Check for completed analyses
    check_and_update_results()
    
    st.title("📈 Financial Signals Dashboard")
    
    # Display current stock and date at the top with enhanced styling
    if st.session_state.analysis_results:
        ticker = st.session_state.analysis_results.get('ticker', '')
        current_date = datetime.now().strftime("%B %d, %Y")
        
        # Get price from parsed signal if available
        price = None
        if st.session_state.parsed_signal and 'technical_data' in st.session_state.parsed_signal:
            price = st.session_state.parsed_signal['technical_data'].get('price')
        
        # Format price display properly
        if price is not None:
            try:
                price_display = f"${price:.2f}"
            except (ValueError, TypeError):
                price_display = f"${price}"
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
            st.session_state.sentiment_in_progress = True
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
                # Bedrock-only mode: Nova Premier for financial + Nova Lite for sentiment (no payment required)
                financial_settings = {
                    "model_type": "bedrock",
                    "bedrock_model_id": "us.amazon.nova-premier-v1:0",
                }
                sentiment_settings = {
                    "model_type": "bedrock",
                    "bedrock_model_id": "amazon.nova-lite-v1:0",  # Use Nova Lite for sentiment (faster/cheaper)
                }
                
                logger.info("Using Bedrock-only mode: Nova Premier for financial analysis + Nova Lite for sentiment")
                
                # Start analysis thread with Bedrock (Nova)
                analysis_thread = threading.Thread(
                    target=run_analysis_thread, 
                    args=(ticker, financial_settings)
                )
                analysis_thread.daemon = True
                analysis_thread.start()
                
                # Start sentiment thread with Bedrock (Claude)
                sentiment_thread = threading.Thread(
                    target=run_sentiment_thread, 
                    args=(ticker, sentiment_settings)
                )
                sentiment_thread.daemon = True
                sentiment_thread.start()
                
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
                    st.plotly_chart(direction_chart, use_container_width=True)
                    
                    # Display confidence score gauge
                    confidence = parsed_signal.get('confidence_score', 50)
                    confidence_chart = create_signal_gauge_chart(confidence)
                    st.plotly_chart(confidence_chart, use_container_width=True)
                
                with col2:
                    # Display risk-reward chart
                    risk_reward_chart = create_risk_reward_chart(parsed_signal)
                    st.plotly_chart(risk_reward_chart, use_container_width=True)
                    
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
                        st.plotly_chart(price_chart, use_container_width=True)
                
                with col2:
                    # Display RSI gauge
                    rsi = tech_data.get('rsi')
                    if rsi:
                        rsi_chart = create_rsi_gauge(rsi)
                        st.plotly_chart(rsi_chart, use_container_width=True)
                
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
            else:
                st.info("No detailed financial analysis available.")
            
        with tab2:
            st.subheader("Sentiment Analysis")
            if st.session_state.sentiment_complete_flag and st.session_state.sentiment_results:
                sentiment_data = st.session_state.sentiment_results.get('sentiment_data', {})
                
                # Check if sentiment data is valid and has required fields
                if not sentiment_data or not isinstance(sentiment_data, dict):
                    st.warning("Sentiment analysis data is not in the expected format. This may happen when using certain models.")
                    if st.session_state.sentiment_results.get('raw_response'):
                        with st.expander("View Raw Response"):
                            st.text(st.session_state.sentiment_results.get('raw_response'))
                    return
                
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
                    
                    st.plotly_chart(fig, use_container_width=True)
                
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
                    
                    st.plotly_chart(fig, use_container_width=True)
                
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
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
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
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display detailed source information
                    for source in sources:
                        st.markdown(f"**{source.get('title')}** - {source.get('source')} ({source.get('date')})")
                        st.markdown(f"*Sentiment: {source.get('sentiment')}*")
                        st.markdown(f"{source.get('summary')}")
                        st.markdown("---")
                
            elif st.session_state.sentiment_in_progress:
                st.info("Sentiment analysis is in progress...")
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
