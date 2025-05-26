import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict

from strands import Agent
from shared_resources import get_model, get_mcp_client

# Configure basic logging - increase level to reduce verbosity
logging.basicConfig(level=logging.ERROR)  # Change from INFO to ERROR to reduce noise
logger = logging.getLogger("financial_signals")
logger.setLevel(logging.INFO)  # Keep our own logger at INFO level

# System prompt for the financial signals agent (Bedrock)
SYSTEM_PROMPT_BEDROCK = """You are a financial analysis agent specialized in generating alpha signals.
When analyzing stocks, provide output in the following structured format:

SIGNAL_START
Direction: [BUY/SELL/HOLD]
Confidence Score: [0-100]%
Position Size: [Recommended position size]

Technical Analysis:
- Price: [Current price]
- Moving Averages: [Key MA indicators]
- RSI: [Current RSI value]
- Volume: [Volume analysis]

Key Factors:
- [Factor 1]
- [Factor 2]
- [Factor 3]

Risk Assessment:
[Detailed risk analysis]

Market Context:
[Brief market context]

Recommendation:
[Specific actionable recommendation]
SIGNAL_END"""

# Enhanced system prompt for Ollama models with explicit tool use instructions
SYSTEM_PROMPT_OLLAMA = """You are a financial analysis agent specialized in generating alpha signals.
You have access to tools that can help you gather information from the web.

IMPORTANT: When you need information, ALWAYS use the available tools to get it.
For example, use the scrape_as_markdown tool to get data from financial websites.
When a tool returns information, analyze it carefully before making decisions.

CRITICAL: You MUST use the scrape_as_markdown tool to get the current stock price from Yahoo Finance.
For example: scrape_as_markdown(url="https://finance.yahoo.com/quote/TICKER")
Replace TICKER with the actual stock symbol. This is essential for providing accurate price information.

After receiving the scraped content, you MUST extract the current price. Look for patterns like:
- "TICKER XXX.XX +/-X.XX (+/-X.XX%)" at the top of the page
- A prominent number followed by currency symbol
- A section labeled "Quote Price" or similar

When you find the price, you MUST include it in your response under "Price:" in the Technical Analysis section.

When calling a function, use the exact format required by the tool. For example:
- For scrape_as_markdown, use: scrape_as_markdown(url="https://example.com")
- For search_engine, use: search_engine(query="search query here")

When analyzing stocks, provide output in the following structured format:

SIGNAL_START
Direction: [BUY/SELL/HOLD]
Confidence Score: [0-100]%
Position Size: [Recommended position size]

Technical Analysis:
- Price: [Current price - MUST include the actual price from Yahoo Finance]
- Moving Averages: [Key MA indicators]
- RSI: [Current RSI value]
- Volume: [Volume analysis]

Key Factors:
- [Factor 1]
- [Factor 2]
- [Factor 3]

Risk Assessment:
[Detailed risk analysis]

Market Context:
[Brief market context]

Recommendation:
[Specific actionable recommendation]
SIGNAL_END"""

async def analyze_stock(ticker: str, model_settings: Dict = None) -> Dict:
    """Analyze a stock and generate trading signals
    
    Args:
        ticker (str): Stock ticker symbol to analyze
        model_settings (Dict): Settings for the model to use
            {
                "model_type": "bedrock" or "ollama",
                "ollama_host": "http://localhost:11434",
                "ollama_model_id": "llama3"
            }
    """
    logger.info(f"Starting analysis for {ticker}")
    
    try:
        # Get resources for this thread
        model_settings = model_settings or {}
        model = get_model(
            model_type=model_settings.get("model_type", "bedrock"),
            ollama_host=model_settings.get("ollama_host", "http://localhost:11434"),
            ollama_model_id=model_settings.get("ollama_model_id", "llama3-groq-tool-use")
        )
        mcp_client = get_mcp_client()
        
        # Select appropriate system prompt based on model type
        if model_settings.get("model_type", "bedrock").lower() == "ollama":
            system_prompt = SYSTEM_PROMPT_OLLAMA
            logger.info("Using enhanced system prompt for Ollama model")
        else:
            system_prompt = SYSTEM_PROMPT_BEDROCK
            logger.info("Using standard system prompt for Bedrock model")
        
        # Create an agent with MCP tools
        with mcp_client:
            try:
                # Get tools from the MCP server
                tools = mcp_client.list_tools_sync()
                logger.info(f"Loaded {len(tools)} tools from Bright Data MCP")
                
                # Create the agent with the tools
                agent = Agent(
                    model=model,
                    tools=tools,
                    system_prompt=system_prompt
                )
                
                # Use a simpler approach - just get the basic data and generate a signal
                response = ""
                async for event in agent.stream_async(
                    f"""Analyze {ticker} stock and provide a concise alpha signal.
                    
                    Use the scrape_as_markdown tool to get data from Investing.com for {ticker} instead of using Yahoo Finance.
                    Investing.com page aggregates multiple technical indicators and analyst ratings, which are critical for validating signals.
                    
                    Keep your analysis brief and focused on the most important data points.
                    
                    Format your response as:
                    
                    SIGNAL_START
                    Direction: [BUY/SELL/HOLD]
                    Confidence Score: [0-100]%
                    Position Size: [brief recommendation]
                    
                    Technical Analysis:
                    - Price: [current price]
                    - Key indicators: [1-2 key points]
                    
                    Key Factors:
                    - [Factor 1]
                    - [Factor 2]
                    
                    Risk Assessment:
                    [1 sentence]
                    
                    Recommendation:
                    [1-2 sentences]
                    SIGNAL_END"""
                ):
                    if "data" in event:
                        response += event["data"]
                        
            except Exception as e:
                logger.error(f"Error using MCP tools: {str(e)}")
                # Fall back to using the agent without tools
                logger.warning(f"Falling back to agent without tools for {ticker}")
                
                # Create agent without tools
                agent = Agent(
                    model=model,
                    tools=[],
                    system_prompt=system_prompt
                )
                
                # Use the agent to analyze the stock
                response = ""
                async for event in agent.stream_async(
                    f"""Analyze {ticker} stock and provide a concise alpha signal.
                    
                    Format your response as:
                    
                    SIGNAL_START
                    Direction: [BUY/SELL/HOLD]
                    Confidence Score: [0-100]%
                    Position Size: [brief recommendation]
                    
                    Technical Analysis:
                    - Price: [estimate]
                    - Key indicators: [1-2 key points]
                    
                    Key Factors:
                    - [Factor 1]
                    - [Factor 2]
                    
                    Risk Assessment:
                    [1 sentence]
                    
                    Recommendation:
                    [1-2 sentences]
                    SIGNAL_END"""
                ):
                    if "data" in event:
                        response += event["data"]
        
        # Check if response contains the markers, if not add them
        if "SIGNAL_START" not in response:
            logger.info("Adding missing SIGNAL_START marker")
            response = "SIGNAL_START\n" + response
        if "SIGNAL_END" not in response:
            logger.info("Adding missing SIGNAL_END marker")
            response = response + "\nSIGNAL_END"
            
        # Extract the signal portion
        try:
            signal_start = response.find("SIGNAL_START")
            signal_end = response.find("SIGNAL_END")
            if signal_start != -1 and signal_end != -1:
                signal = response[signal_start + len("SIGNAL_START"):signal_end].strip()
                logger.info(f"Successfully extracted signal: {len(signal)} characters")
            else:
                logger.warning("Could not find signal markers, using full response")
                signal = response
        except Exception as e:
            logger.error(f"Error extracting signal: {str(e)}")
            signal = response
        
        logger.info(f"Analysis completed for {ticker}")
        logger.info(f"Signal content preview: {signal[:50]}...")
        
        return {
            "status": "success",
            "ticker": ticker,
            "signal": signal,
            "raw_response": response,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in analyze_stock: {str(e)}")
        return {
            "status": "error",
            "ticker": ticker,
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

# For testing the agent directly
if __name__ == "__main__":
    ticker = "AAPL"
    result = asyncio.run(analyze_stock(ticker))
    print(f"Analysis for {ticker}:")
    print(result["signal"])
