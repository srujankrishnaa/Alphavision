import asyncio
import logging
from datetime import datetime
from typing import Dict

from strands import Agent
from shared_resources import get_model, get_mcp_client

# Configure basic logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("sentiment_analysis")
logger.setLevel(logging.INFO)

# Standard system prompt for Bedrock
SYSTEM_PROMPT_BEDROCK = "You are a financial sentiment analysis agent."

# Enhanced system prompt for Ollama models with explicit tool use instructions
SYSTEM_PROMPT_OLLAMA = """You are a financial sentiment analysis agent.
You have access to tools that can help you gather information from the web.

IMPORTANT: When you need information, ALWAYS use the available tools to get it.
DO NOT suggest code or explain how to use tools - ACTUALLY USE THE TOOLS DIRECTLY.

CRITICAL: You MUST use the scrape_as_markdown tool to get sentiment data about stocks.
For example: scrape_as_markdown(url="https://finance.yahoo.com/quote/TICKER")
Replace TICKER with the actual stock symbol. This is essential for providing accurate sentiment information.

When you need to use a tool, follow this exact format:
<tool>
name: scrape_as_markdown
parameters:
  url: "https://finance.yahoo.com/quote/TICKER"
</tool>

Replace TICKER with the actual stock symbol. After getting the data, analyze it and provide your findings.

Your task is to analyze market sentiment for stocks based on news and analyst data.

**For Indian stocks** (.NS or .BO suffix):
- News: Moneycontrol.com, Economic Times, Business Standard, Mint, NDTV Profit
- Analysts: Screener.in, Tickertape.in
- Use INR for prices

**For US stocks**:
- News: Yahoo Finance, CNBC, Bloomberg, MarketWatch
- Analysts: Seeking Alpha, Motley Fool

DO NOT attempt to scrape Twitter, Reddit, or StockTwits — these sites block scrapers
and will cause very long timeouts. Instead, estimate social sentiment from news tone.

You MUST return your analysis in valid JSON format as shown below:

```json
{
    "score": 75,
    "summary": "Overall bullish sentiment based on recent earnings beat and positive analyst ratings",
    "sources": [
        {
            "title": "Example Article Title",
            "source": "Yahoo Finance",
            "date": "2025-05-25",
            "sentiment": "positive",
            "summary": "Brief summary of the article content"
        }
    ],
    "social_media": {
        "twitter": 80,
        "reddit": 65,
        "stocktwits": 70
    }
}
```
"""

async def get_sentiment_analysis(ticker: str, model_settings: Dict = None) -> Dict:
    """Get sentiment analysis for a stock
    
    Args:
        ticker (str): Stock ticker symbol to analyze
        model_settings (Dict): Settings for the model to use
            {
                "model_type": "bedrock" or "ollama",
                "ollama_host": "http://localhost:11434",
                "ollama_model_id": "llama3"
            }
    """
    logger.info(f"Starting sentiment analysis for {ticker}")
    
    try:
        # Get resources for this thread
        model_settings = model_settings or {}
        model = get_model(
            model_type=model_settings.get("model_type", "bedrock"),
            ollama_host=model_settings.get("ollama_host", "http://localhost:11434"),
            ollama_model_id=model_settings.get("ollama_model_id", "llama3-groq-tool-use"),
            bedrock_model_id=model_settings.get("bedrock_model_id"),
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
                logger.info(f"Loaded {len(tools)} tools for sentiment analysis")
                
                # Create the agent with the tools
                agent = Agent(
                    model=model,
                    tools=tools,
                    system_prompt=system_prompt
                )
                
                # Use the agent to analyze sentiment
                response = ""
                async for event in agent.stream_async(
                    f"""Analyze the market sentiment for {ticker} stock.

                    CRITICAL TOOL INSTRUCTIONS:
                    - ONLY use 'scrape_as_markdown' and 'search_engine' tools.
                    - NEVER use the 'discover' tool — it will timeout.
                    - You MUST make at most 1 search_engine call and 3 scrape_as_markdown calls.
                    - Do NOT exceed 4 total tool calls. Going over will cause a timeout crash.

                    Strategy: 1 search to find URLs, then scrape the top 3 results.
                    PRIMARY sources for Indian stocks (.NS): Moneycontrol, Economic Times, Business Standard.
                    PRIMARY sources for US stocks: Yahoo Finance, CNBC, MarketWatch.

                    FALLBACK: If primary sources don't yield 3+ articles, use these backup sources:
                    Indian: Livemint (livemint.com), NDTV Profit (ndtvprofit.com), Financial Express (financialexpress.com), Screener.in, Finology (ticker.finology.in)
                    US: Reuters, Seeking Alpha, Investopedia

                    IMPORTANT: You MUST extract at least 3 news articles in the "sources" array.
                    Each scraped page often contains multiple headlines — pick the most relevant ones.
                    If a page has 5 headlines, include the top 2-3 from that page.

                    Do NOT scrape Twitter, Reddit, StockTwits, or any social media site.
                    ESTIMATE social_media sentiment percentages based on the news tone you find.

                    Return ONLY a valid JSON object with this exact structure (no extra text):
                    {{
                        "score": [overall sentiment score 0 to 100],
                        "summary": "[concise 2-3 sentence summary of overall sentiment]",
                        "sources": [
                            {{
                                "title": "[article title]",
                                "source": "[source name]",
                                "date": "[publication date YYYY-MM-DD]",
                                "sentiment": "[positive/negative/neutral]",
                                "summary": "[1 sentence summary]"
                            }}
                        ],
                        "social_media": {{
                            "twitter": [estimated sentiment % based on news tone],
                            "reddit": [estimated sentiment % based on news tone],
                            "stocktwits": [estimated sentiment % based on news tone]
                        }}
                    }}"""
                ):
                    if "data" in event:
                        response += event["data"]
                
                # Try to parse the response as JSON
                try:
                    import json
                    import re
                    
                    # Strip out <thinking>...</thinking> tags from the model's internal reasoning
                    # These leak into the response and pollute the summary field
                    response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL).strip()
                    
                    # Check if the response looks like a search result list instead of JSON
                    if "search results for the query" in response or "list of search results" in response:
                        logger.warning("Response appears to be search results instead of JSON. Attempting to extract sentiment.")
                        
                        # Create a simplified sentiment data structure from the search results
                        sentiment_data = {
                            "score": 0,  # Neutral by default
                            "summary": "Unable to generate detailed sentiment analysis. Search results were returned instead of analysis.",
                            "sources": [],
                            "social_media": {
                                "twitter": 50,
                                "reddit": 50,
                                "stocktwits": 50
                            }
                        }
                        
                        # Try to extract sentiment from titles
                        bullish_count = 0
                        bearish_count = 0
                        
                        # Look for bullish/bearish indicators in the text
                        bullish_terms = ["buy", "bullish", "positive", "good", "growth", "opportunity", "upside"]
                        bearish_terms = ["sell", "bearish", "negative", "bad", "decline", "risk", "downside"]
                        
                        for term in bullish_terms:
                            bullish_count += response.lower().count(term)
                        
                        for term in bearish_terms:
                            bearish_count += response.lower().count(term)
                        
                        # Calculate a simple sentiment score based on term frequency
                        if bullish_count > bearish_count:
                            sentiment_score = 60 + min(30, (bullish_count - bearish_count) * 5)
                            sentiment_summary = f"Slightly bullish sentiment detected in search results (score: {sentiment_score})"
                        elif bearish_count > bullish_count:
                            sentiment_score = 40 - min(30, (bearish_count - bullish_count) * 5)
                            sentiment_summary = f"Slightly bearish sentiment detected in search results (score: {sentiment_score})"
                        else:
                            sentiment_score = 50
                            sentiment_summary = "Neutral sentiment detected in search results"
                        
                        sentiment_data["score"] = sentiment_score
                        sentiment_data["summary"] = sentiment_summary
                        
                        # Extract potential sources from the response
                        title_matches = re.findall(r'\|\s*\d+\s*\|\s*(.*?)\s*\|', response)
                        for i, title in enumerate(title_matches[:5]):  # Take up to 5 titles
                            if title and not title.startswith('Title'):  # Skip header rows
                                sentiment_data["sources"].append({
                                    "title": title.strip(),
                                    "source": "Search Result",
                                    "date": datetime.now().strftime("%Y-%m-%d"),
                                    "sentiment": "neutral",
                                    "summary": "Extracted from search results"
                                })
                        
                        logger.info(f"Created simplified sentiment data from search results")
                        return {
                            "status": "success",
                            "ticker": ticker,
                            "sentiment_data": sentiment_data,
                            "timestamp": datetime.now().isoformat()
                        }
                    
                    # Extract JSON from the response (it might be surrounded by markdown code blocks)
                    json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        # Try to find outermost JSON object in raw response
                        brace_match = re.search(r'(\{.*\})', response, re.DOTALL)
                        if brace_match:
                            json_str = brace_match.group(1)
                        else:
                            json_str = response
                    
                    # Clean up the string to ensure it's valid JSON
                    json_str = re.sub(r'(?m)^\s*//.*$', '', json_str)  # Remove comments
                    json_str = json_str.strip()
                    
                    try:
                        sentiment_data = json.loads(json_str)
                        logger.info(f"Successfully parsed sentiment data for {ticker}")
                    except json.JSONDecodeError:
                        # If JSON parsing fails, try to create a structured response from the text
                        logger.warning(f"JSON parsing failed, attempting to extract structured data from text")
                        
                        # Create a basic sentiment data structure
                        sentiment_data = {
                            "score": 50,  # Neutral by default
                            "summary": "Sentiment analysis extracted from unstructured text.",
                            "sources": [],
                            "social_media": {
                                "twitter": 50,
                                "reddit": 50,
                                "stocktwits": 50
                            }
                        }
                        
                        # Try to extract sentiment score
                        score_match = re.search(r'sentiment score[:\s]*(-?\d+)', response, re.IGNORECASE)
                        if score_match:
                            try:
                                score = int(score_match.group(1))
                                # Normalize to -100 to 100 range if needed
                                if score < -100 or score > 100:
                                    score = max(-100, min(100, score))
                                sentiment_data["score"] = score
                            except ValueError:
                                pass
                        
                        # Try to extract summary
                        summary_match = re.search(r'summary[:\s]*(.*?)(?:\n\n|\Z)', response, re.IGNORECASE | re.DOTALL)
                        if summary_match:
                            sentiment_data["summary"] = summary_match.group(1).strip()
                        
                        logger.info(f"Created structured sentiment data from text")
                    
                    return {
                        "status": "success",
                        "ticker": ticker,
                        "sentiment_data": sentiment_data,
                        "timestamp": datetime.now().isoformat()
                    }
                except Exception as e:
                    logger.error(f"Error parsing sentiment JSON: {str(e)}")
                    # Return the raw response if parsing fails
                    return {
                        "status": "error",
                        "ticker": ticker,
                        "message": f"Error parsing sentiment data: {str(e)}",
                        "raw_response": response,
                        "timestamp": datetime.now().isoformat()
                    }
                        
            except Exception as e:
                logger.error(f"Error using MCP tools for sentiment: {str(e)}")
                return {
                    "status": "error",
                    "ticker": ticker,
                    "message": f"Error using MCP tools: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
    except Exception as e:
        logger.error(f"Error in get_sentiment_analysis: {str(e)}")
        return {
            "status": "error",
            "ticker": ticker,
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

# For testing directly
if __name__ == "__main__":
    ticker = "AAPL"
    result = asyncio.run(get_sentiment_analysis(ticker))
    print(f"Sentiment for {ticker}:")
    print(result)
