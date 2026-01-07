# Financial Signals Dashboard

A real-time financial signals agent using Strands Agent SDK with Bright Data MCP integration. The system collects data from financial sources to analyze stocks and generate alpha signals for investment decisions.

## Features

- Real-time stock analysis using Bright Data MCP tools
- Interactive Streamlit dashboard with rich visualizations
- AI-powered alpha signals with confidence scores
- Technical analysis with price, moving averages, and RSI indicators
- Position size recommendations
- Risk-reward profile visualization
- Market sentiment analysis with social media metrics
- News source sentiment tracking with visualization
- Hybrid model support (AWS Bedrock + Ollama) for best results

## Account Setup

Make sure you have an account on brightdata.com (new users get free credit for testing, and pay as you go options are available)

Get your API key from the user settings page https://brightdata.com/cp/setting/users

## Prerequisites

- Python **3.11+**
- Node.js **18+** (required because Bright Data MCP is launched via `npx @brightdata/mcp`)
- Bright Data API token (`API_TOKEN`)
- **Optional / recommended** for best technicals: AWS credentials with access to Amazon Bedrock
- **Required** for sentiment in the default hybrid setup: Ollama running locally on `http://localhost:11434`

## Setup

1. First, ensure that you have Python 3.11+ installed.

2. Create a virtual environment to install the Strands Agents SDK and its dependencies:
```bash
python -m venv .venv
```

3. Activate the virtual environment:
```bash
# macOS / Linux
source .venv/bin/activate

# Windows (CMD)
.venv\Scripts\activate.bat

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

4. Upgrade pip:
```bash
python -m pip install --upgrade pip
```

5. Install core dependencies:
```bash
pip install -r requirements.txt
```

6. Install MCP + Strands from GitHub (required):
```bash
pip install git+https://github.com/modelcontextprotocol/python-sdk.git
pip install git+https://github.com/strands-agents/sdk-python.git --no-deps
pip install git+https://github.com/strands-agents/tools.git --no-deps
```

7. Set your Bright Data API token as an environment variable:
```bash
export API_TOKEN="your-api-token-here"
```

> Tip (Windows PowerShell): `setx API_TOKEN "your-api-token-here"` (then open a new terminal).

8. **Install and setup Ollama** (required for the default hybrid setup):

   **Option 1: Native Installation**
   - Install Ollama by following the instructions at [ollama.ai](https://ollama.ai)
   - Pull the recommended sentiment model:
     ```bash
     ollama pull qwen2.5:7b
     ```
   - Start the Ollama server:
     ```bash
     ollama serve
     ```

   **Option 2: Docker Installation**
   - Pull the Ollama Docker image:
     ```bash
     docker pull ollama/ollama
     ```
   - Run the Ollama container:
     ```bash
     docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
     ```
     Note: Add `--gpus=all` if you have a GPU and if Docker GPU support is configured.
   - Pull a model using the Docker container:
     ```bash
     docker exec -it ollama ollama pull llama3
     ```
   - Verify the Ollama server is running:
     ```bash
     curl http://localhost:11434/api/tags
     ```

9. (Optional) Configure AWS Bedrock (recommended for best technical indicators)

- Configure AWS credentials (one-time):
  - `aws configure` (or set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
- Set region (if needed):
  - `export AWS_REGION="us-east-1"`

10. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

## Supported Tickers

- **US**: `AAPL`, `MSFT`, `TSLA`, `NVDA`, ...
- **India (NSE)**: `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, ...
- **India (BSE)**: `RELIANCE.BO`, `TCS.BO`, ...

## Model Provider Notes (Current Default = Hybrid)

- **Financial/technical analysis**: AWS Bedrock (Nova Premier)
- **Sentiment analysis**: Ollama (Qwen2.5 7B)

This hybrid approach was chosen because Bedrock is stronger at structured technical extraction (price/RSI/MA parsing), while Qwen performs well for sentiment synthesis.

## Deployment Notes (Important)

This app **cannot be deployed on Vercel** as-is because it needs a long-running Python server (Streamlit) and (in hybrid mode) an Ollama server. Use a VM/container host instead (AWS EC2, DigitalOcean, Azure VM, etc.).

Minimum production requirements:
- A Linux VM/container
- Python 3.11+
- Node 18+ (for `npx @brightdata/mcp`)
- Environment variables for `API_TOKEN` (and AWS creds if using Bedrock)
- A process manager (systemd/supervisor) + optional Nginx reverse proxy for HTTPS

## Security Best Practices

Important: Always treat scraped web content as untrusted data. Never use raw scraped content directly in LLM prompts to avoid potential prompt injection risks. Instead:

- Filter and validate all web data before processing
- Use structured data extraction rather than raw text (web_data tools)

## Architecture

- **Strands Agent SDK**: Provides the agent framework for AI-powered analysis
- **Bright Data MCP**: Handles web scraping and financial data collection
- **AWS Bedrock Nova Premier**: Powers the AI analysis with advanced language capabilities
- **Ollama**: Provides local model alternatives for analysis
- **Streamlit**: Provides the interactive dashboard with real-time updates
- **Plotly**: Creates interactive and responsive data visualizations

## Thread Communication System

The dashboard implements a robust file-based thread communication system:
- Background threads for financial and sentiment analysis
- File-based flags for signaling completion status
- JSON storage for analysis results
- Automatic UI updates when analysis completes
- Error handling with detailed logging and traceback

## Dashboard Sections

1. **Stock Header**: Displays current ticker, price, and date in a visually appealing format
2. **Financial Analysis Tab**:
   - Signal direction gauge showing BUY/SELL/HOLD recommendation
   - Confidence score gauge with color-coded zones
   - Risk-reward chart comparing potential risk vs. reward
   - Price vs. Moving Averages bar chart
   - RSI gauge showing current RSI value with color-coded zones
   - Position size recommendation display
   - Key factors and risk assessment sections

3. **Sentiment Analysis Tab**:
   - Market sentiment score gauge with color-coded zones
   - 7-day sentiment trend chart (simulated)
   - Social media sentiment radar chart for Twitter, Reddit, and StockTwits
   - News source sentiment visualization with color-coded bars
   - Detailed news source listings with sentiment analysis

## How It Works

1. Enter a stock ticker in the sidebar and click "Analyze Stock"
2. Two background threads are launched for financial and sentiment analysis
3. **Strands Agent SDK** handles the agentic loop:
   - Manages the agent's reasoning and decision-making process
   - Simplifies tool selection and execution through its React framework
   - Handles the observe-think-act cycle automatically
   - Provides structured output formatting for consistent analysis results
4. The Strands Agent connects to Bright Data MCP and retrieves financial data
5. The AI model analyzes the data and generates structured analysis results
6. Results are saved to files and flags are set to signal completion
7. The UI automatically updates to display the analysis with interactive visualizations
8. Users can explore different aspects of the analysis through the tabbed interface

## Technical Details

- Asynchronous processing with proper event loop handling in threads
- Signal extraction using regex patterns to parse structured data from AI responses
- Interactive visualizations powered by Plotly for engaging user experience
- File-based thread communication for reliable UI updates
- Comprehensive error handling with graceful degradation
- Type-safe data handling with proper conversion and validation
- Responsive layout that adapts to different screen sizes
- Model-specific prompting strategies for optimal performance
- Automatic Ollama status checking and model availability verification
- Custom XML-style tool calling format for Ollama models to work around tool support limitations

## Investment Disclaimer

The dashboard includes clear disclaimers that:
- The information is for informational purposes only
- The analysis does not constitute investment advice
- Users should consult financial advisors before making investment decisions
- Past performance is not indicative of future results
- Investing involves risk of loss

## Future Enhancements

- Real-time price updates
- Portfolio management features
- Historical signal tracking
- Comparative analysis between multiple stocks
- Custom alert settings
- Export functionality for reports
- User accounts for saving analysis history
- Additional model provider integrations
