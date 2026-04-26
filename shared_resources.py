import os
import logging
import requests
from botocore.config import Config as BotocoreConfig
from strands.models import BedrockModel
from strands.models.ollama import OllamaModel
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shared_resources")

def check_ollama_status(host="http://localhost:11434"):
    """Check if Ollama is running and return available models
    
    Args:
        host (str): Ollama host address
        
    Returns:
        tuple: (is_running, available_models)
    """
    try:
        response = requests.get(f"{host}/api/tags")
        if response.status_code == 200:
            available_models = [model['name'] for model in response.json().get("models", [])]
            logger.info(f"✅ Ollama is running. Available models: {', '.join(available_models)}")
            return True, available_models
        else:
            logger.warning(f"❌ Ollama returned status code {response.status_code}")
            return False, []
    except requests.exceptions.ConnectionError:
        logger.warning("❌ Ollama is not running. Please start Ollama before proceeding.")
        return False, []
    except Exception as e:
        logger.error(f"❌ Error checking Ollama status: {str(e)}")
        return False, []

def get_model(
    model_type="bedrock",
    ollama_host="http://localhost:11434",
    ollama_model_id="llama3.1:latest",
    bedrock_model_id=None,
):
    """Get a model instance based on the specified type
    
    Args:
        model_type (str): Type of model to use - "bedrock" or "ollama"
        ollama_host (str): Host address for Ollama server
        ollama_model_id (str): Model ID to use with Ollama
        
    Returns:
        A model instance compatible with Strands Agent
    """
    if model_type.lower() == "bedrock":
        logger.info("Using AWS Bedrock model")
        resolved_model_id = bedrock_model_id or os.environ.get("BEDROCK_MODEL_ID") or "us.amazon.nova-premier-v1:0"
        return BedrockModel(
            model_id=resolved_model_id,
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            temperature=0.1,
            boto_client_config=BotocoreConfig(
                read_timeout=300,  # 5 min — prevents timeout on large context
                retries={"max_attempts": 2, "mode": "standard"},
            ),
        )
    elif model_type.lower() == "ollama":
        # Check if Ollama is running before trying to use it
        is_running, available_models = check_ollama_status(ollama_host)
        if not is_running:
            raise ConnectionError(f"Cannot connect to Ollama at {ollama_host}. Please ensure Ollama is running.")
        
        # Extract base model name for checking (without :latest tag)
        base_model_id = ollama_model_id.split(':')[0] if ':' in ollama_model_id else ollama_model_id
        
        # Check if the requested model is available (checking both with and without :latest)
        model_available = ollama_model_id in available_models or base_model_id in available_models
        
        if available_models and not model_available:
            logger.warning(f"Model '{ollama_model_id}' not found in available models: {available_models}")
            if len(available_models) > 0:
                logger.info(f"Falling back to available model: {available_models[0]}")
                ollama_model_id = available_models[0]
            else:
                raise ValueError(f"No models available in Ollama. Please pull a model first.")
        
        logger.info(f"Using Ollama model {ollama_model_id} at {ollama_host}")
        return OllamaModel(
            host=ollama_host,
            model_id=ollama_model_id,
            params={
                "temperature": 0.3,  # Lower for more deterministic responses, higher for more creative
                "top_p": 0.9,  # Nucleus sampling parameter
                "stream": True,  # Enable streaming responses
            },
        )
    else:
        logger.error(f"Unknown model type: {model_type}")
        raise ValueError(f"Unknown model type: {model_type}. Supported types: bedrock, ollama")

def get_mcp_client():
    """Get an MCP client instance"""
    # Create a new instance each time to ensure thread safety
    # Use environment variable for API_TOKEN
    api_token = os.environ.get("API_TOKEN")
    
    if not api_token:
        logger.error("API_TOKEN environment variable is not set. Please set it before running the application.")
        raise ValueError("API_TOKEN environment variable is not set")
    
    # Increase startup timeout to 120 seconds (default is 30) for slow first-time MCP server initialization
    return MCPClient(
        lambda: stdio_client(
            StdioServerParameters(
                command="npx",
                args=["-y", "@brightdata/mcp"],
                env={
                    "API_TOKEN": api_token
                }
            )
        ),
        startup_timeout=120
    )

# Test Ollama status if this file is run directly
if __name__ == "__main__":
    is_running, available_models = check_ollama_status()
    if is_running:
        print("Available Ollama models:")
        for model in available_models:
            print(f"- {model}")
    else:
        print("Ollama is not running or not available.")
