#!/usr/bin/env python

"""
Question Answering Agent Implementation

Detects and answers questions using an LLM. The agent identifies questions based on both
natural language patterns and vector similarity, then uses an LLM to generate accurate
and helpful answers.
"""

import json
import logging
import os
import re
import uuid
import asyncio
import time
from typing import Dict, Any, Optional, List, Union

# For containerized agents, use the local base agent
# This avoids dependencies on the semsubscription module
try:
    # First try to import from semsubscription if available (for local development)
    from semsubscription.agents.EnhancedAgent import EnhancedAgent as BaseAgent
except ImportError:
    try:
        # Fall back to local agent_base for containerized environments
        # Don't use relative import (from .) in templates - it causes errors in containers
        from agent_base import BaseAgent
    except ImportError:
        try:
            # Last resort for Docker environment with current directory
            import sys
            # Add the current directory to the path to find agent_base.py
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from agent_base import BaseAgent
        except ImportError:
            # If all else fails, define a minimal BaseAgent class for compatibility
            class BaseAgent:
                """Minimal implementation of BaseAgent for compatibility"""
                def __init__(self, agent_id=None, name=None, description=None, similarity_threshold=0.7, **kwargs):
                    self.agent_id = agent_id or str(uuid.uuid4())
                    self.name = name or self.__class__.__name__
                    self.description = description or ""
                    self.similarity_threshold = similarity_threshold
                    self.config = kwargs.get('config', {})
                    self.classifier_threshold = 0.5  # Fixed threshold for testing

                def calculate_interest(self, message):
                    """
                    Calculate interest level for a message.
                    
                    This uses the fine-tuned model when available, or falls back to the
                    development implementation when no model is available.
                    
                    Args:
                        message: Message to calculate interest for (dict or object)
                        
                    Returns:
                        float: Interest score between 0.0 and 1.0
                    """
                    # Extract content based on message type (dict or object)
                    content = ""
                    message_id = "unknown"
                    
                    if isinstance(message, dict):
                        content = message.get('content', 'No content in dict')
                        message_id = message.get('id', 'unknown-id')
                    else:
                        content = getattr(message, 'content', 'No content in object')
                        message_id = getattr(message, 'id', 'unknown-id')
                    
                    logging.info(f"Calculating interest for message {message_id}")
                    logging.info(f"Message content: {content[:100]}...")
                    
                    # Use the interest model properly if it exists
                    if hasattr(self, 'interest_model') and self.interest_model:
                        try:
                            # This is the key method to call for the interest model
                            interest_score = self.interest_model.calculate_similarity(content)
                            logging.info(f"Interest model score: {interest_score}")
                            return interest_score
                        except Exception as e:
                            logging.error(f"Error using interest model: {e}")
                            # Continue to fallback implementation
                    
                    # Fallback implementation
                    # Simple implementation: keyword matching for domain relevance
                    keywords = [
                        # Add domain-specific keywords here
                        "question", "answer", "why", "how", "what", "when", "who"
                    ]
                    
                    # Count keyword matches
                    matches = sum(1 for keyword in keywords if keyword.lower() in content.lower())
                    
                    # Calculate interest score based on keyword density
                    if matches > 0:
                        # At least one keyword match - express interest
                        interest_score = min(0.5 + (matches * 0.1), 1.0)  # Scale with matches, cap at 1.0
                    else:
                        # No keywords match, still provide minimal interest
                        interest_score = 0.6  # Default minimal interest
                    
                    logging.info(f"Fallback interest calculation: {interest_score} (based on {matches} keyword matches)")
                    return interest_score
                
                def process_message(self, message):
                    """Process a message and return a test confirmation response"""
                    # Extract message details for better logging
                    content = message.get('content', 'No content') if isinstance(message, dict) else getattr(message, 'content', 'No content')
                    message_id = message.get('id', 'unknown-id') if isinstance(message, dict) else getattr(message, 'id', 'unknown-id')
                    
                    logger.info(f"Test agent processing message {message_id}")
                    logger.info(f"Message content: {content[:100]}...")
                    
                    # Build and return a response
                    response = {
                        "agent": self.__class__.__name__,
                        "status": "success",
                        "message": "This is a test confirmation from the default agent template",
                        "received": content,
                        "processed_at": time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    logger.info(f"Generated response: {str(response)[:200]}...")
                    return response

# Import required modules
import requests
import aiohttp
import os

# Define a robust Core API client for LLM and memory interactions
class CoreLLMClient:
    def __init__(self, core_api_url=None):
        self.core_api_url = core_api_url or os.environ.get("CORE_API_URL", "http://host.docker.internal:8888")
        logger.info(f"Initialized CoreLLMClient with API URL: {self.core_api_url}")
        
    def get_completion(self, prompt, model="gpt-4", temperature=0.7, system_prompt=None):
        """
        Get a completion from the LLM via the core API
        """
        try:
            # Prepare messages based on whether system_prompt is provided
            if system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            else:
                messages = [{"role": "user", "content": prompt}]
                
            # Make the API call
            response = requests.post(
                f"{self.core_api_url}/api/llm/generate",
                json={
                    "messages": messages,
                    "model": model,
                    "temperature": temperature
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("text", "")
            else:
                logger.error(f"API Error ({response.status_code}): {response.text}")
                return f"Error getting completion: {response.status_code}"
        except Exception as e:
            logger.error(f"Error in get_completion: {str(e)}")
            return f"This is a fallback response due to an error: {str(e)}"
            
    async def get_completion_async(self, prompt, model="gpt-4", temperature=0.7, system_prompt=None, max_tokens=None):
        """
        Get a completion from the LLM via the core API (async version)
        """
        try:
            # Prepare messages based on whether system_prompt is provided
            if system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            else:
                messages = [{"role": "user", "content": prompt}]
                
            # Prepare request payload
            payload = {
                "messages": messages,
                "model": model,
                "temperature": temperature
            }
            
            # Add max_tokens if provided
            if max_tokens:
                payload["max_tokens"] = max_tokens
                
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.core_api_url}/api/llm/generate",
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("text", "")
                    else:
                        error_text = await response.text()
                        logger.error(f"API Error ({response.status}): {error_text}")
                        return f"Error getting completion: {response.status}"
        except Exception as e:
            logger.error(f"Error in get_completion_async: {str(e)}")
            return f"This is a fallback async response due to an error: {str(e)}"
    
    def search_memory(self, query, limit=5, threshold=0.65):
        """
        Search for relevant context in the memory system
        """
        try:
            # Make the API call
            response = requests.post(
                f"{self.core_api_url}/api/memory/search",
                json={
                    "query": query,
                    "limit": limit,
                    "threshold": threshold
                }
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Memory API Error ({response.status_code}): {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error in search_memory: {str(e)}")
            return []
    
    async def process_message(self, message_id, agent_id, result):
        """
        Post processing results back to the core system
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.core_api_url}/api/messages/{message_id}/process",
                    json={
                        "agent_id": agent_id,
                        "result": result
                    }
                ) as response:
                    if response.status == 200:
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Process API Error ({response.status}): {error_text}")
                        return False
        except Exception as e:
            logger.error(f"Error in process_message: {str(e)}")
            return False

# Create a global client instance
try:
    core_llm_client = CoreLLMClient()
except Exception as e:
    logger.error(f"Failed to initialize CoreLLMClient: {str(e)}")
    # Define fallback functions to prevent further errors
    def get_completion(prompt, model="gpt-4", temperature=0.7, system_prompt=None):
        logger.warning("Using fallback LLM completion function - no actual LLM calls will be made")
        return f"This is a fallback response. In production, this would be answered using an LLM with prompt: {prompt[:50]}..."
        
    async def get_completion_async(prompt, model="gpt-4", temperature=0.7, system_prompt=None):
        logger.warning("Using fallback async LLM completion function - no actual LLM calls will be made")
        return f"This is a fallback async response. In production, this would be answered using an LLM with prompt: {prompt[:50]}..."

# Try to import memory system
try:
    from semsubscription.memory.vector_memory import get_memory_system
except ImportError:
    # Define a dummy function if not available
    def get_memory_system():
        logger.warning("Using fallback memory system - no actual memory retrieval will be performed")
        return None
        
logger = logging.getLogger(__name__)

# Primary class name without Agent suffix (modern style)
class Test_agent14(BaseAgent):
    """
    Agent that detects and answers questions using an LLM
    
    This agent specializes in identifying questions in natural language and providing
    accurate, helpful answers using an LLM. It uses both pattern matching and
    vector similarity to determine if a message contains a question.
    """
    
    def __init__(self, agent_id=None, name=None, description=None, similarity_threshold=0.6, **kwargs):
        """
        Initialize the agent with its parameters and setup the classifier
        
        Args:
            agent_id: Optional unique identifier for the agent
            name: Optional name for the agent (defaults to class name)
            description: Optional description of the agent
            similarity_threshold: Threshold for similarity-based interest determination
        """
        # Set default name if not provided
        name = name or "Question Answering Agent"  # Use a descriptive name
        description = description or "Detects and answers questions using an LLM"  # Describe what your agent does
        
        # Call the parent class init with our parameters
        super().__init__(
            agent_id=agent_id, 
            name=name,
            description=description,
            similarity_threshold=similarity_threshold,
            **kwargs
        )
        
        # Set up configuration with defaults if not specified
        self.config = kwargs.get('config', {})
            
        # Extract classifier settings
        self.use_classifier = self.config.get('use_classifier', True)
        self.classifier_threshold = self.config.get('classifier_threshold', 0.5)
        
        # LLM configuration
        self.llm_config = self.config.get('llm', {})
        self.llm_model = self.llm_config.get('model', 'gpt-4')
        self.llm_temperature = self.llm_config.get('temperature', 0.7)
        self.llm_max_tokens = self.llm_config.get('max_tokens', 1000)
        self.system_prompt = self.llm_config.get('system_prompt', 
            "You are an AI assistant specializing in answering questions. "  
            "Provide accurate, helpful, and concise answers.")
            
        # Memory system for context retrieval
        self.memory_system = get_memory_system()
        
        # Setup the interest model (for determining what messages to process)
        self.setup_interest_model()
    
    def setup_interest_model(self):
        """
        Set up the agent's interest model, which determines what messages it processes
        This is called automatically during initialization
        """
        # Check for fine-tuned model directory
        # Look for two different types of fine-tuned models:
        # 1. An embedding model (for SentenceTransformer)
        # 2. An interest model (numpy saved file)
        
        model_dir = os.path.join(os.path.dirname(__file__), "fine_tuned_model")
        embedding_model_path = model_dir  # The embedding model would be in the directory itself
        interest_model_path = os.path.join(model_dir, "interest_model.npz")  # The interest vectors
        
        logger.info(f"Looking for fine-tuned models in: {model_dir}")
        
        # List the model directory for debugging
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            files = os.listdir(model_dir)
            logger.info(f"Fine-tuned model directory contains: {files}")
            
            try:
                # Import necessary components for fine-tuned model
                try:
                    # First try importing from semsubscription
                    from semsubscription.vector_db.embedding import EmbeddingEngine, InterestModel
                except ImportError:
                    # Fall back to local implementation for containerized environments
                    from .interest_model import CustomInterestModel as InterestModel
                    from .embedding_engine import EmbeddingEngine
                
                # Create the interest model first
                if os.path.exists(interest_model_path):
                    logger.info(f"Found pre-calculated interest model: {interest_model_path}")
                    # Create a default embedding engine and then load the saved interest vectors
                    embedding_engine = EmbeddingEngine()  # Using default model
                    self.interest_model = InterestModel(embedding_engine=embedding_engine, model_path=interest_model_path)
                    logger.info(f"Successfully loaded pre-calculated interest model")
                # If we have a custom embedding model, load it
                elif os.path.exists(os.path.join(embedding_model_path, "config.json")):
                    logger.info(f"Found custom sentence transformer model at: {embedding_model_path}")
                    embedding_engine = EmbeddingEngine(model_name=embedding_model_path)
                    logger.info(f"Successfully loaded custom embedding model")
                    self.interest_model = InterestModel(embedding_engine=embedding_engine)
                else:
                    logger.warning(f"No valid fine-tuned model found in {model_dir}")
                    # Fall back to standard setup
                    super().setup_interest_model()
                    return
                    
                # Set threshold for the model
                self.interest_model.threshold = self.similarity_threshold
                
                # Domain-specific keywords can be added here
                # self.interest_model.keywords.extend([
                #     "specific_keyword",
                #     "another_keyword"
                # ])
                
                return  # Exit early, we've set up the model successfully
            except Exception as e:
                logger.error(f"Error setting up fine-tuned model: {e}")
                logger.warning("Falling back to default interest model setup")
        
        # Fall back to standard setup if fine-tuned model doesn't exist or fails
        super().setup_interest_model()
        
        # Add domain-specific customizations to the default model
        # For example, to add keywords that should always be of interest:
        # self.interest_model.keywords.extend([
        #     "specific_keyword",
        #     "another_keyword"
        # ])
        
        # Set up question detection patterns for is_interested method
        self.question_patterns = [
            re.compile(r"\b(?:what|who|where|when|why|how|can|could|would|should|is|are|will|do|does)\b.*\?", re.IGNORECASE),  # Question words + question mark
            re.compile(r"\?\s*$"),  # Ends with question mark
            re.compile(r"\bcan you\b|\bcould you\b|\bwould you\b", re.IGNORECASE),  # Implicit questions
            re.compile(r"\btell me\b|\bexplain\b|\bdescribe\b", re.IGNORECASE),  # Knowledge seeking
            re.compile(r"\bi (?:need|want|would like) to know\b", re.IGNORECASE),  # Explicit knowledge request
            re.compile(r"\bi'm (?:curious|wondering|asking|interested)\b", re.IGNORECASE)  # Curiosity indicators
        ]
    
    def is_interested(self, message):
        """
        Determine if the message contains a question
        
        Args:
            message: The message to check
            
        Returns:
            True if the message contains a question, False otherwise
        """
        # Extract content based on message type
        if isinstance(message, dict):
            content = message.get('content', '')
        else:
            content = getattr(message, 'content', '')
            
        # Use pattern matching to detect questions
        if hasattr(self, 'question_patterns'):
            for pattern in self.question_patterns:
                if pattern.search(content):
                    logger.info(f"Question pattern matched: {pattern.pattern}")
                    return True
                    
        # If no pattern matches, use the classifier or similarity model
        # This will be handled by the parent class calculate_interest method
        return False
    
    async def process_message_async(self, message):
        """
        Process questions using LLM and retrieve relevant context
        
        Args:
            message: The message to process
            
        Returns:
            Response data with the answer
        """
        try:
            # Extract content based on message type (dict or object)
            if isinstance(message, dict):
                content = message.get('content', '')
                message_id = message.get('id', 'unknown-id')
            else:
                content = getattr(message, 'content', '')
                message_id = getattr(message, 'id', 'unknown-id')
                
            logger.info(f"Processing question asynchronously: {content[:100]}...")
            
            # Retrieve relevant context using CoreLLMClient
            context_items = []
            try:
                if hasattr(self, 'memory_system') and self.memory_system:
                    # Try using the memory system first
                    logger.info(f"Retrieving context from memory system")
                    context_items = await asyncio.to_thread(
                        self._retrieve_relevant_context, content
                    )
                else:
                    # Fall back to the CoreLLMClient for memory search
                    logger.info(f"Retrieving context from CoreLLMClient")
                    context_items = await asyncio.to_thread(
                        core_llm_client.search_memory,
                        query=content,
                        limit=5,
                        threshold=0.65
                    )
                
                logger.info(f"Retrieved {len(context_items)} context items")
            except Exception as ctx_error:
                logger.error(f"Error retrieving context: {str(ctx_error)}")
            
            # Enhance the system prompt with context
            system_prompt = self._get_enhanced_system_prompt(context_items)
            
            # Set up the prompt for the LLM
            user_prompt = f"Question: {content}\n\nPlease provide a helpful answer."
            
            # Get completion from LLM using CoreLLMClient
            try:
                # Try with the CoreLLMClient first
                answer = await core_llm_client.get_completion_async(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    model=self.llm_model if hasattr(self, 'llm_model') else "gpt-4",
                    temperature=self.llm_temperature if hasattr(self, 'llm_temperature') else 0.7,
                    max_tokens=self.llm_max_tokens if hasattr(self, 'llm_max_tokens') else 1000
                )
                
                logger.info(f"LLM generated answer: {answer[:100]}...")
                
                # Prepare result for API posting
                result = {
                    "answer": answer,
                    "confidence": 0.95,
                    "model_used": self.llm_model if hasattr(self, 'llm_model') else "gpt-4",
                    "context_used": len(context_items) > 0
                }
                
                # Post the response back to the event bus using CoreLLMClient
                success = await core_llm_client.process_message(
                    message_id=message_id,
                    agent_id=self.agent_id,
                    result=result
                )
                
                if success:
                    logger.info(f"Successfully posted response to event bus for message {message_id}")
                else:
                    logger.error(f"Failed to post response to event bus for message {message_id}")
                    
                # Create response data for internal use
                response_data = {
                    "agent": self.name,
                    "message_id": message_id,
                    "question": content,
                    "answer": answer,
                    "timestamp": time.time(),
                    "context_used": len(context_items) > 0
                }
                
                return response_data
            except Exception as e:
                logger.error(f"Error getting LLM completion: {e}")
                # Fall back to synchronous version
                result = self.process_message(message)
                return result
                
        except Exception as e:
            logger.error(f"Error in async message processing: {e}")
            return {
                "agent": self.name,
                "error": str(e),
                "message_id": message_id if 'message_id' in locals() else "unknown"
            }
            
    def process_message(self, message):
        """
        Fallback for processing without LLM
        
        Args:
            message: The message to process
            
        Returns:
            Response data with a simple answer
        """
        try:
            # Extract content based on message type (dict or object)
            if isinstance(message, dict):
                content = message.get('content', '')
                query = content.lower()
                message_id = message.get('id', 'unknown-id')
            else:
                content = getattr(message, 'content', '')
                query = content.lower()
                message_id = getattr(message, 'id', 'unknown-id')
                
            logger.info(f"Processing message {message_id}: {content[:100]}...")
            
            # Use CoreLLMClient to get completion and post results
            try:
                # Simple version of the prompt without context
                system_prompt = self.system_prompt if hasattr(self, 'system_prompt') else "You are a helpful assistant."
                user_prompt = f"Question: {content}\n\nPlease provide a helpful answer."
                
                # Use the CoreLLMClient to get a completion
                answer = core_llm_client.get_completion(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    model=self.llm_model if hasattr(self, 'llm_model') else "gpt-4",
                    temperature=self.llm_temperature if hasattr(self, 'llm_temperature') else 0.7
                )
                
                # Prepare API result
                result = {
                    "answer": answer,
                    "confidence": 0.95,
                    "model_used": self.llm_model if hasattr(self, 'llm_model') else "gpt-4"
                }
                
                # Use requests directly since we're in sync context
                try:
                    process_url = f"{core_llm_client.core_api_url}/api/messages/{message_id}/process"
                    logger.info(f"Posting response to {process_url}")
                    
                    process_response = requests.post(
                        process_url,
                        json={
                            "agent_id": self.agent_id,
                            "result": result
                        }
                    )
                    
                    if process_response.status_code == 200:
                        logger.info(f"Successfully posted response to event bus for message {message_id}")
                    else:
                        logger.error(f"Error posting to API: {process_response.status_code} - {process_response.text}")
                except Exception as api_error:
                    logger.error(f"Error posting to API: {str(api_error)}")
                
                # Return the response data for internal use
                return {
                    "agent": self.name,
                    "message_id": message_id,
                    "question": content,
                    "answer": answer,
                    "timestamp": time.time()
                }
            except Exception as e:
                logger.error(f"Error in LLM completion: {e}")
                # Fallback pattern matching for basic responses
                answer = None
                if 'help' in query or 'hello' in query:
                    answer = f"Hello! I'm {self.name}, an agent that {self.description.lower()}. Without my LLM connection, I can only provide limited responses, but I'd be happy to try to help you with your questions."
                elif 'what' in query and 'you' in query and 'do' in query:
                    answer = f"I am a question answering agent. I detect questions in messages and provide helpful answers using an LLM. I'm designed to identify questions based on both language patterns and semantic meaning."
                else:
                    # Default response if no pattern matches and LLM is unavailable
                    answer = f"I understand you're asking a question, but I'm currently operating without my LLM capabilities. In production, I would provide a detailed answer to your question using an advanced language model."
                
                # Try to post the fallback answer to the event bus
                try:
                    process_url = f"{core_llm_client.core_api_url}/api/messages/{message_id}/process"
                    logger.info(f"Posting fallback response to {process_url}")
                    
                    process_response = requests.post(
                        process_url,
                        json={
                            "agent_id": self.agent_id,
                            "result": {
                                "answer": answer,
                                "confidence": 0.8,
                                "model_used": "fallback"
                            }
                        }
                    )
                    
                    if process_response.status_code == 200:
                        logger.info(f"Successfully posted fallback response to event bus for message {message_id}")
                    else:
                        logger.error(f"Error posting fallback to API: {process_response.status_code} - {process_response.text}")
                except Exception as api_error:
                    logger.error(f"Error posting fallback to API: {str(api_error)}")
                    
                # Return the response data
                return {
                    "agent": self.name,
                    "message_id": message_id,
                    "question": content,
                    "answer": answer,
                    "timestamp": time.time()
                }
            
        except Exception as e:
            logger.error(f"Error in Question Answering Agent processing: {e}")
            return {
                "agent": self.name,
                "error": str(e),
                "question": content if 'content' in locals() else "unknown question",
                "message_id": message_id if 'message_id' in locals() else "unknown"
            }
            
    def _retrieve_relevant_context(self, question: str, max_items: int = 5, threshold: float = 0.65):
        """
        Retrieve relevant context from memory system
        
        Args:
            question: The question to find context for
            max_items: Maximum number of memory items to retrieve
            threshold: Similarity threshold for retrieval
            
        Returns:
            List of relevant memory items
        """
        try:
            if not self.memory_system:
                logger.warning("No memory system available for context retrieval")
                return []
                
            # Query the memory system for relevant information
            context_items = self.memory_system.search_memories(
                query=question,
                limit=max_items,
                threshold=threshold
            )
            
            if not context_items:
                logger.info(f"No relevant context found for question: {question[:50]}...")
                return []
                
            logger.info(f"Found {len(context_items)} relevant context items")
            return context_items
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
            
    def _get_enhanced_system_prompt(self, context_items: List[Dict[str, Any]]):
        """
        Enhance the system prompt with relevant context
        
        Args:
            context_items: List of relevant memory items
            
        Returns:
            Enhanced system prompt
        """
        # If no context, return the base prompt
        if not context_items:
            return self.system_prompt
            
        # Add context to the prompt
        context_parts = []
        context_parts.append(self.system_prompt)
        context_parts.append("\n\nYou have access to the following relevant information that may help you answer the question.")
        context_parts.append("Use this information if relevant to the question:\n")
        
        # Add each context item
        for i, item in enumerate(context_items):
            content = item.get('content', item.get('text', ''))
            context_parts.append(f"Context {i+1}: {content}\n")
            
        # Add closing instructions
        context_parts.append("\nEnd of context information.\n")
        context_parts.append("If the context doesn't contain information relevant to the question, just answer based on your knowledge.")
        context_parts.append("Do not disclose that you were given any context information.")
        
        # Join all parts into the final prompt
        return '\n'.join(context_parts)


# Define the class with Agent suffix for backwards compatibility
# This prevents import errors in the container
class Test_agent14Agent(Test_agent14):
    """Legacy class name with Agent suffix"""
    pass

# For backwards compatibility, also create a more descriptive name
class QuestionAnsweringAgent(Test_agent14):
    """Descriptive name for the question answering agent"""
    pass

# Legacy compatibility for BaseAgent fallback imports
BaseAgent = Test_agent14

# For standalone testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create the agent
    agent = Test_agent14Agent()
    print(f"Agent created: {agent.name}")
    
    # Test classifier setup
    print("\nClassifier Status:")
    if hasattr(agent, 'classifier_model') and hasattr(agent, 'classification_head'):
        print(f"  Classifier Model: Loaded successfully")
        print(f"  Classification Head: Loaded successfully")
        print(f"  Classifier Threshold: {agent.classifier_threshold}")
    else:
        print("  Warning: Classifier not fully loaded!")
        if not hasattr(agent, 'classifier_model'):
            print("  - Missing classifier_model")
        if not hasattr(agent, 'classification_head'):
            print("  - Missing classification_head")
    
    # Test with sample messages
    test_messages = [
        "What is the purpose of the semantic subscription system?",
        "Hello, nice to meet you.",
        "Can you explain how agents work in this framework?",
        "Today is a sunny day.",
        "I need to know how to configure my agent properly."
    ]
    
    for i, test_message in enumerate(test_messages):
        print(f"\nTest {i+1}: '{test_message}'")
        
        # Test interest calculation
        from semsubscription.vector_db.database import Message
        message = Message(content=test_message)
        interest_score = agent.calculate_interest(message)
        
        print(f"Interest Score: {interest_score:.4f} (Threshold: {agent.similarity_threshold} for similarity, {agent.classifier_threshold} for classifier)")
        print(f"Agent would {'process' if interest_score >= max(agent.similarity_threshold, agent.classifier_threshold) else 'ignore'} this message")
        
        # If interested, test processing
        if interest_score >= max(agent.similarity_threshold, agent.classifier_threshold):
            result = agent.process_message(message)
            print("Processing Result:")
            print(json.dumps(result, indent=2))
            
    print("\nAgent testing complete.")

