#!/usr/bin/env python

"""
Test_agent14 Agent Implementation

test
"""

import json
import logging
import os
import re
import uuid
from typing import Dict, Any, Optional

# Import requests for API access
try:
    import requests
except ImportError:
    pass

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
                    
                    # Method 1: Try using direct SentenceTransformer and classification head if available
                    if hasattr(self, 'embedding_model') and hasattr(self, 'classification_head'):
                        try:
                            import torch
                            # Get embedding using SentenceTransformer
                            embedding = self.embedding_model.encode([content], convert_to_tensor=True)
                            # Classify using classification head
                            with torch.no_grad():
                                output = self.classification_head(embedding)
                                score = torch.sigmoid(output).item()
                            logging.info(f"Direct classification score: {score}")
                            return score
                        except Exception as e:
                            logging.error(f"Error using direct classification: {e}")
                            # Continue to next method
                    
                    # Method 2: Use the interest model if it exists
                    if hasattr(self, 'interest_model') and self.interest_model:
                        try:
                            # Use the interest model's built-in calculation method
                            interest_score = self.interest_model.calculate_similarity(content)
                            logging.info(f"Interest model score: {interest_score}")
                            return interest_score
                        except Exception as e:
                            logging.error(f"Error using interest model: {e}")
                            # Continue to fallback implementation
                    
                    # Fallback implementation - keyword matching
                    # Use either the predefined keywords or a default set
                    keywords = getattr(self, 'keywords', [
                        "question", "answer", "why", "how", "what", "when", "who", "where",
                        "explain", "tell", "describe", "help", "information"
                    ])
                    
                    # Count keyword matches
                    matches = sum(1 for keyword in keywords if keyword.lower() in content.lower())
                    
                    # Calculate interest score based on keyword density
                    if matches > 0:
                        # At least one keyword match - express interest
                        interest_score = min(0.5 + (matches * 0.1), 1.0)  # Scale with matches, cap at 1.0
                    else:
                        # No keywords match, still provide minimal interest
                        interest_score = 0.65  # Default interest level for Q&A agent
                    
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

logger = logging.getLogger(__name__)

# Primary class name without Agent suffix (modern style)
class Test_agent14(BaseAgent):
    """
    Agent that test
    """
    
    def __init__(self, agent_id=None, name=None, description=None, similarity_threshold=0.7, **kwargs):
        """
        Initialize the agent with its parameters and setup the classifier
        
        Args:
            agent_id: Optional unique identifier for the agent
            name: Optional name for the agent (defaults to class name)
            description: Optional description of the agent
            similarity_threshold: Threshold for similarity-based interest determination
        """
        # Set default name if not provided
        name = name or "Test_agent14 Agent"
        description = description or "test"
        
        # Call parent constructor
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            similarity_threshold=similarity_threshold,
            **kwargs
        )
        
        # Set classifier threshold (since BaseAgent may not have use_classifier parameter)
        self.classifier_threshold = 0.5  # Lower threshold for testing
        
        logger.info(f"{name} agent initialized")
    
    def setup_interest_model(self):
        """
        Set up the agent's interest model, which determines what messages it processes
        This is called automatically during initialization
        """
        # Check for fine-tuned model directory
        # Look for two different types of fine-tuned models:
        # 1. An embedding model (for SentenceTransformer)
        # 2. An interest model (numpy saved file)
        
        # Try multiple possible model directories (for different environments)
        model_dirs = [
            os.path.join(os.path.dirname(__file__), "fine_tuned_model"),  # Standard path
            "/app/fine_tuned_model",  # Docker container path
            os.path.join(os.getcwd(), "fine_tuned_model")  # Current working directory
        ]
        
        # Find the first valid model directory
        model_dir = None
        for directory in model_dirs:
            if os.path.exists(directory) and os.path.isdir(directory):
                model_dir = directory
                break
        
        if not model_dir:
            logger.warning("Could not find fine_tuned_model directory in any of the expected locations")
            super().setup_interest_model()
            return
            
        logger.info(f"Looking for fine-tuned models in: {model_dir}")
        
        # Model paths
        embedding_model_path = model_dir  # The embedding model would be in the directory itself
        interest_model_path = os.path.join(model_dir, "interest_model.npz")  # The interest vectors
        classification_head_path = os.path.join(model_dir, "classification_head.pt")  # The classification head
        
        # List the model directory for debugging
        files = os.listdir(model_dir)
        logger.info(f"Fine-tuned model directory contains: {files}")
        
        # Configure the fallback keywords for interest calculation
        self.keywords = [
            "question", "answer", "why", "how", "what", "when", "who", "where",
            "explain", "tell", "describe", "help", "information"
        ]
        
        # First try to use a direct approach with SentenceTransformers
        model_loaded = False
        
        try:
            # Try to import SentenceTransformers directly (should be installed in container)
            import torch
            from sentence_transformers import SentenceTransformer
            
            # Check for config.json which indicates a valid model directory
            if os.path.exists(os.path.join(model_dir, "config.json")):
                try:
                    # Load the model directly
                    logger.info("Attempting to load SentenceTransformer model directly")
                    self.embedding_model = SentenceTransformer(model_dir)
                    
                    # Try to load classification head if it exists
                    if os.path.exists(classification_head_path):
                        try:
                            logger.info(f"Loading classification head from {classification_head_path}")
                            self.classification_head = torch.load(classification_head_path, map_location=torch.device('cpu'))
                            self.classification_head.eval()  # Set to evaluation mode
                            model_loaded = True
                            logger.info("Successfully loaded SentenceTransformer model and classification head")
                        except Exception as e:
                            logger.error(f"Error loading classification head: {e}")
                    else:
                        # No classification head, but we can still use the embedding model
                        model_loaded = True
                        logger.info("Successfully loaded SentenceTransformer model (no classification head)")
                except Exception as e:
                    logger.error(f"Error loading SentenceTransformer model directly: {e}")
        except ImportError:
            logger.warning("SentenceTransformer not available, will try alternative methods")
            
        # If the direct approach failed, try the standard methods
        if not model_loaded:
            try:
                # Try standard import patterns
                try:
                    # First try importing from semsubscription
                    from semsubscription.vector_db.embedding import EmbeddingEngine, InterestModel
                    logger.info("Successfully imported from semsubscription modules")
                except ImportError:
                    # Try absolute imports for containerized environments
                    try:
                        from interest_model import CustomInterestModel as InterestModel
                        from embedding_engine import EmbeddingEngine
                        logger.info("Successfully imported using absolute imports")
                    except ImportError:
                        # Last resort - try relative imports if we're in a package
                        try:
                            import sys
                            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                            from agent_base import InterestModel, EmbeddingEngine
                            logger.info("Successfully imported by adding parent directory to path")
                        except Exception as e:
                            logger.error(f"All import methods failed: {e}")
                            # Fall back to standard setup
                            super().setup_interest_model()
                            return
                
                # Create the interest model based on available files
                if os.path.exists(interest_model_path):
                    logger.info(f"Found pre-calculated interest model: {interest_model_path}")
                    # Create a default embedding engine and then load the saved interest vectors
                    embedding_engine = EmbeddingEngine()  # Using default model
                    self.interest_model = InterestModel(embedding_engine=embedding_engine, model_path=interest_model_path)
                    logger.info(f"Successfully loaded pre-calculated interest model")
                    model_loaded = True
                # If we have a custom embedding model, load it
                elif os.path.exists(os.path.join(embedding_model_path, "config.json")):
                    logger.info(f"Found custom sentence transformer model at: {embedding_model_path}")
                    embedding_engine = EmbeddingEngine(model_name=embedding_model_path)
                    logger.info(f"Successfully loaded custom embedding model")
                    self.interest_model = InterestModel(embedding_engine=embedding_engine)
                    model_loaded = True
            except Exception as e:
                logger.error(f"Error in standard model loading approach: {e}")
        
        # Final fallback if nothing worked        
        if not model_loaded:
            logger.warning(f"No valid fine-tuned model could be loaded from {model_dir}")
            # Fall back to standard setup
            super().setup_interest_model()
            return
                    
        # Set threshold for the model if we have one
        if hasattr(self, 'interest_model') and self.interest_model:
            self.interest_model.threshold = self.similarity_threshold
            
            # Add our keywords to the model
            if hasattr(self.interest_model, 'keywords'):
                self.interest_model.keywords.extend(self.keywords)
        super().setup_interest_model()
        
        # Add domain-specific customizations to the default model
        # For example, to add keywords that should always be of interest:
        # self.interest_model.keywords.extend([
        #     "specific_keyword",
        #     "another_keyword"
        # ])
    
    # Import necessary modules for LLM access
    try:
        # First try to import from semsubscription if available (for local development)
        from semsubscription.llm.completion import get_completion
        from semsubscription.memory.memories import search_memories, create_memory
    except ImportError:
        # Fallback for containerized environments - use API endpoints
        pass
    
    def get_relevant_memories(self, query, limit=3):
        """
        Retrieve relevant memories for context based on the query
        Memory operations are optional - agent functions without memory
        
        Args:
            query: The query to search memories for
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of relevant memories or empty list if no matches or error occurs
        """
        # Skip memory operations if not needed - just basic safety check
        if not query or not isinstance(query, str) or len(query.strip()) < 3:
            return []
            
        # Keep track if we've logged memory errors already to avoid log spam
        memory_error_logged = False
            
        try:
            # Try to use direct memory API if available
            try:
                memories = search_memories(
                    query=query,
                    limit=limit,
                    min_similarity=0.7
                )
                return memories
            except (NameError, AttributeError):
                # Fall back to API endpoint for containerized environments
                try:
                    # Only proceed if requests is available
                    if 'requests' not in globals() and 'requests' not in locals():
                        if not memory_error_logged:
                            logger.info("Memory search skipped: requests module not available")
                            memory_error_logged = True
                        return []
                        
                    core_api_url = os.environ.get("CORE_API_URL", "http://host.docker.internal:8888")
                    
                    response = requests.post(
                        f"{core_api_url}/api/memories/search",
                        json={
                            "query": query,
                            "limit": limit,
                            "min_similarity": 0.7
                        },
                        timeout=3  # Add timeout to prevent hanging
                    )
                    
                    if response.status_code == 200:
                        return response.json()
                    else:
                        if not memory_error_logged:
                            logger.info(f"Memory search returned status code: {response.status_code}")
                            memory_error_logged = True
                except Exception as e:
                    if not memory_error_logged:
                        logger.info(f"Memory search unavailable: {str(e)}")
                        memory_error_logged = True
        except Exception as e:
            if not memory_error_logged:
                logger.info(f"Memory operations skipped: {str(e)}")
                memory_error_logged = True
                
        # Always continue with the agent's processing
        return []
    
    def store_memory(self, content, tags=[]):
        """
        Store important information as a memory
        Memory operations are optional - agent functions without memory
        
        Args:
            content: The content to store
            tags: List of tags to associate with the memory
            
        Returns:
            Boolean indicating success
        """
        # Skip memory operations if not needed
        if not content or not isinstance(content, str) or len(content.strip()) < 10:
            return False
            
        try:
            # Try to use direct memory API if available
            try:
                memory = create_memory(
                    content=content,
                    metadata={
                        "source": self.name,
                        "priority": "medium"
                    },
                    tags=tags
                )
                logger.info(f"Successfully stored memory with tags: {tags}")
                return True
            except (NameError, AttributeError):
                # Don't try API call if requests isn't available
                if 'requests' not in globals() and 'requests' not in locals():
                    logger.info("Memory storage skipped: requests module not available")
                    return False
                    
                # Fallback to API endpoint for containerized environments
                try:
                    core_api_url = os.environ.get("CORE_API_URL", "http://host.docker.internal:8888")
                    
                    response = requests.post(
                        f"{core_api_url}/api/memories/",
                        json={
                            "content": content,
                            "metadata": {
                                "source": self.name,
                                "priority": "medium"
                            },
                            "tags": tags
                        },
                        timeout=3  # Add timeout to prevent hanging
                    )
                    
                    if response.status_code in [200, 201]:
                        logger.info(f"Successfully stored memory via API with tags: {tags}")
                        return True
                    else:
                        logger.info(f"Memory creation skipped: API returned {response.status_code}")
                        return False
                except Exception as e:
                    logger.info(f"Memory storage unavailable: {str(e)}")
                    return False
        except Exception as e:
            logger.info(f"Memory operations skipped: {str(e)}")
            return False
        
        return False
    
    def get_llm_response(self, messages, model="gpt-4o", temperature=0.7):
        """
        Get a response from the LLM using either direct module or API
        
        Args:
            messages: List of message dictionaries with role and content
            model: The model to use
            temperature: Creativity parameter
            
        Returns:
            The LLM's response text
        """
        try:
            # Try to use direct LLM API if available
            try:
                # For completion-style models
                system_message = next((m["content"] for m in messages if m["role"] == "system"), None)
                user_message = next((m["content"] for m in messages if m["role"] == "user"), None)
                
                if system_message and user_message:
                    prompt = f"{system_message}\n\nUser: {user_message}"
                else:
                    prompt = user_message
                
                response = get_completion(
                    prompt=prompt,
                    model=model,
                    temperature=temperature
                )
                return response
            except NameError:
                # Fallback to API endpoint for containerized environments
                try:
                    core_api_url = os.environ.get("CORE_API_URL", "http://host.docker.internal:8888")
                    
                    response = requests.post(
                        f"{core_api_url}/api/llm/chat",
                        json={
                            "messages": messages,
                            "model": model,
                            "temperature": temperature,
                            "agent_id": self.agent_id
                        },
                        timeout=5  # Slightly longer timeout for LLM calls
                    )
                    
                    if response.status_code == 200:
                        llm_response = response.json()
                        return llm_response.get("text", "")
                    else:
                        logger.info(f"LLM request failed with status code: {response.status_code}")
                        return "I'm currently unable to access my knowledge base. Please try again later."
                except Exception as e:
                    logger.info(f"Error getting LLM response via API: {e}")
                    return f"I encountered an error while processing your request: {str(e)}"
            
        except Exception as e:
            logger.info(f"Error getting LLM response: {e}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    def process_message(self, message) -> Optional[Dict[str, Any]]:
        """
        Process domain-specific queries
        
        Args:
            message: The message to process (dict in containerized version)
            
        Returns:
            Response data
        """
        try:
            # Handle both Message objects and dictionary messages (for container compatibility)
            if hasattr(message, 'content'):
                content = message.content
                message_id = getattr(message, 'id', 'unknown')
            else:
                content = message.get('content', '')
                message_id = message.get('id', 'unknown')
                
            query = content.lower()
            
            # Log the message being processed
            logger.info(f"Processing message {message_id} with content: '{content[:50]}...'")
            logger.info(f"Message successfully received via event bus")
            
            # Handle basic help or greeting queries directly
            if query.strip() in ['help', 'hello', 'hi']:
                return {
                    "agent": self.name,
                    "response": f"Hello! I'm {self.name}, a Q&A agent that can answer your questions using my knowledge. How can I help you?"
                }
                
            # Try to retrieve relevant memories for context, but continue even if it fails
            try:
                memories = self.get_relevant_memories(query)
                memory_context = ""
                if memories:
                    memory_context = "Relevant context:\n"
                    for idx, memory in enumerate(memories[:3]):
                        if isinstance(memory, dict):
                            memory_content = memory.get('content', '')
                        else:
                            memory_content = getattr(memory, 'content', '')
                        
                        if memory_content:
                            memory_context += f"{idx+1}. {memory_content}\n"
            except Exception as e:
                logger.info(f"Skipping memory retrieval due to error: {str(e)}")
                memories = []
                memory_context = ""
            
            # Prepare the prompt for the LLM with retrieved context
            system_prompt = (
                "You are a helpful and knowledgeable assistant that provides accurate, concise answers to questions. "
                "If you don't know the answer to a question, admit that you don't know rather than making up information. "
                "Always cite sources if mentioned in the context provided. "
            )
            
            if memory_context:
                system_prompt += f"\n\nUse the following context to help answer the question, but only if relevant:\n{memory_context}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ]
            
            # Get response from LLM
            llm_response = self.get_llm_response(messages)
            
            # Try to store the response in memory if it seems valuable, but continue even if it fails
            try:
                if len(llm_response) > 50 and not query.strip() in ['help', 'hello', 'hi']:
                    memory_content = f"Question: {content}\nAnswer: {llm_response}"
                    # Extract potential tags from the query for better retrievability
                    tags = ["qa", "answer"]
                    
                    # Add domain-specific tags based on keywords
                    keywords = ["how", "what", "why", "when", "who", "where"]
                    for keyword in keywords:
                        if keyword in query:
                            tags.append(keyword)
                    
                    # Don't let memory errors block the response
                    self.store_memory(memory_content, tags)
            except Exception as e:
                logger.info(f"Skipping memory storage due to error: {str(e)}")
            
            # Instead of returning the response directly, use the new Create Response to Message endpoint
            try:
                # Only proceed if requests is available
                if 'requests' not in globals() and 'requests' not in locals():
                    logger.info("Response submission skipped: requests module not available")
                    # Fall back to returning the response directly
                    return {
                        "agent": self.name,
                        "query_type": "qa_response",
                        "response": llm_response,
                        "message_id": message_id
                    }
                
                # Prepare the response data
                response_data = {
                    "content": llm_response,
                    "metadata": {
                        "agent_id": self.agent_id,
                        "agent": self.name,
                        "response_type": "generated",
                        "query_type": "qa_response"
                    }
                }
                
                # Determine the API endpoint base URL
                # Try several common patterns for containerized environments
                core_api_url = os.environ.get("CORE_API_URL", None)
                
                # If no explicit URL is provided, we'll try several common patterns
                if not core_api_url:
                    # List of potential base URLs to try, in order of priority
                    potential_base_urls = [
                        # Docker internal service name (most common in containerized environments)
                        "http://api:8888",
                        # Core service name
                        "http://core-api:8888",
                        # Application name
                        "http://semsubscription-api:8888",
                        # Host machine from container (Windows/Mac)
                        "http://host.docker.internal:8888",
                        # Host machine from container (Linux)
                        "http://172.17.0.1:8888",
                        # Localhost (for non-containerized testing)
                        "http://localhost:8888"
                    ]
                    
                    # Log that we're using auto-detection
                    logger.info(f"No CORE_API_URL environment variable found, will try common containerized patterns")
                    
                    # We'll try these URLs later in the request logic
                    core_api_url = potential_base_urls[0]  # Start with the first one
                
                # Try different URLs and endpoint formats until one works
                response_created = False
                last_error = None
                last_status_code = None
                
                # Get potential base URLs if we're auto-detecting
                potential_urls = [core_api_url]  # Start with the configured or default URL
                if os.environ.get("CORE_API_URL", None) is None:
                    # If we're auto-detecting, use our list of potential URLs
                    potential_urls = [
                        "http://api:8888",
                        "http://core-api:8888",
                        "http://semsubscription-api:8888",
                        "http://host.docker.internal:8888",
                        "http://172.17.0.1:8888",
                        "http://localhost:8888"
                    ]
                
                # Define endpoint patterns to try
                endpoint_patterns = [
                    "/api/messages/{message_id}/responses",  # With /api prefix
                    "/messages/{message_id}/responses"      # Without /api prefix
                ]
                
                # Try each combination of base URL and endpoint pattern
                for base_url in potential_urls:
                    for endpoint_pattern in endpoint_patterns:
                        try:
                            # Construct the full URL
                            full_url = f"{base_url}{endpoint_pattern}".format(message_id=message_id)
                            
                            logger.info(f"Attempting to create response using URL: {full_url}")
                            
                            # Make the request
                            api_response = requests.post(
                                full_url,
                                json=response_data,
                                timeout=3  # Shorter timeout since we're trying multiple endpoints
                            )
                            
                            # Store the result
                            last_status_code = api_response.status_code
                            
                            # If successful, break out of both loops
                            if api_response.status_code in [200, 201]:
                                logger.info(f"Successfully created response with URL: {full_url}")
                                response_created = True
                                break  # Exit the endpoint pattern loop
                        except Exception as e:
                            last_error = str(e)
                            continue  # Try the next endpoint pattern
                    
                    # If we successfully created a response, break out of the base URL loop too
                    if response_created:
                        break
                
                # If we couldn't create a response, log the details
                if not response_created:
                    if last_status_code:
                        logger.info(f"Failed to create response message: tried all URL patterns, last status code: {last_status_code}")
                    elif last_error:
                        logger.info(f"Failed to create response message: tried all URL patterns, last error: {last_error}")
                    else:
                        logger.info("Failed to create response message: tried all URL patterns, no response received")
                
                if response_created:
                    # Get the created response message
                    response_message = api_response.json()
                    response_id = response_message.get('id', 'unknown')
                    
                    # Return the response information
                    return {
                        "agent": self.name,
                        "query_type": "qa_response",
                        "response": llm_response,
                        "message_id": message_id,
                        "response_id": response_id,
                        "response_created": True
                    }
                else:
                    # Fall back to returning the response directly
                    return {
                        "agent": self.name,
                        "query_type": "qa_response",
                        "response": llm_response,
                        "message_id": message_id,
                        "response_created": False
                    }
                    
            except Exception as e:
                logger.info(f"Error creating response message: {str(e)}")
                # Fall back to returning the response directly
                return {
                    "agent": self.name,
                    "query_type": "qa_response",
                    "response": llm_response,
                    "message_id": message_id,
                    "error": str(e)
                }
            
        except Exception as e:
            logger.error(f"Error in Test_agent14 Agent processing: {e}")
            return {
                "agent": self.name,
                "error": str(e),
                "query": content if 'content' in locals() else "unknown query"
            }


# Define the class with Agent suffix for backwards compatibility
# This prevents import errors in the container
class Test_agent14Agent(Test_agent14):
    """Legacy class name with Agent suffix"""
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
        "Your test query specific to this agent's domain",
        "A query that should probably not be handled by this agent",
        "Another domain-specific query to test routing"
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

