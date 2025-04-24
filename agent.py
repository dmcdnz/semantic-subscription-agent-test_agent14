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
import asyncio
import time
from typing import Dict, Any, Optional, List, Union

# Set up logger first to avoid reference before assignment in fallback functions
logger = logging.getLogger(__name__)

# Note: LLMAgent now uses the EnhancedAgent as its base through the alias
from semsubscription.agents.llm_agent import LLMAgent
from semsubscription.vector_db.database import Message
# Memory system implementations
try:
    from semsubscription.memory.vector_memory import get_memory_system
    logger.info("Successfully imported memory system from semsubscription module")
except ImportError:
    logger.warning("Could not import memory system directly, will use API client fallback")
    
    # Define a API-based memory client for containerized environments
    class CoreMemoryClient:
        """Client for interacting with the memory system via the Core API"""
        
        def __init__(self, core_api_url=None):
            """Initialize with API URL"""
            self.core_api_url = core_api_url or os.environ.get("CORE_API_URL", "http://host.docker.internal:8888")
            logger.info(f"Initialized CoreMemoryClient with API URL: {self.core_api_url}")
        
        def search_similar(self, query, k=5, threshold=0.65):
            """Search for similar memory items"""
            try:
                import requests
                url = f"{self.core_api_url}/api/memory/search"
                data = {
                    "query": query,
                    "limit": k,
                    "min_similarity": threshold
                }
                response = requests.post(url, json=data, timeout=10)
                
                if response.status_code == 200:
                    # Convert API response to memory items format
                    results = response.json()
                    memory_items = []
                    for item in results:
                        memory_items.append(type('MemoryItem', (), {
                            'content': item.get('content', ''),
                            'tags': item.get('tags', []),
                            'title': item.get('metadata', {}).get('title', ''),
                            'similarity': item.get('similarity', 0.0)
                        }))
                    return memory_items
                else:
                    logger.error(f"Memory search failed with status {response.status_code}: {response.text}")
                    return []
            except Exception as e:
                logger.error(f"Error searching memory: {e}")
                return []
    
    # Global client instance
    _memory_client = None

    # Fallback implementation using the API client
    def get_memory_system():
        global _memory_client
        if _memory_client is None:
            try:
                _memory_client = CoreMemoryClient()
            except Exception as e:
                logger.error(f"Failed to initialize CoreMemoryClient: {e}")
                return None
        return _memory_client

class QuestionAnsweringAgentAgent(LLMAgent):
    """
    Agent that detects and answers questions using an LLM
    
    This agent specializes in identifying questions in natural language and providing
    accurate, helpful answers using an LLM. It uses both pattern matching and
    vector similarity to determine if a message contains a question.
    """
    
    def setup_interest_model(self):
        """
        Configure the agent's interest model with domain-specific knowledge
        """
        # Don't call super().setup_interest_model() as we're creating a new custom interest model
        
        # Get the absolute path to the fine-tuned model
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "fine_tuned_model"))
        logger.info(f"Using fine-tuned model from: {model_path}")
        
        # Check if the model directory exists and contains essential files
        missing_files = []
        essential_files = ['model.safetensors', 'classification_head.pt', 'config.json', 'tokenizer.json']
        for file in essential_files:
            if not os.path.exists(os.path.join(model_path, file)):
                missing_files.append(file)
                
        if missing_files:
            logger.warning(f"Fine-tuned model directory is missing essential files: {', '.join(missing_files)}")
        else:
            logger.info("Fine-tuned model directory contains all essential files")
            
        try:
            # Create embedding engine with the local fine-tuned model
            from semsubscription.vector_db.embedding import EmbeddingEngine, InterestModel
            
            # Create a custom embedding engine pointing to our fine-tuned model
            embedding_engine = EmbeddingEngine(model_name=model_path)
            logger.info(f"Successfully loaded fine-tuned model with dimension: {embedding_engine.get_dimension()}")
            
            # Create interest model with the custom embedding engine
            self.interest_model = InterestModel(embedding_engine=embedding_engine)
            
            # Lower the threshold to catch more potential questions
            self.similarity_threshold = 0.5  # Lower this from default (likely 0.67)
            self.interest_model.threshold = self.similarity_threshold
            logger.info(f"Set similarity threshold to {self.similarity_threshold}")
            
            # Add question-related keywords for backup matching
            self.interest_model.keywords = [
                'question', 'answer', 'how', 'what', 'when', 'where', 'why',
                'who', 'which', 'whose', 'explain', 'tell me', 'describe',
                'help me understand', '?'
            ]
            
            # Get question patterns from config
            self.question_patterns = []
            if 'question_patterns' in self.config.get('custom', {}):
                pattern_strings = self.config['custom']['question_patterns']
                self.question_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in pattern_strings]
            
            # Add default patterns if none are defined
            if not self.question_patterns:
                self.question_patterns = [
                    re.compile(r'^(?:what|who|where|when|why|how|is|are|can|could|would|will|should)', re.IGNORECASE),
                    re.compile(r'.*\?$', re.IGNORECASE),
                    re.compile(r'\bi (?:need|want|would like) to know\b', re.IGNORECASE),  # Information seeking
                    re.compile(r'\bexplain(?:\s+to\s+me)?\b', re.IGNORECASE),  # Explanation requests
                    re.compile(r'\bdo you know\b.*?', re.IGNORECASE),  # Knowledge queries
                    re.compile(r'\bi\'m (?:curious|wondering|asking|interested)\b', re.IGNORECASE)  # Curiosity indicators
                ]
                
            # Configure other agent settings from config
            custom_config = self.config.get('custom', {})
            self.max_answer_length = custom_config.get('max_answer_length', 1500)
            self.answer_format = custom_config.get('answer_format', 'markdown')
            
            # Train with example questions if available
            examples = [
                "What is the capital of France?",
                "How does photosynthesis work?",
                "Can you explain quantum computing?",
                "Who wrote Pride and Prejudice?",
                "When was the first computer invented?",
                "Why is the sky blue?",
                "Tell me about the history of chocolate",
                "What's the difference between machine learning and AI?",
                "How far is the moon from earth?",
                "What are the main challenges of climate change?"
            ]
            
            # Force retraining of interest model to ensure vectors are stored properly
            logger.info(f"Training question answering agent with {len(examples)} examples using fine-tuned model")
            self.interest_model.interest_vectors = []  # Clear any existing vectors
            if examples and len(examples) > 0:
                self.interest_model.train(examples, method="average")  # Use simpler averaging method
                
        except Exception as e:
            logger.error(f"Error setting up fine-tuned model: {e}")
            logger.warning("Falling back to basic question detection without the fine-tuned model")
            
            # Create a minimal interest model
            from semsubscription.vector_db.embedding import InterestModel
            try:
                # Try to create a basic interest model
                self.interest_model = InterestModel()
                
                # Add question-related keywords for backup matching
                self.interest_model.keywords = [
                    'question', 'answer', 'how', 'what', 'when', 'where', 'why',
                    'who', 'which', 'whose', 'explain', 'tell me', 'describe',
                    'help me understand', '?'
                ]
            except Exception as e2:
                logger.error(f"Error creating basic interest model: {e2}")
                # Continue with patterns only
                pass
                
            # Always set up the question patterns even if interest model fails
            self.question_patterns = [
                re.compile(r'^(?:what|who|where|when|why|how|is|are|can|could|would|will|should)', re.IGNORECASE),
                re.compile(r'.*\?$', re.IGNORECASE),
                re.compile(r'\bi (?:need|want|would like) to know\b', re.IGNORECASE),  # Information seeking
                re.compile(r'\bexplain(?:\s+to\s+me)?\b', re.IGNORECASE),  # Explanation requests
                re.compile(r'\bdo you know\b.*?', re.IGNORECASE),  # Knowledge queries
                re.compile(r'\bi\'m (?:curious|wondering|asking|interested)\b', re.IGNORECASE)  # Curiosity indicators
            ]
            
            # Set a lower threshold to catch more potential questions
            self.similarity_threshold = 0.4  # Even lower for fallback mode
    
    # CRITICAL METHOD: This directly overrides the BaseAgent's calculate_interest method
    def calculate_interest(self, message) -> float:
        """
        Calculate interest level for a message.
        
        This method DIRECTLY overrides BaseAgent.calculate_interest to ensure proper
        interest calculation for question detection, prioritizing the fine-tuned model.
        
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
        
        # Structured content handling
        if isinstance(content, dict) and 'message' in content:
            content = content['message']
        elif isinstance(content, dict) and 'content' in content:
            content = content['content']
        
        # Convert to string if necessary
        if not isinstance(content, str):
            try:
                content = str(content)
            except Exception as e:
                logger.warning(f"Cannot convert message content to string: {e}")
                content = ""
                
        logger.info(f"Calculating interest for message {message_id}")
        logger.info(f"Message content: {content[:100]}...")
        
        # PRIORITY 1: Use the fine-tuned interest model - this is the primary approach
        # The fine-tuned model should be loaded during setup_interest_model
        if hasattr(self, 'interest_model') and self.interest_model:
            try:
                # This is the main calculation using the fine-tuned model
                interest_score = self.interest_model.calculate_similarity(content)
                logger.info(f"Fine-tuned model interest score: {interest_score}")
                
                # If score is above our interest threshold, return it immediately
                if interest_score >= self.similarity_threshold:
                    return interest_score
                    
                # If fine-tuned model gave us a non-zero score but below threshold,
                # we'll keep it as a baseline but still check patterns
                if interest_score > 0:
                    baseline_score = interest_score
                else:
                    # Something's wrong with the model score, set a reasonable baseline
                    baseline_score = 0.4
            except Exception as e:
                logger.error(f"Error using fine-tuned interest model: {e}")
                baseline_score = 0.4  # Set a reasonable baseline for fallback
        else:
            logger.warning("No fine-tuned interest model available")
            baseline_score = 0.4  # Set a reasonable baseline for fallback
            
        # PRIORITY 2: Check for direct question patterns as additional detection layer
        # This can catch questions even if the fine-tuned model fails to recognize them
        if hasattr(self, 'question_patterns'):
            for pattern in self.question_patterns:
                if pattern.search(content):
                    pattern_score = 0.9  # High confidence for direct question matches
                    logger.info(f"Question pattern matched: {pattern.pattern}. Score: {pattern_score}")
                    return pattern_score
        
        # PRIORITY 3: Use keyword matching as final detection layer
        # This catches questions that might be phrased in ways the model or patterns miss
        keywords = []
        
        # Use keywords from the interest model if available
        if hasattr(self, 'interest_model') and hasattr(self.interest_model, 'keywords'):
            keywords = self.interest_model.keywords
        
        # Fallback to default keywords if needed
        if not keywords:
            keywords = [
                # Common question keywords
                'question', 'answer', 'how', 'what', 'when', 'where', 'why',
                'who', 'which', 'whose', 'explain', 'tell me', 'describe',
                'help me understand', '?'
            ]
            
            # Add these to the interest model if available for future use
            if hasattr(self, 'interest_model') and not hasattr(self.interest_model, 'keywords'):
                self.interest_model.keywords = keywords
        
        # Count keyword matches
        matches = 0
        content_lower = content.lower()
        for keyword in keywords:
            if str(keyword).lower() in content_lower:
                matches += 1
                logger.info(f"Keyword match on '{keyword}'")
        
        # Calculate interest score based on keyword density
        if matches > 0:
            # At least one keyword match - express interest
            keyword_score = min(0.5 + (matches * 0.1), 1.0)  # Scale with matches, cap at 1.0
            logger.info(f"Keyword interest calculation: {keyword_score} (based on {matches} keyword matches)")
            return max(keyword_score, baseline_score)  # Take the highest of keyword vs baseline
            
        # FINAL FALLBACK: Return our baseline interest with a log message
        logger.info(f"Using baseline interest score: {baseline_score}")
        return baseline_score  # Return the baseline we established earlier
        
    def is_interested(self, message) -> bool:
        """
        Determine if the message contains a question
        
        Args:
            message: The message to check
            
        Returns:
            True if the message contains a question, False otherwise
        """
        # Use our calculate_interest method and compare to threshold
        interest_score = self.calculate_interest(message)
        interested = interest_score >= self.similarity_threshold
        
        # Log the result
        logger.info(f"Interest score: {interest_score}, threshold: {self.similarity_threshold}, interested: {interested}")
        return interested
        
    async def process_message_async(self, message: Message) -> Optional[Dict[str, Any]]:
        """
        Process questions using LLM and retrieve relevant context
        
        Args:
            message: The message to process
            
        Returns:
            Response data with the answer
        """
        try:
            # Extract the question from the message
            question = message.content.strip()
            
            # Skip if there is no question (shouldn't happen due to interest check)
            if not question:
                return None
                
            start_time = time.time()
            logger.info(f"Processing question: {question[:100]}...")
            
            # Get relevant context from memory system if available
            memory_context = self._retrieve_relevant_context(question)
            
            # If no memory context available and LLM is available, use it directly
            if not memory_context and self.llm:
                result = await super().process_message_async(message)
                result["processing_time"] = round(time.time() - start_time, 2)
                result["source"] = "llm_direct"
                return result
                
            # Prepare enhanced context for the LLM
            system_prompt = self._get_enhanced_system_prompt(memory_context)
            
            # Call the LLM with the enhanced prompt and context
            if self.llm:
                llm_response = await self.llm.complete_chat(
                    system_prompt=system_prompt,
                    messages=[{"role": "user", "content": question}],
                    temperature=self.config.get('llm', {}).get('temperature', 0.7),
                    max_tokens=min(self.max_answer_length, self.config.get('llm', {}).get('max_tokens', 1000))
                )
                
                answer = llm_response.get('content', "I couldn't generate an answer at this time.")
                
                result = {
                    "agent": self.name,
                    "query_type": "question",
                    "question": question,
                    "answer": answer,
                    "format": self.answer_format,
                    "processing_time": round(time.time() - start_time, 2),
                    "source": "llm_with_context" if memory_context else "llm_direct"
                }
                
                if memory_context:
                    result["context_sources"] = len(memory_context)
                    
                return result
            else:
                # Fallback to non-LLM processing
                return self.process_message(message)
                
        except Exception as e:
            logger.error(f"Error in Question Answering Agent async processing: {e}")
            return {
                "agent": self.name,
                "error": str(e),
                "query": message.content
            }
            
    def process_message(self, message: Message) -> Optional[Dict[str, Any]]:
        """
        Fallback for processing without LLM
        
        Args:
            message: The message to process
            
        Returns:
            Response data with a simple answer
        """
        try:
            question = message.content
            
            # Simple question detection
            is_question = False
            for pattern in self.question_patterns:
                if pattern.search(question):
                    is_question = True
                    break
                    
            if not is_question:
                # Not a question, provide a generic response
                return {
                    "agent": self.name,
                    "query_type": "non_question",
                    "message": f"I don't recognize that as a question. Please try rephrasing as a question."
                }
            
            # Simple question answering without LLM
            simple_answers = {
                "what is your name": f"My name is {self.name}.",
                "who are you": f"I am {self.name}, an AI agent designed to answer questions.",
                "what can you do": "I can answer questions on a wide range of topics. Without my LLM connection, my capabilities are limited though.",
                "help": "I'm designed to answer questions. Please ask me something specific.",
            }
            
            # Try to find a simple answer match
            question_lower = question.lower().strip('?!., ')
            for key, answer in simple_answers.items():
                if key in question_lower:
                    return {
                        "agent": self.name,
                        "query_type": "question",
                        "question": question,
                        "answer": answer,
                        "source": "fallback"
                    }
            
            # Generic fallback response for questions
            return {
                "agent": self.name,
                "query_type": "question",
                "question": question,
                "answer": "I'm sorry, I can't provide a detailed answer without my LLM connection. Please try again later.",
                "source": "fallback"
            }
            
        except Exception as e:
            logger.error(f"Error in Question Answering Agent processing: {e}")
            return {
                "agent": self.name,
                "error": str(e),
                "query": message.content
            }


    def _retrieve_relevant_context(self, question: str, max_items: int = 5, threshold: float = 0.65) -> List[Dict[str, Any]]:
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
            memory_system = get_memory_system()
            if not memory_system:
                return []
                
            # Search for similar memory items
            results = memory_system.search_similar(question, k=max_items, threshold=threshold)
            
            # Extract and format the memory items
            context_items = []
            for item in results:
                context_items.append({
                    "content": item.content,
                    "tags": item.tags,
                    "title": item.title,
                    "similarity": item.similarity
                })
                
            logger.debug(f"Retrieved {len(context_items)} context items for question: {question[:50]}...")
            return context_items
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
            
    def _get_enhanced_system_prompt(self, context_items: List[Dict[str, Any]]) -> str:
        """
        Enhance the system prompt with relevant context
        
        Args:
            context_items: List of relevant memory items
            
        Returns:
            Enhanced system prompt
        """
        # Get the base system prompt
        default_prompt = "You are an AI assistant specializing in answering questions. Provide accurate, helpful, and concise answers."
        base_prompt = self.config.get('llm', {}).get('system_prompt', default_prompt)
        
        # If no context, return the base prompt
        if not context_items:
            return base_prompt
            
        # Add context to the prompt
        context_parts = []
        context_parts.append(base_prompt)
        context_parts.append("\n\nYou have access to the following relevant information that may help you answer the question.")
        context_parts.append("Use this information if relevant to the question:\n")
        
        # Add each context item
        for i, item in enumerate(context_items):
            context_parts.append(f"Context {i+1}: {item['content']}\n")
            
        # Add closing instructions
        context_parts.append("\nEnd of context information.\n")
        context_parts.append("If the context doesn't contain information relevant to the question, just answer based on your knowledge.")
        context_parts.append("Do not disclose that you were given any context information.")
        
        # Join all parts into the final prompt
        return '\n'.join(context_parts)


# For standalone testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = QuestionAnsweringAgentAgent()
    print(f"Agent created: {agent.name}")
    
    # Test with a sample question
    test_message = Message(content="What is the capital of France?")
    result = asyncio.run(agent.process_message_async(test_message))
    print(f"Test result: {json.dumps(result, indent=2)}")
