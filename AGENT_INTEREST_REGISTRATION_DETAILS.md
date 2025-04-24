# Agent Interest Registration: Detailed Technical Documentation

This document provides a comprehensive technical overview of the agent interest registration system in the Semantic Subscription Service, detailing all components, data flows, and implementation details.

## Overview

Agent interest registration is the core mechanism that determines which agents should process which messages. It uses a multi-tier approach that combines machine learning classifiers, vector similarity, and keyword matching to provide high-precision routing with multiple fallback mechanisms.

## System Components

### Core Components

| Component | Description | Location |
|-----------|-------------|----------|
| Agent Base Class | Provides core interest determination logic | `semsubscription/agents/base.py` |
| Interest Model | Calculates vector similarity between messages and agent interests | `semsubscription/vector_db/embedding.py` |
| Classification Head | Neural network for binary relevance classification | `semsubscription/agents/base.py` |
| Embedding Engine | Converts text to vector representations | `semsubscription/vector_db/embedding.py` |
| Agent Manager | Manages agent lifecycle and registration | `semsubscription/core/agent_manager.py` |
| Message Bus | Routes messages to interested agents | `semsubscription/core/message_bus.py` |
| Agent Package Loader | Loads agent packages with their classifiers | `semsubscription/core/agent_package_loader.py` |

## Folder Structure

```
/agents/
  /{agent_name}/
    agent.py             # Agent implementation
    config.yaml          # Agent configuration including thresholds
    examples.jsonl       # Training data for agent interest
    model_weights.pkl    # Serialized interest vectors
    fine_tuned_model/    # Fine-tuned classifier model
      config.json        # Model configuration
      classification_head.pt  # PyTorch classification head
      sentence_bert/     # Modified SentenceBERT model

/semsubscription/
  /agents/
    base.py             # Base agent implementation with interest logic
  /vector_db/
    embedding.py        # Embedding engine and interest model
  /core/
    agent_manager.py    # Agent registration and management
    message_bus.py      # Message routing to interested agents
    agent_package_loader.py  # Load agent packages with models
```

## Data Flow

1. **Agent Initialization**
   - Agent packages are discovered in the `/agents/` directory
   - Agent configurations loaded from `config.yaml`
   - Classifiers and interest models loaded from `fine_tuned_model/` and `model_weights.pkl`
   - Agents register with the agent manager

2. **Message Processing**
   - New message arrives via API or CLI
   - Message is embedded and stored in vector database
   - Message bus notifies all agents of new message
   - Each agent determines interest using the multi-tier approach
   - Interested agents register to process the message
   - Only interested agents process the message

## Key Functions and Methods

### Agent Base Class (`semsubscription/agents/base.py`)

| Method | Inputs | Outputs | Description |
|--------|--------|---------|-------------|
| `setup_classifier()` | None | None | Loads fine-tuned classifier from agent package |
| `setup_interest_model()` | None | None | Loads or trains interest model from examples |
| `calculate_interest()` | `Message` object | Float (0.0-1.0) | Calculates interest score for a message |
| `is_interested()` | `Message` object | Boolean | Determines if agent is interested in a message |

### Interest Model (`semsubscription/vector_db/embedding.py`)

| Method | Inputs | Outputs | Description |
|--------|--------|---------|-------------|
| `train()` | List of example texts | None | Trains interest vectors from examples |
| `calculate_similarity()` | Text string | Float (0.0-1.0) | Calculates similarity between text and interest |

### Classification Head (`semsubscription/agents/base.py`)

| Class | Methods | Description |
|-------|---------|-------------|
| `InterestClassificationHead` | `forward()` | Neural network that takes an embedding and returns interest score |

### Agent Package Loader (`semsubscription/core/agent_package_loader.py`)

| Function | Inputs | Outputs | Description |
|----------|--------|---------|-------------|
| `load_agent_package()` | Directory path | Agent instance | Loads agent from package with all models |

## Multi-tier Interest Determination Process

Agents determine their interest in messages through this sequence:

1. **Fine-tuned Classifier** (Primary Method)
   - Inputs: Message content
   - Process: 
     - Message text → SentenceTransformer → Embedding → Classification Head → Score
     - If score ≥ `classifier_threshold` → Interested
     - If score < `classifier_threshold/2` → Not interested
     - If borderline → Try next method
   - Location: `Agent.is_interested()` method

2. **Vector Similarity** (Secondary Method)
   - Inputs: Message content
   - Process:
     - Message text → Embedding → Cosine similarity with interest vectors
     - If similarity ≥ `similarity_threshold` → Interested
     - If similarity < `similarity_threshold` → Try next method
   - Location: `InterestModel.calculate_similarity()` method

3. **Keyword Matching** (Final Fallback)
   - Inputs: Message content
   - Process:
     - Check if domain-specific keywords exist in message
     - If keyword match → Interested
     - If no match → Not interested
   - Location: End of `Agent.is_interested()` method

## Implementation Details

### Classifier Setup

The classifier is initialized in `run.py` through these steps:

```python
# Load calendar agent
calendar_agent = load_agent_package(calendar_agent_path)
if calendar_agent:
    # Initialize the classifier
    calendar_agent.setup_interest_model()
    calendar_agent.setup_classifier()
    calendar_agent.use_classifier = True
    agent_manager.register_agent(calendar_agent)
```

The `setup_classifier()` method:
1. Locates the agent directory in `/agents/{agent_type}`
2. Loads the fine-tuned model from `fine_tuned_model/`
3. Creates a classification head for binary relevance prediction
4. Sets `use_classifier = True` if successful

### Interest Model Setup

The interest model is set up as a fallback mechanism:

```python
def setup_interest_model(self):
    # Try to load pre-trained model from model_weights.pkl
    model_path = os.path.join(agent_dir, 'model_weights.pkl')
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            # Load the interest vectors
            self.interest_model.interest_vectors = model_data['interest_vectors']
            # Use the trained threshold
            self.similarity_threshold = model_data['threshold']
```

### Fine-tuned Classifier Details

The classifier consists of:
1. A SentenceTransformer base model that converts text to embeddings
2. A small classification head (neural network) that takes embeddings and outputs relevance scores

The classification head is defined as:

```python
class InterestClassificationHead(nn.Module):
    def __init__(self, input_dim, dropout_prob=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, embeddings):
        x = self.dropout(embeddings)
        x = self.linear(x)
        return self.sigmoid(x)
```

## Configuration

### Agent Configuration (`config.yaml`)

```yaml
# Agent identification
name: "Web Search Agent"
description: "Handles information retrieval requests"

# Interest determination configuration
use_classifier: true
classifier_threshold: 0.5
similarity_threshold: 0.7

# Runtime configuration
polling_interval: 5.0
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|--------|
| `EMBEDDING_MODEL` | Model name for SentenceTransformer | `all-MiniLM-L6-v2` |
| `VECTOR_DB_BACKEND` | Vector database implementation | `faiss` |

## Training Process

Agent classifiers are trained using:

1. Examples from `examples.jsonl` in the agent package
2. The `train_embedding_classifier.py` script
3. Training outputs are saved to the `fine_tuned_model/` directory

### Training Script Flow

1. Load positive examples from agent's `examples.jsonl`
2. Generate negative examples from other agents' examples
3. Use SentenceTransformer as base model
4. Add and train classification head
5. Save fine-tuned model to agent's directory

## Possible Issues and Solutions

### Dimension Mismatch

**Problem**: Vector dimension error when models expect different embedding sizes

**Solution**: Ensure consistent embedding models throughout the system

```python
# Check embedding dimensions
embedding_dim = embedding_engine.get_embedding_dimension()
classifier_dim = classifier_model.get_sentence_embedding_dimension()

assert embedding_dim == classifier_dim, "Dimension mismatch between embedding engine and classifier"
```

### Classifier Loading Failures

**Problem**: Classifier fails to load, falls back to vector similarity

**Solution**: Ensure proper error handling and logging

```python
try:
    # Load classifier
except Exception as e:
    logger.warning(f"Classifier loading failed: {e}, falling back to similarity method")
```

## Performance Considerations

1. **Caching**: SentenceTransformer models are cached to reduce load times
2. **GPU Acceleration**: Classifier uses GPU when available
3. **Batching**: Messages are processed in batches where possible
4. **Multi-tier Approach**: Allows for quick rejection of irrelevant messages

## Extending the System

To add a new agent with interest classification:

1. Create a new agent package in `/agents/{agent_name}/`
2. Create `examples.jsonl` with positive examples
3. Run the training script to create a fine-tuned classifier
4. Implement the agent class extending the base Agent
5. Configure thresholds in `config.yaml`
