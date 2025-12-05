# 💼 Soros Investment Advisor Chatbot

An AI-powered chatbot that answers questions about George Soros's investment philosophy and strategies using RAG (Retrieval-Augmented Generation) with Groq API and Pinecone vector database.

## 🌟 Features

- **RAG-Powered Responses**: Combines semantic search with LLM reformulation for accurate, contextual answers
- **Vector Database**: 1,011 Q&A pairs indexed in Pinecone for fast semantic search
- **Groq API**: Uses Llama 3.3 70B for intelligent answer reformulation
- **Transparency**: Shows confidence scores, source Q&A pairs, and original answers
- **Clean UI**: Modern Streamlit interface with dark theme
- **Entity Verification**: Prevents confusion between similar names (e.g., George vs Paul Soros)

## 🏗️ Architecture

```
User Question
    ↓
Vector Search (Pinecone)
    ↓
Retrieve Top Matches
    ↓
LLM Reformulation (Groq)
    ↓
Conversational Answer
```

## 📁 Project Structure

```
soros-chatbot/
├── app.py                          # Main Streamlit application
├── chatbot_v2.py                   # RAG chatbot with Groq API
├── vector_store.py                 # Pinecone vector database operations
├── transformer_engine.py           # Custom transformer backend (future)
├── pairs_trading.py                # Pairs trading analysis (future)
├── create_enhanced_dataset.py      # Dataset enhancement script
├── Train_Soros_Transformer_Colab.ipynb  # Training notebook (Google Colab)
├── data/
│   ├── Soros_sample.xlsx          # Q&A dataset (1,011 pairs)
│   └── soros_enhanced_dataset.xlsx # Enhanced meta Q&A
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Groq API Key (get from [console.groq.com](https://console.groq.com))
- Pinecone API Key (get from [pinecone.io](https://pinecone.io))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mhitjain/soros-chatbot.git
   cd soros-chatbot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your-groq-api-key-here
   PINECONE_API_KEY=your-pinecone-api-key-here
   ```

5. **Index the database** (first time only)
   
   Run this script to upload Q&A pairs to Pinecone:
   ```python
   from chatbot_v2 import SorosAdvisorV2
   
   chatbot = SorosAdvisorV2()
   result = chatbot.train('data/Soros_sample.xlsx')
   print(f"Indexed {result['total_pairs']} Q&A pairs")
   ```

6. **Run the application**
   ```bash
   streamlit run app.py
   ```

7. **Open in browser**
   
   Navigate to `http://localhost:8501`

## 💡 Usage

### Chat Interface

Ask questions about George Soros's investment strategies:

- "What is Soros's investment philosophy?"
- "How does Soros manage risk?"
- "What is reflexivity theory?"
- "When did George Soros die?" (Tests entity verification)

### Response Details

Click "📊 Response Details" to see:
- **Confidence Score**: How well the question matched (0-100%)
- **Method**: `groq_reformulation` or `direct_answer`
- **Source Q&A Pairs**: Top 3 matching questions from database
- **Original Answer**: Exact answer before LLM reformulation

## 🔧 Configuration

### Chatbot Settings

In `chatbot_v2.py`:

```python
CONFIDENCE_THRESHOLD = 0.60  # Minimum confidence to answer (0.0-1.0)
MAX_QUERY_LENGTH = 500       # Maximum question length
```

### Vector Search Settings

In `vector_store.py`:

```python
dimension = 384              # Embedding dimension (all-MiniLM-L6-v2)
metric = "cosine"            # Similarity metric
top_k = 3                    # Number of results to retrieve
```

## 📊 Dataset

The knowledge base contains **1,011 Q&A pairs** covering:

- Personal life and biography
- Investment philosophy (reflexivity theory)
- Trading strategies and tactics
- Risk management principles
- Market analysis approaches
- Historical trades and decisions
- Philanthropy and Open Society Foundations

### Dataset Format

Excel file with columns:
- **Questions**: User questions
- **Answers**: Expert answers
- **Label**: Category tags (optional)

## 🎯 Future Enhancements

### 1. Custom Transformer Backend
- Train TensorFlow transformer on Q&A dataset
- Offline mode without API dependency
- Located in `transformer_engine.py`

### 2. Pairs Trading Analysis
- Stock pair selection
- Cointegration testing
- Spread calculation and backtesting
- Located in `pairs_trading.py`

### 3. Enhanced Dataset
- Meta questions (greetings, capabilities)
- Out-of-scope handling
- Located in `data/soros_enhanced_dataset.xlsx`

## 🧪 Testing

Test the chatbot programmatically:

```python
from chatbot_v2 import SorosAdvisorV2

chatbot = SorosAdvisorV2()
result = chatbot.chat("What is reflexivity theory?")

print(f"Answer: {result['response']}")
print(f"Confidence: {result['confidence']*100:.1f}%")
print(f"Method: {result['method']}")
```

## 🐛 Known Issues & Solutions

### Entity Disambiguation
- **Issue**: Similar names (e.g., George Soros vs Paul Soros) can confuse vector search
- **Solution**: Entity verification in Groq prompt checks if matched question is about the same person
- If entities differ, system responds: "I don't have that information in my database"

### Hallucination Prevention
- **Issue**: LLM might add information not in the database
- **Solution**: 
  - Source answers stored exactly as in dataset
  - Strict LLM prompt: use ONLY provided information
  - Original answer shown for verification

### Incomplete Indexing
- **Issue**: Some Q&A pairs not indexed initially
- **Solution**: Re-indexing clears old vectors and uploads all 1,011 pairs
- Verify: Vector count should match dataset size

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Dataset based on publicly available information about George Soros
- Powered by [Groq](https://groq.com) and [Pinecone](https://pinecone.io)
- Built with [Streamlit](https://streamlit.io)

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This chatbot is for educational purposes only. Investment advice should always be sought from qualified financial advisors.
