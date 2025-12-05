# Soros Investment Advisor 💼

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

An AI-powered chatbot that answers questions about George Soros's investment philosophy and strategies, featuring:
- **RAG-powered Chat**: Using Groq API (Llama 3.3 70B) + Pinecone vector database
- **Pairs Trading Analysis**: Statistical arbitrage with cointegration testing
- **1,011 Q&A pairs**: Comprehensive knowledge base about Soros's strategies

## 🚀 Features

### 💬 Chat Interface
- Ask questions about George Soros's investment philosophy
- Get accurate answers from 1,011 curated Q&A pairs
- Powered by Groq API with RAG (Retrieval-Augmented Generation)
- Real-time confidence scores and source references

### 📊 Pairs Trading Analysis
- Cointegration testing (Engle-Granger)
- Spread visualization with trading thresholds
- Automated long/short signal generation
- P&L backtesting with performance metrics
- Real-time data from Yahoo Finance

### ℹ️ Educational Content
- Learn about pairs trading strategy
- Understand George Soros's approach
- Interactive tutorials and resources

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Groq API (Llama 3.3 70B)
- **Vector DB**: Pinecone
- **Data**: Yahoo Finance API
- **Analysis**: statsmodels, pandas, numpy
- **Visualization**: matplotlib

## 📦 Installation

### Prerequisites
- Python 3.12+
- Groq API key ([Get one here](https://console.groq.com/keys))
- Pinecone API key ([Get one here](https://www.pinecone.io/))

### Local Setup

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

Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

5. **Run the app**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 🌐 Deployment on Streamlit Cloud

### Step 1: Push to GitHub
Ensure your code is pushed to GitHub (secrets in `.env` are not committed).

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your repository: `mhitjain/soros-chatbot`
4. Main file path: `app.py`
5. Click "Advanced settings"
6. Add secrets:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   PINECONE_API_KEY = "your_pinecone_api_key_here"
   ```
7. Click "Deploy!"

Your app will be live at: `https://your-app-name.streamlit.app`

## 📊 Usage Examples

### Chat Examples
- "What is George Soros's investment philosophy?"
- "How did Soros break the Bank of England?"
- "What is reflexivity theory?"
- "What was Soros's father's name?"

### Pairs Trading Examples
- **Tech pairs**: GOOGL + META
- **Banks**: JPM + BAC
- **Energy**: XOM + CVX
- **ETFs**: SPY + IWM (default)

## 📁 Project Structure

```
soros-chatbot/
├── app.py                          # Main Streamlit app
├── chatbot_v2.py                   # RAG chatbot engine
├── vector_store.py                 # Pinecone vector DB
├── data/
│   └── Soros_sample.xlsx          # 1,011 Q&A pairs
├── requirements.txt                # Python dependencies
├── .streamlit/
│   └── config.toml                # Streamlit config
└── README.md                       # This file
```

## 🔑 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Groq API key for LLM | ✅ Yes |
| `PINECONE_API_KEY` | Pinecone API key for vector DB | ✅ Yes |

## 📝 Features Roadmap

- [x] RAG-powered chatbot
- [x] Pairs trading analysis
- [x] Cointegration testing
- [x] P&L visualization
- [ ] More statistical tests (Johansen)
- [ ] Custom transformer model training
- [ ] Portfolio optimization
- [ ] Real-time alerts

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- George Soros's investment wisdom
- Groq for LLM API
- Pinecone for vector database
- Streamlit for the amazing framework

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Disclaimer**: This chatbot is for educational purposes only. Not financial advice.
