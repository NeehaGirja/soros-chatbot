"""
Soros Investment Chatbot - Streamlit Interface with Dual Backend
Backends: 
1. Groq API + Pinecone RAG (Online)
2. Custom Transformer (Offline - when model is trained)
"""
import streamlit as st
from chatbot_v2 import SorosAdvisorV2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Try to import transformer engine (optional)
try:
    from transformer_engine import SorosTransformerEngine
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    SorosTransformerEngine = None

# Page configuration
st.set_page_config(
    page_title="Soros Investment Advisor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize chatbots
@st.cache_resource
def get_groq_chatbot():
    return SorosAdvisorV2()

@st.cache_resource
def get_transformer_chatbot():
    if not TRANSFORMER_AVAILABLE:
        return None
    model_path = "models/soros_transformer_model"
    if os.path.exists(model_path):
        return SorosTransformerEngine(model_path=model_path)
    return None

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    
    h1 {
        color: #FAFAFA;
        text-align: center;
        padding: 1rem 0;
    }
    
    .stChatMessage {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .metric-container {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Backend selection
    st.markdown("### 🤖 Backend Selection")
    
    backend_option = st.radio(
        "Choose Backend:",
        ["Groq API + RAG", "Custom Transformer"],
        help="Groq API uses cloud-based LLM. Custom Transformer runs locally (requires trained model)."
    )
    
    # Show backend status
    if backend_option == "Groq API + RAG":
        st.success("**Active:** Groq API + Pinecone")
        st.caption("🤖 Model: Llama 3.3 70B")
        st.caption("📊 Vector DB: 1,011 Q&A pairs")
        st.caption("☁️ Status: Online")
    else:
        if not TRANSFORMER_AVAILABLE:
            st.error("**Status:** TensorFlow Not Installed")
            st.caption("⚠️ Install TensorFlow: `pip install tensorflow`")
            st.info("Falling back to Groq API...")
            backend_option = "Groq API + RAG"
        else:
            transformer_bot = get_transformer_chatbot()
            if transformer_bot:
                st.success("**Active:** Custom Transformer")
                st.caption("🧠 Model: TensorFlow Transformer")
                st.caption("💾 Status: Loaded")
            else:
                st.error("**Status:** Model Not Found")
                st.caption("⚠️ Train model first using Colab notebook")
                st.info("Falling back to Groq API...")
                backend_option = "Groq API + RAG"  # Fallback
    
    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    This chatbot answers questions about George Soros's investment philosophy 
    and strategies.
    
    **Two Backend Options:**
    - **Groq API**: Cloud-based, always up-to-date
    - **Custom Transformer**: Offline, privacy-focused
    """)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Create tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Pairs Trading", "ℹ️ About"])

# ============================================================================
# TAB 1: CHAT
# ============================================================================
with tab1:
    st.title("💼 Soros Investment Advisor")
    st.caption("Ask me anything about George Soros's investment strategies and philosophy")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history in tab1
with tab1:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show response details if available
            if message["role"] == "assistant" and "details" in message and message["details"]:
                details = message["details"]
                with st.expander("📊 Response Details", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Confidence Score", f"{details['confidence']*100:.1f}%")
                    with col2:
                        st.info(f"**Method:** {details['method']}")
                    
                    if details.get('sources'):
                        st.markdown("**📚 Top Source Q&A Pairs:**")
                        for i, source in enumerate(details['sources'][:3], 1):
                            st.text(f"Source {i}:\n{source}\n")
                    
                    if details.get('exact_answer'):
                        st.markdown("**📝 Original Answer from Database:**")
                        st.info(details['exact_answer'])

# Chat input - OUTSIDE tabs so it stays at bottom
if prompt := st.chat_input("Ask about Soros's investment strategies..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get response from chatbot
    # Select backend
    if backend_option == "Groq API + RAG":
        chatbot = get_groq_chatbot()
        response_data = chatbot.chat(prompt)
        
        # Extract response
        answer = response_data.get('response', 'Sorry, I could not generate a response.')
        confidence = response_data.get('confidence', 0.0)
        in_db = response_data.get('in_db', False)
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "details": {
                "confidence": confidence,
                "method": response_data.get('method', 'unknown'),
                "sources": response_data.get('sources', []),
                "exact_answer": response_data.get('exact_answer')
            } if in_db else None
        })
        
    else:  # Custom Transformer
        transformer_bot = get_transformer_chatbot()
        if transformer_bot:
            answer = transformer_bot.get_answer(prompt)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "details": None
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Custom transformer model not found. Please train the model first.",
                "details": None
            })
    
    # Rerun to display the new messages
    st.rerun()

# ============================================================================
# TAB 2: PAIRS TRADING
# ============================================================================
with tab2:
    st.title("📊 Pairs Trading Analysis")
    st.caption("Analyze cointegrated stock pairs for statistical arbitrage opportunities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Stock Pair Selection")
        
        stock1 = st.text_input("Stock 1 (Ticker)", value="SPY", help="Enter first stock ticker symbol")
        stock2 = st.text_input("Stock 2 (Ticker)", value="IWM", help="Enter second stock ticker symbol")
        
        # Default dates
        default_start = datetime.now() - timedelta(days=365)
        default_end = datetime.now()
        
        start_date = st.date_input("Start Date", value=default_start)
        end_date = st.date_input("End Date", value=default_end)
        
        # Threshold parameter
        z_threshold = st.slider("Z-Score Threshold", min_value=0.5, max_value=3.0, value=1.18, step=0.1, 
                               help="Higher values = fewer trades, lower risk")
        
        analyze_button = st.button("🔍 Analyze Pair", type="primary")
    
    with col2:
        st.markdown("### 📈 What is Pairs Trading?")
        st.markdown("""
        Pairs trading is a market-neutral strategy that matches a long position 
        with a short position in a pair of highly correlated instruments.
        
        **Key Concepts:**
        - **Cointegration**: Statistical relationship where two stocks move together
        - **Spread**: Difference between normalized prices
        - **Mean Reversion**: Assumption that spread returns to historical mean
        
        **Soros's Approach:**
        George Soros used statistical arbitrage as part of his broader toolkit, 
        combining quantitative signals with macroeconomic analysis.
        """)
        
        st.markdown("### 📚 Resources")
        st.markdown("""
        - [Introduction to Pairs Trading](https://www.investopedia.com/terms/p/pairstrade.asp)
        - [Cointegration Explained](https://www.investopedia.com/terms/c/cointegration.asp)
        - Reference: `pairs_trading.py` in this repository
        """)
    
    # Perform analysis when button is clicked
    if analyze_button:
        try:
            import yfinance as yf
            import statsmodels.api as sm
            from statsmodels.tsa.stattools import coint
            
            with st.spinner(f"📥 Downloading data for {stock1} and {stock2}..."):
                # Download data
                data = yf.download([stock1, stock2], start=start_date, end=end_date, progress=False)['Close']
                
                if data.empty or len(data) < 30:
                    st.error("❌ Insufficient data. Please check ticker symbols and date range.")
                else:
                    # Display data summary
                    st.success(f"✅ Downloaded {len(data)} trading days of data")
                    
                    # ========== COINTEGRATION TEST ==========
                    st.markdown("---")
                    st.markdown("### 🧪 Cointegration Test")
                    
                    # Clean data and align series
                    X = data[stock1].dropna()
                    y = data[stock2].dropna()
                    common_index = X.index.intersection(y.index)
                    X = X.loc[common_index]
                    y = y.loc[common_index]
                    
                    # Perform cointegration test
                    score, p_value, _ = coint(X, y)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Test Statistic", f"{score:.4f}")
                    with col2:
                        st.metric("P-Value", f"{p_value:.4f}")
                    with col3:
                        if p_value < 0.05:
                            st.success("✅ Cointegrated")
                        else:
                            st.warning("⚠️ Not Cointegrated")
                    
                    st.caption("""
                    **Interpretation**: 
                    - P-value < 0.05: Series are cointegrated (good for pairs trading)
                    - P-value ≥ 0.05: Series are NOT cointegrated (risky for pairs trading)
                    """)
                    
                    # ========== SPREAD CALCULATION ==========
                    st.markdown("---")
                    st.markdown("### 📉 Spread Analysis")
                    
                    # Use cleaned aligned data
                    data_clean = data.loc[common_index].copy()
                    X_clean = data_clean[stock1]
                    y_clean = data_clean[stock2]
                    
                    # Calculate spread using OLS regression
                    X_const = sm.add_constant(X_clean)
                    model = sm.OLS(y_clean, X_const).fit()
                    data_clean['spread'] = y_clean - model.predict(X_const)
                    
                    # Calculate statistics
                    mean_spread = data_clean['spread'].mean()
                    std_spread = data_clean['spread'].std()
                    upper_threshold = mean_spread + z_threshold * std_spread
                    lower_threshold = mean_spread - z_threshold * std_spread
                    
                    # Plot spread
                    fig1, ax1 = plt.subplots(figsize=(12, 6))
                    ax1.plot(data_clean.index, data_clean['spread'], label='Spread', color='blue', linewidth=1.5)
                    ax1.axhline(mean_spread, color='black', linestyle='--', label='Mean', linewidth=1)
                    ax1.axhline(upper_threshold, color='red', linestyle='--', label=f'Upper ({z_threshold}σ)', linewidth=1)
                    ax1.axhline(lower_threshold, color='green', linestyle='--', label=f'Lower ({z_threshold}σ)', linewidth=1)
                    ax1.fill_between(data_clean.index, lower_threshold, upper_threshold, alpha=0.1, color='gray')
                    ax1.legend(loc='best')
                    ax1.set_title(f'Pairs Trading Strategy: Spread between {stock1} and {stock2}', fontsize=14, fontweight='bold')
                    ax1.set_xlabel('Date', fontsize=12)
                    ax1.set_ylabel('Spread', fontsize=12)
                    ax1.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig1)
                    
                    # Spread statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean Spread", f"{mean_spread:.4f}")
                    with col2:
                        st.metric("Std Dev", f"{std_spread:.4f}")
                    with col3:
                        st.metric("Upper Threshold", f"{upper_threshold:.4f}")
                    with col4:
                        st.metric("Lower Threshold", f"{lower_threshold:.4f}")
                    
                    # ========== TRADING SIGNALS ==========
                    st.markdown("---")
                    st.markdown("### 🎯 Trading Signals & P&L")
                    
                    # Generate signals
                    data_clean['long'] = data_clean['spread'] < lower_threshold
                    data_clean['short'] = data_clean['spread'] > upper_threshold
                    
                    # Calculate returns
                    data_clean['returns_stock1'] = data_clean[stock1].pct_change()
                    data_clean['returns_stock2'] = data_clean[stock2].pct_change()
                    
                    # Calculate P&L
                    data_clean['pnl'] = np.where(data_clean['long'], data_clean['returns_stock2'] - data_clean['returns_stock1'], 0) + \
                                  np.where(data_clean['short'], data_clean['returns_stock1'] - data_clean['returns_stock2'], 0)
                    
                    data_clean['cumulative_pnl'] = data_clean['pnl'].cumsum()
                    
                    # Plot cumulative P&L
                    fig2, ax2 = plt.subplots(figsize=(12, 6))
                    ax2.plot(data_clean.index, data_clean['cumulative_pnl'], label='Cumulative P&L', color='purple', linewidth=2)
                    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
                    ax2.fill_between(data_clean.index, 0, data_clean['cumulative_pnl'], 
                                    where=(data_clean['cumulative_pnl'] >= 0), alpha=0.3, color='green', label='Profit')
                    ax2.fill_between(data_clean.index, 0, data_clean['cumulative_pnl'], 
                                    where=(data_clean['cumulative_pnl'] < 0), alpha=0.3, color='red', label='Loss')
                    ax2.legend(loc='best')
                    ax2.set_title(f'Pairs Trading Strategy: Cumulative Profit and Loss', fontsize=14, fontweight='bold')
                    ax2.set_xlabel('Date', fontsize=12)
                    ax2.set_ylabel('Cumulative P&L', fontsize=12)
                    ax2.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig2)
                    
                    # Performance metrics
                    total_pnl = data_clean['cumulative_pnl'].iloc[-1]
                    num_long_signals = data_clean['long'].sum()
                    num_short_signals = data_clean['short'].sum()
                    total_signals = num_long_signals + num_short_signals
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total P&L", f"{total_pnl:.4f}", 
                                 delta="Profit" if total_pnl > 0 else "Loss",
                                 delta_color="normal" if total_pnl > 0 else "inverse")
                    with col2:
                        st.metric("Long Signals", num_long_signals)
                    with col3:
                        st.metric("Short Signals", num_short_signals)
                    
                    # Recent trading activity
                    st.markdown("#### 📋 Recent Trading Activity (Last 30 Days)")
                    recent_data = data_clean[['spread', 'long', 'short', 'pnl', 'cumulative_pnl']].tail(30)
                    st.dataframe(recent_data.style.format({
                        'spread': '{:.4f}',
                        'pnl': '{:.6f}',
                        'cumulative_pnl': '{:.4f}'
                    }), use_container_width=True)
                    
        except ImportError as e:
            st.error("❌ Required libraries not installed. Please run: `pip install yfinance statsmodels`")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.caption("Please check ticker symbols and date range.")

# ============================================================================
# TAB 3: ABOUT
# ============================================================================
with tab3:
    st.title("ℹ️ About This Chatbot")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 💼 Soros Investment Advisor
        
        An AI-powered chatbot that answers questions about George Soros's investment 
        philosophy, strategies, and trading approaches using advanced RAG 
        (Retrieval-Augmented Generation) technology.
        
        ### 🎯 Purpose
        
        This chatbot is designed to help investors and students of finance learn from 
        one of history's most successful investors. It provides insights into:
        
        - **Investment Philosophy**: Reflexivity theory and market psychology
        - **Trading Strategies**: Macro trading, currency speculation, short selling
        - **Risk Management**: Position sizing, stop losses, portfolio hedging
        - **Market Analysis**: Fundamental vs technical analysis approaches
        - **Historical Trades**: Famous trades like breaking the Bank of England
        
        ### 🏗️ Architecture
        
        **Dual Backend System:**
        
        1. **Groq API + Pinecone RAG** (Default)
           - Cloud-based Llama 3.3 70B model
           - Vector database with 1,011 Q&A pairs
           - Real-time semantic search
           - Conversational reformulation
        
        2. **Custom Transformer** (Optional)
           - TensorFlow-based transformer model
           - Trained on Soros Q&A dataset
           - Fully offline operation
           - Privacy-focused
        
        ### 📊 Dataset
        
        The knowledge base contains **1,011 curated Q&A pairs** covering:
        - Personal biography and history
        - Investment philosophy and principles
        - Trading strategies and tactics
        - Risk management approaches
        - Market analysis methods
        - Philanthropy and Open Society work
        
        ### 🔒 Transparency Features
        
        Unlike black-box AI systems, this chatbot provides:
        - **Confidence Scores**: How well your question matched the database
        - **Source Citations**: Exact Q&A pairs used to generate answers
        - **Original Answers**: Pre-reformulation answers from the database
        - **Method Tracking**: Whether answer used LLM reformulation or direct retrieval
        
        ### ⚠️ Limitations
        
        - Information based on publicly available data up to 2024
        - Not financial advice - consult qualified professionals
        - Best for educational and research purposes
        - May not reflect latest market developments
        
        ### 🛠️ Technology Stack
        
        - **Frontend**: Streamlit
        - **Vector DB**: Pinecone (cloud)
        - **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
        - **LLM**: Groq (Llama 3.3 70B)
        - **ML Framework**: TensorFlow 2.15 (custom transformer)
        
        ### 📝 License & Attribution
        
        This project is for educational purposes only. The dataset is compiled from 
        publicly available information about George Soros and his investment approaches.
        
        **Disclaimer**: This chatbot does not provide personalized investment advice. 
        Always do your own research and consult with licensed financial advisors 
        before making investment decisions.
        """)
    
    with col2:
        st.markdown("### 📈 Key Metrics")
        st.metric("Q&A Pairs", "1,011")
        st.metric("Vector Dimension", "384")
        st.metric("Confidence Threshold", "60%")
        
        st.markdown("---")
        
        st.markdown("### 🎓 Learn More")
        st.markdown("""
        **Recommended Reading:**
        - *The Alchemy of Finance* by George Soros
        - *Soros on Soros* by George Soros
        - *The New Paradigm for Financial Markets* by George Soros
        
        **External Resources:**
        - [Soros Fund Management](https://www.soros.com/)
        - [Open Society Foundations](https://www.opensocietyfoundations.org/)
        """)
        
        st.markdown("---")
        
        st.markdown("### 💻 Source Code")
        st.markdown("""
        This project is open source!
        
        [View on GitHub](https://github.com/mhitjain/soros-chatbot)
        """)
        
        st.markdown("---")
        
        st.markdown("### 📧 Contact")
        st.markdown("""
        Questions or feedback?
        
        Open an issue on GitHub or contact the maintainer.
        """)
