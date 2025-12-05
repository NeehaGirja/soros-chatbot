# Deployment Checklist for Streamlit Cloud

## ✅ Pre-Deployment Checklist

### 1. Repository Setup
- [x] Code pushed to GitHub
- [x] `.env` file in `.gitignore` (secrets not committed)
- [x] `requirements.txt` updated
- [x] `.streamlit/config.toml` created
- [x] `README.md` updated

### 2. Files to Check
```bash
# Verify these files exist:
app.py                    # Main Streamlit app
chatbot_v2.py            # RAG engine
vector_store.py          # Pinecone integration
requirements.txt         # Dependencies
data/Soros_sample.xlsx   # Q&A dataset
.streamlit/config.toml   # Streamlit config
```

### 3. Environment Variables Required
- `GROQ_API_KEY` - Your Groq API key
- `PINECONE_API_KEY` - Your Pinecone API key

---

## 🚀 Deployment Steps

### Step 1: Go to Streamlit Cloud
1. Visit: https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"

### Step 2: Configure App
Fill in the deployment form:
- **Repository**: `mhitjain/soros-chatbot`
- **Branch**: `main`
- **Main file path**: `app.py`

### Step 3: Add Secrets
Click "Advanced settings" → "Secrets"

Paste your API keys in this format:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
PINECONE_API_KEY = "your_pinecone_api_key_here"
```

**Important**: Replace with your actual API keys from:
- Groq: https://console.groq.com/keys
- Pinecone: https://www.pinecone.io/

### Step 4: Deploy
1. Click "Deploy!"
2. Wait 2-5 minutes for deployment
3. Your app will be live at: `https://your-app-name.streamlit.app`

---

## 🔍 Testing After Deployment

### Test Checklist
1. **Chat Tab**
   - [ ] Ask: "What is George Soros's investment philosophy?"
   - [ ] Verify response shows confidence score
   - [ ] Check that sources are displayed

2. **Pairs Trading Tab**
   - [ ] Keep default stocks (SPY + IWM)
   - [ ] Click "Analyze Pair"
   - [ ] Verify cointegration test appears
   - [ ] Check spread chart renders
   - [ ] Verify P&L chart displays
   - [ ] Check trading activity table shows data

3. **About Tab**
   - [ ] Verify all text displays correctly
   - [ ] Check links work

---

## 🐛 Troubleshooting

### Common Issues

**1. "ModuleNotFoundError"**
- Solution: Ensure `requirements.txt` has all dependencies
- Check: `pip freeze > requirements.txt` locally

**2. "API Key Error"**
- Solution: Verify secrets are correctly pasted in Streamlit Cloud
- Format: `KEY_NAME = "value"` (no extra spaces)

**3. "Data file not found"**
- Solution: Ensure `data/Soros_sample.xlsx` is committed to GitHub
- Check: File is not in `.gitignore`

**4. Charts not rendering**
- Solution: Matplotlib backend issue - already handled in code
- Streamlit Cloud uses `Agg` backend automatically

**5. Slow loading**
- Normal: First load takes 30-60 seconds (downloading dependencies)
- Subsequent loads are faster (cached)

---

## 📊 Post-Deployment

### Monitor App
1. Check logs in Streamlit Cloud dashboard
2. Monitor usage metrics
3. Watch for errors in real-time

### Update App
```bash
# Make changes locally
git add .
git commit -m "Update feature"
git push origin main

# Streamlit Cloud auto-deploys on push!
```

### App URL
Once deployed, share your app:
- **Public URL**: `https://your-app-name.streamlit.app`
- **Custom domain**: Available in Streamlit Cloud settings (paid plan)

---

## 🎯 Next Steps

After successful deployment:
1. Test all features thoroughly
2. Share the link!
3. Monitor for any errors
4. Consider adding analytics
5. Update README with live app link

---

## 📞 Support

- **Streamlit Docs**: https://docs.streamlit.io/streamlit-community-cloud
- **Community Forum**: https://discuss.streamlit.io
- **GitHub Issues**: Report bugs in your repository

---

**Good luck with your deployment! 🚀**
