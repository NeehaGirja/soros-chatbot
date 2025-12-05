"""
Soros Custom Transformer Backend
Loads and uses the trained transformer model for inference.
"""

import os
import re
import json
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Dict, Any


class SorosTransformerEngine:
    """Custom Transformer model for George Soros Q&A."""
    
    def __init__(
        self,
        model_path: str = 'soros_transformer_model',
        tokenizer_path: str = 'soros_tokenizer.subwords',
        config_path: str = 'training_config.json'
    ):
        """
        Initialize the Soros Transformer engine.
        
        Args:
            model_path: Path to saved model directory
            tokenizer_path: Path to tokenizer file (.subwords)
            config_path: Path to training configuration JSON
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.config_path = config_path
        
        self.model = None
        self.tokenizer = None
        self.config = None
        self.MODEL_READY = False
        
        print("[TRANSFORMER] Initializing Soros Custom Transformer...")
        self._load_model()
    
    def _load_model(self):
        """Load the trained model, tokenizer, and configuration."""
        try:
            # Check if files exist
            if not os.path.exists(self.model_path):
                print(f"[ERROR] Model not found at '{self.model_path}'")
                print("[INFO] Please train the model first using Train_Soros_Transformer_Colab.ipynb")
                return
            
            if not os.path.exists(self.tokenizer_path):
                print(f"[ERROR] Tokenizer not found at '{self.tokenizer_path}'")
                return
            
            if not os.path.exists(self.config_path):
                print(f"[WARN] Config not found at '{self.config_path}', using defaults")
                self.config = self._default_config()
            else:
                # Load config
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                print(f"[INFO] Loaded config: {self.config['epochs_trained']} epochs, "
                      f"accuracy: {self.config['final_accuracy']:.3f}")
            
            # Load tokenizer
            print("[INFO] Loading tokenizer...")
            self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
                self.tokenizer_path.replace('.subwords', '')
            )
            print(f"[INFO] Tokenizer loaded (vocab size: {self.tokenizer.vocab_size})")
            
            # Load model
            print(f"[INFO] Loading model from '{self.model_path}'...")
            self.model = tf.keras.models.load_model(
                self.model_path,
                custom_objects={
                    'MultiHeadAttentionLayer': MultiHeadAttentionLayer,
                    'PositionalEncoding': PositionalEncoding
                }
            )
            print("[SUCCESS] Model loaded successfully!")
            
            self.MODEL_READY = True
            
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            self.MODEL_READY = False
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration if config file not found."""
        return {
            'max_length': 100,
            'start_token': 8192,
            'end_token': 8193,
            'vocab_size': 8194
        }
    
    def _preprocess_sentence(self, sentence: str) -> str:
        """
        Preprocess text (same as training).
        
        Args:
            sentence: Raw text input
            
        Returns:
            Cleaned and normalized text
        """
        sentence = str(sentence).lower().strip()
        
        # Create space between punctuation
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        
        # Handle contractions
        sentence = re.sub(r"i'm", "i am", sentence)
        sentence = re.sub(r"he's", "he is", sentence)
        sentence = re.sub(r"she's", "she is", sentence)
        sentence = re.sub(r"it's", "it is", sentence)
        sentence = re.sub(r"that's", "that is", sentence)
        sentence = re.sub(r"what's", "what is", sentence)
        sentence = re.sub(r"where's", "where is", sentence)
        sentence = re.sub(r"how's", "how is", sentence)
        sentence = re.sub(r"'ll", " will", sentence)
        sentence = re.sub(r"'ve", " have", sentence)
        sentence = re.sub(r"'re", " are", sentence)
        sentence = re.sub(r"'d", " would", sentence)
        sentence = re.sub(r"won't", "will not", sentence)
        sentence = re.sub(r"can't", "cannot", sentence)
        sentence = re.sub(r"n't", " not", sentence)
        sentence = re.sub(r"n'", "ng", sentence)
        sentence = re.sub(r"'bout", "about", sentence)
        
        # Keep only letters, numbers, and basic punctuation
        sentence = re.sub(r"[^a-zA-Z0-9?.!,]+", " ", sentence)
        sentence = sentence.strip()
        
        return sentence
    
    def _is_soros_related(self, question: str) -> bool:
        """
        Check if question is related to Soros/investing.
        
        Args:
            question: User's question
            
        Returns:
            True if related to Soros/investing, False otherwise
        """
        # Keywords that indicate Soros/investment topic
        soros_keywords = [
            'soros', 'george', 'investment', 'invest', 'market', 'trading', 'trade',
            'reflexivity', 'fund', 'stock', 'portfolio', 'risk', 'hedge', 'finance',
            'bubble', 'currency', 'short', 'long', 'strategy', 'money', 'profit',
            'loss', 'buy', 'sell', 'economy', 'economic', 'bank', 'financial',
            'asset', 'bond', 'equity', 'derivative', 'leverage', 'position',
            'rally', 'crash', 'bull', 'bear', 'trend', 'analysis', 'forecast'
        ]
        
        # Greetings and meta questions (always allow)
        greeting_keywords = [
            'hello', 'hi', 'hey', 'help', 'thank', 'bye', 'goodbye',
            'what can you', 'who are you', 'who created', 'what are you',
            'your name', 'understand', 'explain', 'example', 'how are you',
            'good morning', 'good evening'
        ]
        
        question_lower = question.lower()
        
        # Allow if it's a greeting/meta question
        if any(keyword in question_lower for keyword in greeting_keywords):
            return True
        
        # Allow if Soros/investment related
        if any(keyword in question_lower for keyword in soros_keywords):
            return True
        
        return False
    
    def predict(self, question: str) -> str:
        """
        Generate answer for a given question.
        
        Args:
            question: User's question
            
        Returns:
            Generated answer
        """
        if not self.MODEL_READY:
            return "⚠️ Custom Transformer model not loaded. Please check model files."
        
        # Check if question is in scope
        if not self._is_soros_related(question):
            return ("I'm specialized in George Soros' investment philosophy and strategies. "
                    "Please ask me questions about investing, markets, trading, or Soros' theories and approach.")
        
        try:
            # Preprocess question
            question_clean = self._preprocess_sentence(question)
            
            # Get tokens from config
            start_token = [self.config['start_token']]
            end_token = [self.config['end_token']]
            max_length = self.config.get('max_length', 100)
            
            # Tokenize input
            sentence = tf.expand_dims(
                start_token + self.tokenizer.encode(question_clean) + end_token, 
                axis=0
            )
            
            # Initialize output with start token
            output = tf.expand_dims(start_token, 0)
            
            # Generate tokens one by one
            for i in range(max_length):
                predictions = self.model(
                    inputs=[sentence, output], 
                    training=False
                )
                
                # Get last predicted token
                predictions = predictions[:, -1:, :]
                predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
                
                # Stop if end token is generated
                if tf.equal(predicted_id, end_token[0]):
                    break
                
                # Append predicted token to output
                output = tf.concat([output, predicted_id], axis=-1)
            
            # Decode output tokens to text
            prediction = tf.squeeze(output, axis=0)
            predicted_sentence = self.tokenizer.decode(
                [i for i in prediction.numpy() if i < self.tokenizer.vocab_size]
            )
            
            return predicted_sentence
            
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return f"⚠️ Error generating answer: {str(e)}"
    
    def get_answer(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Get answer in a format compatible with RAG engine.
        
        Args:
            question: User's question
            **kwargs: Additional parameters (ignored for transformer)
            
        Returns:
            Dictionary with answer and metadata
        """
        answer = self.predict(question)
        
        return {
            'answer': answer,
            'retrieved': [],  # No retrieval for transformer
            'confidence': 0.0,  # Not applicable
            'model': 'Custom Transformer',
            'source': 'Trained on Soros Q&A dataset'
        }


# Custom layers needed for model loading
class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    """Multi-head attention layer for transformer."""
    
    def __init__(self, num_heads, d_model, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        
        self.query_dense = tf.keras.layers.Dense(d_model)
        self.key_dense = tf.keras.layers.Dense(d_model)
        self.value_dense = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)
    
    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads, "d_model": self.d_model})
        return config
    
    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])
    
    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]
        
        query = self.split_heads(self.query_dense(query), batch_size)
        key = self.split_heads(self.key_dense(key), batch_size)
        value = self.split_heads(self.value_dense(value), batch_size)
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)
        
        if mask is not None:
            logits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(logits, axis=-1)
        scaled_attention = tf.matmul(attention_weights, value)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        
        return self.dense(concat_attention)


class PositionalEncoding(tf.keras.layers.Layer):
    """Positional encoding layer for transformer."""
    
    def __init__(self, position, d_model, **kwargs):
        super().__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)
    
    def get_config(self):
        config = super().get_config()
        config.update({"position": self.position, "d_model": self.d_model})
        return config
    
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            tf.cast(tf.range(position)[:, tf.newaxis], tf.float32),
            tf.cast(tf.range(d_model)[tf.newaxis, :], tf.float32),
            tf.cast(d_model, tf.float32)
        )
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        return pos_encoding[tf.newaxis, ...]
    
    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000.0, (2 * (i // 2)) / d_model)
        return position * angles
    
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


# Test function
if __name__ == "__main__":
    print("Testing Soros Custom Transformer Engine...")
    print("="*60)
    
    # Initialize engine
    engine = SorosTransformerEngine()
    
    if engine.MODEL_READY:
        # Test questions
        test_questions = [
            "What is George Soros' investment philosophy?",
            "How does reflexivity work in markets?",
            "What are Soros' views on risk management?"
        ]
        
        print("\n🧪 Testing model with sample questions:\n")
        for i, question in enumerate(test_questions, 1):
            print(f"{i}. Q: {question}")
            result = engine.get_answer(question)
            print(f"   A: {result['answer']}\n")
    else:
        print("\n⚠️ Model not ready. Please train the model first.")
        print("   Use: Train_Soros_Transformer_Colab.ipynb on Google Colab")
