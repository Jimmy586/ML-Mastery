import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Masking, LayerNormalization, MultiHeadAttention, Dropout, Layer
from tensorflow.keras.models import Model
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

class LookAheadMask(Layer):
    """
    Custom Keras Layer for creating a look-ahead mask for the decoder.
    """
    def call(self, x):
        size = tf.shape(x)[1]
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

def encoder_layer(x, num_heads, dff, d_model, dropout_rate=0.1):
    """
    Build an encoder layer.
    """
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
    attn_output = attention(x, x)
    attn_output = Dropout(dropout_rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)

    ffn = tf.keras.Sequential([
        Dense(dff, activation='relu'),
        Dense(d_model)
    ])
    ffn_output = ffn(out1)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    return out2

def decoder_layer(x, enc_output, num_heads, dff, d_model, dropout_rate=0.1):
    """
    Build a decoder layer.
    """
    attention1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
    attn_output1 = attention1(x, x, attention_mask=LookAheadMask()(x))
    attn_output1 = Dropout(dropout_rate)(attn_output1)
    out1 = LayerNormalization(epsilon=1e-6)(attn_output1 + x)

    attention2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
    attn_output2 = attention2(out1, enc_output)
    attn_output2 = Dropout(dropout_rate)(attn_output2)
    out2 = LayerNormalization(epsilon=1e-6)(attn_output2 + out1)

    ffn = tf.keras.Sequential([
        Dense(dff, activation='relu'),
        Dense(d_model)
    ])
    ffn_output = ffn(out2)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    out3 = LayerNormalization(epsilon=1e-6)(ffn_output + out2)
    return out3

def transformer_model(input_vocab_size, target_vocab_size, max_length_input, max_length_target, d_model, num_heads, dff, num_layers):
    """
    Build the Transformer model.
    """
    encoder_inputs = Input(shape=(max_length_input,))
    encoder_embedding = Embedding(input_vocab_size, d_model)(encoder_inputs)
    encoder_embedding *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    encoder_embedding = Masking(mask_value=0.0)(encoder_embedding)
    encoder_outputs = encoder_embedding
    for i in range(num_layers):
        encoder_outputs = encoder_layer(encoder_outputs, num_heads, dff, d_model, dropout_rate=0.1)

    decoder_inputs = Input(shape=(max_length_target,))
    decoder_embedding = Embedding(target_vocab_size, d_model)(decoder_inputs)
    decoder_embedding *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    decoder_embedding = Masking(mask_value=0.0)(decoder_embedding)
    decoder_outputs = decoder_embedding
    for i in range(num_layers):
        decoder_outputs = decoder_layer(decoder_outputs, encoder_outputs, num_heads, dff, d_model, dropout_rate=0.1)

    decoder_outputs = Dense(target_vocab_size, activation='softmax')(decoder_outputs)
    return Model([encoder_inputs, decoder_inputs], decoder_outputs)

def calculate_bleu(reference, candidate):
    """
    Calculate BLEU score for a single reference and candidate pair.
    """
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    return sentence_bleu([reference_tokens], candidate_tokens)
