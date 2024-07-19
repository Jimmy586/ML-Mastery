import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Masking, LayerNormalization, Dropout, MultiHeadAttention, Layer
from tensorflow.keras.models import Model
import numpy as np
import pickle

class LookAheadMask(Layer):
    def call(self, x):
        size = tf.shape(x)[1]
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

def encoder_layer(x, num_heads, dff, d_model, dropout_rate=0.1):
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
    print("Shape of x (decoder input):", x.shape)
    print("Shape of enc_output (encoder output):", enc_output.shape)

    attention1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
    attn_output1 = attention1(x, x, attention_mask=LookAheadMask()(x))
    attn_output1 = Dropout(dropout_rate)(attn_output1)
    out1 = LayerNormalization(epsilon=1e-6)(attn_output1 + x)

    print("Shape of out1 after first attention:", out1.shape)

    enc_output = tf.keras.layers.Reshape((out1.shape[1], d_model))(enc_output)
    print("Shape of aligned enc_output:", enc_output.shape)

    attention2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
    attn_output2 = attention2(out1, enc_output)
    attn_output2 = Dropout(dropout_rate)(attn_output2)
    out2 = LayerNormalization(epsilon=1e-6)(attn_output2 + out1)

    print("Shape of out2 after second attention:", out2.shape)

    ffn = tf.keras.Sequential([
        Dense(dff, activation='relu'),
        Dense(d_model)
    ])
    ffn_output = ffn(out2)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    out3 = LayerNormalization(epsilon=1e-6)(ffn_output + out2)

    print("Shape of out3 after feed-forward:", out3.shape)

    return out3

def transformer_model(input_vocab_size, target_vocab_size, max_length_input, max_length_target, d_model, num_heads, dff, num_layers):
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

def load_tokenizers():
    with open('assembly_tokenizer.pkl', 'rb') as f:
        assembly_tokenizer, max_length_assembly = pickle.load(f)
    with open('c_tokenizer.pkl', 'rb') as f:
        c_tokenizer, max_length_c = pickle.load(f)
    return assembly_tokenizer, c_tokenizer, max_length_assembly, max_length_c

def decode_sequence(input_seq, encoder_model, decoder_model, c_tokenizer, max_length_assembly, max_length_c):
    input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=max_length_assembly)
    enc_output = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = c_tokenizer.word_index['<start>']
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens = decoder_model.predict([target_seq, enc_output])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = c_tokenizer.index_word.get(sampled_token_index, '')

        if (sampled_char == '<end>' or len(decoded_sentence.split()) > max_length_c):
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_char
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
    return decoded_sentence.strip()

def main():
    assembly_tokenizer, c_tokenizer, max_length_assembly, max_length_c = load_tokenizers()

    d_model = 64
    num_heads = 4
    dff = 64
    num_layers = 2

    assembly_vocab_size = len(assembly_tokenizer.word_index) + 1
    c_vocab_size = len(c_tokenizer.word_index) + 1

    model = transformer_model(assembly_vocab_size, c_vocab_size, max_length_assembly, max_length_c, d_model, num_heads,
                              dff, num_layers)
    model.load_weights('saved_model/assembly_to_c_model.weights.h5')

    encoder_inputs = model.input[0]
    encoder_outputs = model.get_layer(index=-6).output
    encoder_model = Model(encoder_inputs, encoder_outputs)

    decoder_inputs = model.input[1]
    decoder_state_input_h = Input(shape=(max_length_assembly, d_model))
    decoder_embedding = model.layers[5](decoder_inputs)
    decoder_outputs = decoder_embedding
    for i in range(num_layers):
        decoder_outputs = decoder_layer(decoder_outputs, decoder_state_input_h, num_heads, dff, d_model,
                                        dropout_rate=0.1)
    decoder_outputs = model.layers[-1](decoder_outputs)
    decoder_model = Model([decoder_inputs, decoder_state_input_h], decoder_outputs)

    with open("inference/hello.s") as f:
        in_test = f.read()

    test_input_seq = assembly_tokenizer.texts_to_sequences([in_test])
    decoded_sentence = decode_sequence(test_input_seq, encoder_model, decoder_model, c_tokenizer, max_length_assembly,
                                       max_length_c)
    print('Decoded C code:', decoded_sentence)

if __name__ == "__main__":
    main()
