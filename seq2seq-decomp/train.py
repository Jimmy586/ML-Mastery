import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
import pickle
import os
from transformer_model import transformer_model, calculate_bleu
import nltk

nltk.download('punkt')

def load_data(base_path, subfolder, num_files, input_assembly, output_c_code):
    for i in range(1,num_files+1):
        asm_path = os.path.join(base_path, subfolder, "asm", f"{i}.s")
        c_path = os.path.join(base_path, subfolder, "c", f"{i}.c")

        if not os.path.exists(asm_path):
            print(f"File not found: {asm_path}")
            continue
        if not os.path.exists(c_path):
            print(f"File not found: {c_path}")
            continue

        with open(asm_path) as asm_file:
            input_assembly.append(asm_file.read())

        with open(c_path) as c_file:
            output_c_code.append(c_file.read())

    print(f"Loaded {len(input_assembly)} assembly files and {len(output_c_code)} C files from {subfolder}")

def prepare_data(input_assembly, output_c_code):
    """
    Prepare data by tokenizing, padding, and creating shifted target sequences.

    Parameters:
    input_assembly (list): A list containing the content of assembly files.
    output_c_code (list): A list containing the content of C code files.

    Returns:
    tuple: A tuple containing tokenized and padded input sequences, decoder input sequences, and decoder target sequences.
    """
    if len(input_assembly) == 0 or len(output_c_code) == 0:
        raise ValueError("No data to process. Check the data loading step.")

    output_c_code = ['<start> ' + seq + ' <end>' for seq in output_c_code]

    # Tokenization
    assembly_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    c_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')

    assembly_tokenizer.fit_on_texts(input_assembly)
    c_tokenizer.fit_on_texts(output_c_code)

    assembly_vocab_size = len(assembly_tokenizer.word_index) + 1
    c_vocab_size = len(c_tokenizer.word_index) + 1

    # Padding sequences
    input_sequences = assembly_tokenizer.texts_to_sequences(input_assembly)
    max_length_assembly = max(map(len, input_sequences))
    input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_length_assembly)

    output_sequences = c_tokenizer.texts_to_sequences(output_c_code)
    max_length_c = max(map(len, output_sequences))
    output_sequences = tf.keras.preprocessing.sequence.pad_sequences(output_sequences, maxlen=max_length_c)

    # Create shifted target sequences for teacher forcing
    decoder_input_sequences = output_sequences[:, :-1]
    decoder_target_sequences = output_sequences[:, 1:]

    input_sequences = input_sequences[:len(decoder_input_sequences)]
    decoder_input_sequences = decoder_input_sequences[:len(input_sequences)]
    decoder_target_sequences = decoder_target_sequences[:len(input_sequences)]

    return input_sequences, decoder_input_sequences, decoder_target_sequences, assembly_tokenizer, c_tokenizer, max_length_assembly, max_length_c

def build_and_compile_model(assembly_vocab_size, c_vocab_size, max_length_assembly, max_length_c, d_model, num_heads, dff, num_layers):
    """
    Build and compile the transformer model.
    """
    model = transformer_model(assembly_vocab_size, c_vocab_size, max_length_assembly, max_length_c - 1, d_model, num_heads, dff, num_layers)
    model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

def main():
    """
    Main function to load data, prepare data, build and train the model, and save the model and tokenizers.
    """
    input_assembly = []
    output_c_code = []

    base_path = "training_data/data"
    num_files = 20

    load_data(base_path, "1", num_files, input_assembly, output_c_code)
    load_data(base_path, "2", num_files, input_assembly, output_c_code)

    input_sequences, decoder_input_sequences, decoder_target_sequences, assembly_tokenizer, c_tokenizer, max_length_assembly, max_length_c = prepare_data(input_assembly, output_c_code)

    d_model = 64
    num_heads = 4
    dff = 64
    num_layers = 2

    model = build_and_compile_model(assembly_vocab_size=len(assembly_tokenizer.word_index) + 1, c_vocab_size=len(c_tokenizer.word_index) + 1, max_length_assembly=max_length_assembly, max_length_c=max_length_c, d_model=d_model, num_heads=num_heads, dff=dff, num_layers=num_layers)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard()

    model.fit([input_sequences, decoder_input_sequences], decoder_target_sequences, batch_size=64, epochs=2, validation_split=0.2, callbacks=[callback, tensorboard_callback])

    if not os.path.exists('saved_model'):
        os.makedirs('saved_model')

    model.save_weights('saved_model/assembly_to_c_model.weights.h5')

    with open('assembly_tokenizer.pkl', 'wb') as f:
        pickle.dump((assembly_tokenizer, max_length_assembly), f)

    with open('c_tokenizer.pkl', 'wb') as f:
        pickle.dump((c_tokenizer, max_length_c), f)

    model.summary()
    total_params = model.count_params()
    print(f"Total number of parameters: {total_params}")

if __name__ == "__main__":
    main()
