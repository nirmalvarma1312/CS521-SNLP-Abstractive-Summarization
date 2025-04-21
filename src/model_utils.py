
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from src.attention_layer import AttentionLayer

def build_seq2seq_model(x_voc_size, y_voc_size, embedding_dim, latent_dim, max_len_text, max_len_summary):
    # Encoder
    encoder_inputs = Input(shape=(max_len_text,))
    enc_emb = Embedding(x_voc_size, embedding_dim, trainable=True)(encoder_inputs)
    encoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
    encoder_output1, _, _ = encoder_lstm1(enc_emb)
    encoder_lstm2 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
    encoder_output2, _, _ = encoder_lstm2(encoder_output1)
    encoder_lstm3 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
    encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2)

    # Decoder
    decoder_inputs = Input(shape=(max_len_summary,))
    dec_emb_layer = Embedding(y_voc_size, embedding_dim, trainable=True)
    dec_emb = dec_emb_layer(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.2)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

    # Attention
    attn_layer = AttentionLayer()
    attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])
    decoder_concat_input = Concatenate(axis=-1)([decoder_outputs, attn_out])

    # Output layer
    decoder_dense = TimeDistributed(Dense(y_voc_size, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_concat_input)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model
