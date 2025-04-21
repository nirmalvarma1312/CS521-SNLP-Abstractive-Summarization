
import numpy as np

def decode_sequence(input_seq, encoder_model, decoder_model, target_word_index, reverse_target_word_index, max_len_summary):
    input_seq = np.array(input_seq, dtype=np.int32)
    enc_out, state_h, state_c = encoder_model.predict(input_seq)
    states_value = [state_h, state_c]

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_word_index['starttoken']

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq, enc_out] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_target_word_index.get(sampled_token_index, '')

        if sampled_word == 'endtoken' or len(decoded_sentence.split()) >= max_len_summary:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence.strip()
