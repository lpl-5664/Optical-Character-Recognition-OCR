import tensorflow as tf
from tensorflow.keras.layers import Layer, LSTM, Bidirectional, Concatenate, Dense

class Encoder(Layer):
    def __init__(self, units):
        super(Encoder, self).__init__()

        self.rnn = Bidirectional(LSTM(units, return_sequences=True, return_state=True))
        self.concath = Concatenate()
        self.concatc = Concatenate()

    def call(self, inputs, state=None):
        """
        Args:
            inputs: a tensor of shape (batch, timesteps, feature)
            state: list of initial states for the encoder, None by default
            
        Returns:
            outputs: output of a BiLSTM of shape (batch, timesteps, 2*feature)
            states: list of output states of a BiLSTM
        """
        output, forward_h, forward_c, backward_h, backward_c = self.rnn(inputs, initial_state=state)
        state_h = self.concath([forward_h, backward_h])
        state_c = self.concatc([forward_c, backward_c])
        
        return output, [state_h, state_c]

class Attention(Layer):
    def __init__(self, units):
        super(Attention, self).__init__()

        self.fc1 = Dense(units, use_bias=False)
        self.fc2 = Dense(units, use_bias=False)
        #self.attention = AdditiveAttention()

    def call(self, query, value, key):
        """
        Args:
            query: a tensor of shape [bs, Tq, dim]
            value: a tensor of shape [bs, Tv, dim]
            key: often resemble teh value tensor
        
        Return:
            context_vector: shape [bs, Tq, dim_q]
            attention_weights: shape [bs, Tq, Tv]
        """
        
        fc_key = self.fc1(key)
        fc_value = self.fc2(value)
        
        query_key_score = tf.matmul(query, fc_key, transpose_b=True)
        distribution = tf.nn.softmax(query_key_score)
        
        # Get context vector of the attention layer
        context_vector = tf.matmul(distribution, fc_value)
        '''context_vector, attention_weights = self.attention(
            inputs = [fc1_query, value, fc2_key], return_attention_scores = True)'''
        
        return context_vector, distribution   

class Decoder(Layer):
    def __init__(self, output_dim, units, max_seq_length):
        super(Decoder, self).__init__()
        
        self.output_dim = output_dim
        self.seq_size = max_seq_length

        self.rnn = LSTM(units, return_state=True)
        self.attention = Attention(units)
        self.query = Dense(units)
        self.fc_out = Dense(output_dim, activation=tf.nn.softmax)
        
    def to_one_hot(self, distributed):
        """
        Args:
            distributed: a tensor of probability of shape (batch_size, 1, vocab_size)
        
        Returns:
            onehot_output: a onehot vector of prediction
        """
        # reduce dim to (batch_size, vocab_size)
        distributed = tf.squeeze(distributed, 1)
        
        indices = tf.argmax(distributed, -1)
        onehot_output = tf.expand_dims(tf.one_hot(indices, self.output_dim), 1)
        return onehot_output

    def call(self, init_input, states, enc_output, target_label=None, training=False):
        """
        Args:
            init_input: a zeros vector of shape (batch_size, 1, vocab_size)
            states: output states by the encoder of shape (batch_size, 2*enc_units)
            enc_output: output result of the encoder of shape (batch_size, T, 2*enc_units)
            target_label: a onehot label of shape (bs, max_seq_length, vocab_size), None by default for inference
            training: set to True for training process, False by default for inference
            
        Returns:
            dec_output: output of a decoder of shape (batch_size, max_seq_length, vocab_size)
            
        """
        dec_states = states
        enc_state_h, enc_state_c = states
        enc_state_h = self.query(enc_state_h)
        context_vector = tf.expand_dims(enc_state_h, 1)
        
        input_t = init_input
        dec_output = tf.zeros_like(init_input)
        
        # for training
        if training:
            for i in range(self.seq_size):
                rnn_outputs  = self.rnn(Concatenate()([context_vector, input_t]), initial_state=dec_states)
                rnn_output, rnn_state_h, rnn_state_c = rnn_outputs
                dec_states = [rnn_state_h, rnn_state_c]
                # reshape rnn output to shape (batch_size, 1, vocab_size)
                rnn_output = self.fc_out(rnn_output)
                rnn_output = tf.expand_dims(rnn_output, 1)
                # reshape rnn states to (batch_size, 1, 2*enc_units)
                rnn_state_h = tf.expand_dims(rnn_state_h, 1)
                
                
                context_vector, attention_weights = self.attention(rnn_state_h, enc_output, enc_output)
                dec_output = tf.concat([dec_output, rnn_output], 1)
                input_t = tf.expand_dims(target_label[:, i], 1)
            return dec_output[:, 1:]
        
        # for inference 
        else:
            for i in range(self.seq_size):
                rnn_outputs  = self.rnn(Concatenate()([context_vector, input_t]), initial_state=dec_states)
                rnn_output, rnn_state_h, rnn_state_c = rnn_outputs
                dec_states = [rnn_state_h, rnn_state_c]
                # reshape rnn output to shape (batch_size, 1, vocab_size)
                rnn_output = self.fc_out(rnn_output)
                rnn_output = tf.expand_dims(rnn_output, 1)
                # reshape rnn states to (batch_size, 1, 2*enc_units)
                rnn_state_h = tf.expand_dims(rnn_state_h, 1)
                
                context_vector, attention_weights = self.attention(rnn_state_h, enc_output, enc_output)
                dec_output = tf.concat([dec_output, rnn_output], 1)
                
                # Previous version
                """rnn_outputs = self.rnn(Concatenate()([context_vector, input_t]))
                rnn_output, rnn_state = rnn_outputs[1], rnn_outputs[2]
                # reshape rnn output to shape (batch_size, 1, vocab_size)
                rnn_output = self.fc_out(rnn_output)
                rnn_output = tf.expand_dims(rnn_output, 1)
                # reshape rnn state to (batch_size, 1, 2*enc_units)
                rnn_state = tf.expand_dims(rnn_state, 1)
                
                context_vector, attention_weights = self.attention(rnn_state, enc_output, enc_output)
                dec_output = tf.concat([dec_output, rnn_output], 1)"""
                input_t = self.to_one_hot(rnn_output)
            
            return dec_output[:, 1:]