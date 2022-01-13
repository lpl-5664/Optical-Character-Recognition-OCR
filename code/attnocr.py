import tensorflow as tf
from tensorflow.keras import Model
from cnn import RepVGG
from attn_enc_dec import Encoder, Decoder

class AttentionOCR(Model):
    def __init__(self, num_blocks, width_multipliers, units, dropout=0.5, vocab=None):
        """
        Args:
            units: output channels through a LSTM layer
            dropout: default=0.5
            vocab: class Vocab containing mapping word2vec and vec2word
        """
        super(AttentionOCR, self).__init__()
        
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.max_seq_length = vocab.max_seq_length
        
        #self.cnn = EfficientNetB0(input_shape=(64, 128, 3))
        self.cnn = RepVGG(num_blocks, width_multipliers, dropout)
        self.encoder = Encoder(units)
        self.decoder = Decoder(self.vocab_size, 2*units, self.max_seq_length)
        
    def call(self, inputs, training=False):
        # 
        if len(inputs) == 2:
            input_img, onehot_label = inputs
        else:
            input_img = inputs
            onehot_label = None
            training = False
            
        # Start feeding the inputs to the model
        feature_map = self.cnn(input_img, training=training)
        enc_output, enc_states = self.encoder(feature_map)
        
        # get the shape to initialize the init_input
        channel = enc_output.get_shape()[-1]
        if channel < self.vocab_size:
            t1 = enc_output[:, :1, :self.vocab_size]
            t2 = enc_output[:, :1, :(self.vocab_size-channel)]
            temp = tf.concat([t1, t2], axis=-1)
        else:
            temp = enc_output[:, :1, :self.vocab_size]
        init_input = tf.zeros_like(temp)
        
        # Feed the results above to the decoder
        dec_output = self.decoder(init_input, enc_states, enc_output, onehot_label, training)
        return dec_output
        
    def train_step(self, inputs):
        #print(len(input))
        input_img, onehot_label = inputs
        
        with tf.GradientTape() as tape:
            y_pred = self((input_img, onehot_label), training=True)
            
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(onehot_label, y_pred, regularization_losses=self.losses)

        # Apply an optimization step
        variables = self.trainable_variables 
        gradients = tape.gradient(loss, variables)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, variables))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(onehot_label, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    
    def test_step(self, inputs):
        input_img, onehot_label = inputs
        
        # Calculate the prediction
        y_pred = self((input_img, onehot_label), training=False)
        
        # Update mstrics
        loss = self.compiled_loss(onehot_label, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(onehot_label, y_pred)
        
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}