import tensorflow as tf

class Vocab():
    def __init__(self):
        self.pad = 0
        self.go = 1
        self.eos = 2
        self.mask_token = 3

        self.chars = """aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐðÐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~–°−“”—’ """
        self.max_seq_length = 20

        self.c2i = {c:i+4 for i, c in enumerate(self.chars)}

        self.i2c = {i+4:c for i, c in enumerate(self.chars)}
        
        self.i2c[0] = '<pad>'
        self.i2c[1] = '<sos>'
        self.i2c[2] = '<eos>'
        self.i2c[3] = '*'
    
    def encode(self, text):
        """ 
        A function that converts a word into a vector 
        according to its position in self.chars
        Arguments: 
            text: word that needs to be converted
        Return:
            text: text that has been converted
        """
        if tf.is_tensor(text):
            if text.get_shape() != ():
                batch = []
                for sample in text:
                    encoded = [self.go] + [self.c2i[c] for c in sample.numpy().decode()] + [self.eos]
                    if len(encoded) < self.max_seq_length:
                        encoded += [self.pad]*(self.max_seq_length-len(encoded))
                    batch.append(encoded)
                text = tf.constant(batch, dtype=tf.float32)
                del batch
            else:
                encoded = [self.go] + [self.c2i[c] for c in text.numpy().decode()] + [self.eos]
                if len(encoded) < self.max_seq_length:
                    encoded += [self.pad]*(self.max_seq_length-len(encoded))
                text = tf.constant(encoded, dtype=tf.float32)
        
        else:
            for idx in range(len(text)):
                encoded = [self.go] + [self.c2i[c] for c in text[idx]] + [self.eos] + [self.pad]
                if len(encoded) < self.max_seq_length:
                    encoded += [self.pad]*(self.max_seq_length-len(encoded))
                text[idx] = encoded
        return text
    
    def tf_encode(self, text):
        """
        A function that enables the encode function to work 
        in graph mode (while tensorflow runs the model)
        Arguments:
            text: a word that needs to be converted
        Return:
            encoded_text: text that is encoded
        """
        # Wrap in py_function to enable the function while eager execution is off
        encoded_text = tf.py_function(self.encode, [text], tf.float32)
        # Reshape the vector to its original shape
        encoded_text = tf.reshape(encoded_text, [self.max_seq_length])
        return encoded_text
    
    def decode(self, ids):
        """
        A function that decodes vectors to human-readable text
        Arguments: 
            ids: the vector of text
        Return:
            sent: message after decoded
        """
        if tf.is_tensor(ids):
            ids = ids.numpy()
        ids = list(ids)
        first = 1 if self.go in ids else 0
        last = ids.index(self.eos) if self.eos in ids else None
        sent = ''.join([self.i2c[i] for i in ids[first:last]])
        return sent
    
    '''def encode(self, chars):
        return [self.go] + [self.c2i[c] for c in chars] + [self.eos]
    
    def decode(self, ids):
        first = 1 if self.go in ids else 0
        last = ids.index(self.eos) if self.eos in ids else None
        sent = ''.join([self.i2c[i] for i in ids[first:last]])
        return sent'''
    
    def __len__(self):
        return len(self.c2i) + 4
    
    def batch_decode(self, arr):
        """
        A function to decode in batches
        Arguments:
            arr: a batch of encoded messages
        Return:
            texts: a batch of decoded messages
        """
        if tf.is_tensor(arr):
            texts = [self.decode(ids) for ids in arr.numpy()]
        else:
            texts = [self.decode(ids) for ids in arr]
        return texts

    def __str__(self):
        return self.chars

