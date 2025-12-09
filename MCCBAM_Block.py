class MCCBAMBlock(Layer):
    def __init__(self, ratio=8):
        super(MCCBAMBlock, self).__init__()
        self.ratio = ratio

    def build(self, input_shape):
        self.channel_shared_dense = Dense(input_shape[-1] // self.ratio, activation='relu')
        self.channel_dense = Dense(input_shape[-1])

        self.conv_1x1 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')
        self.conv_3x3 = Conv2D(1, (3, 3), padding='same', activation='sigmoid')
        self.conv_5x5 = Conv2D(1, (5, 5), padding='same', activation='sigmoid')

    def call(self, input_tensor):
        channel = input_tensor.shape[-1]

        # Channel attention
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        avg_pool = Reshape((1, 1, channel))(avg_pool)
        avg_pool = self.channel_shared_dense(avg_pool)
        avg_pool = self.channel_dense(avg_pool)

        max_pool = GlobalMaxPooling2D()(input_tensor)
        max_pool = Reshape((1, 1, channel))(max_pool)
        max_pool = self.channel_shared_dense(max_pool)
        max_pool = self.channel_dense(max_pool)

        channel_attention = Activation('sigmoid')(Add()([avg_pool, max_pool]))
        channel_attention = Multiply()([input_tensor, channel_attention])

        # Spatial attention with cascaded convolutions
        avg_pool_spatial = tf.reduce_mean(channel_attention, axis=-1, keepdims=True)
        max_pool_spatial = tf.reduce_max(channel_attention, axis=-1, keepdims=True)
        spatial_attention = concatenate([avg_pool_spatial, max_pool_spatial], axis=-1)
          
        # Multi-Cascading Spatial Attention (sequential cascade)
        attention_1x1 = self.conv_1x1(spatial_attention)
        attention_3x3 = self.conv_3x3(attention_1x1)  # 1x1 output as input to 3x3
        attention_5x5 = self.conv_5x5(attention_3x3)  # 3x3 output as input to 5x5
          
        spatial_attention = Add()([attention_1x1, attention_3x3, attention_5x5])
          
        return Multiply()([channel_attention, spatial_attention])

        
