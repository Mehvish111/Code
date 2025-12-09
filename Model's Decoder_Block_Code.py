# Decoder
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = residual_block(u6, 512)
    u6 = MCCBAMBlock()(u6)

    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(u6)
    u7 = concatenate([u7, c3])
    u7 = residual_block(u7, 256)
    u7 = MCCBAMBlock()(u7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(u7)
    u8 = concatenate([u8, c2])
    u8 = residual_block(u8, 128)
    u8 = MCCBAMBlock()(u8)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u8)
    u9 = concatenate([u9, c1])
    u9 = residual_block(u9, 64)
    u9 = MCCBAMBlock()(u9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(u9)

    model = Model(inputs=[inputs], outputs=[outputs])
   
