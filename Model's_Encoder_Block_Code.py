def build_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    inputs = Input(input_shape)

    # Encoder
    c1 = residual_block(inputs, 64)
    c1 = MCCBAMBlock()(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = residual_block(p1, 128)
    c2 = MCCBAMBlock()(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = residual_block(p2, 256)
    c3 = MCCBAMBlock()(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = residual_block(p3, 512)
    c4 = MCCBAMBlock()(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = residual_block(p4, 1024)
    c5 = MCCBAMBlock()(c5)
