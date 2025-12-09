def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding='same')(x)
    shortcut = BatchNormalization()(shortcut)

    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([x, shortcut])
    x = LeakyReLU(alpha=0.1)(x)
    return x
