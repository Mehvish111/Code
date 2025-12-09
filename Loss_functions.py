def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def iou_coefficient(y_true, y_pred, smooth=1):
    y_true_f = tf.cast(y_true, tf.float32)  # Ensure both tensors are float32
    y_pred_f = tf.cast(y_pred, tf.float32)  # Ensure both tensors are float32
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    true_pos = tf.keras.backend.sum(y_true_f * y_pred_f)
    false_neg = tf.keras.backend.sum(y_true_f * (1 - y_pred_f))
    false_pos = tf.keras.backend.sum((1 - y_true_f) * y_pred_f)
    return 1 - (true_pos + 1) / (true_pos + alpha * false_neg + beta * false_pos + 1)

def combined_loss(y_true, y_pred):
    return 0.5 * tversky_loss(y_true, y_pred) + 0.5 * (1 - dice_coefficient(y_true, y_pred))
