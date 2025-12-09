 model.compile(optimizer=Adam(learning_rate=1e-4), loss=combined_loss, metrics=[dice_coefficient, iou_coefficient])
    return model
