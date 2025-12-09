model = build_model()
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
history = model.fit(train_data, train_masks, 
                    validation_data=(test_images, test_masks),
                    batch_size=8, epochs=50, 
                    callbacks=[reduce_lr])
