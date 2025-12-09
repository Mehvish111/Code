loss, dice_score, iou_score = model.evaluate(test_images, test_masks)
print(f'Test loss: {loss}')
print(f'Dice coefficient: {dice_score}')
print(f'IoU coefficient: {iou_score}')
