import tensorflow as tf
def quoridor_loss(pred_pis, target_pis, pred_vs, target_vs):
        # Initialize loss functions
        cce = tf.keras.losses.CategoricalCrossentropy()
        mse = tf.keras.losses.MeanSquaredError()

        # Calculate loss
        loss_pi = cce(target_pis, pred_pis)
        loss_v = mse(target_vs, tf.reshape(pred_vs, shape=[-1,]))
        return loss_pi + loss_v