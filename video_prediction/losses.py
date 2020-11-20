import tensorflow as tf
import numpy as np

from video_prediction.ops import sigmoid_kl_with_logits

def l2_loss(pred, target):
    #Unpack the channels 
    red, green, blue, nir, _, _, _, _, _, _, _, cloud_mask = tf.unstack(target, axis=4, name='unstack_target')
    pred_red, pred_green, pred_blue, pred_nir, _, _, _, _, _, _, _, _ = tf.unstack(pred, axis=4, name='unstack_pred')

    tf_zero = tf.constant(0, dtype=tf.float32)
    
    #Stack channels
    cloud_mask = tf.stack([cloud_mask, cloud_mask, cloud_mask, cloud_mask], axis=4, name='stack_cloud_mask')
    target = tf.stack([red, green, blue, nir], axis=4, name='stack_target')
    pred = tf.stack([pred_red, pred_green, pred_blue, pred_nir], axis=4, name='stack_pred')
    
    square_error = tf.square(target - pred)
    square_error = tf.where(cloud_mask > tf_zero, tf.zeros_like(square_error), square_error)

    return tf.reduce_mean(square_error)

def l1_loss(pred, target):
    #Unpack the channels 
    red, green, blue, nir, _, _, _, _, _, _, _, cloud_mask = tf.unstack(target, axis=4, name='unstack_target')
    pred_red, pred_green, pred_blue, pred_nir, _, _, _, _, _, _, _, _ = tf.unstack(pred, axis=4, name='unstack_pred')

    tf_zero = tf.constant(0, dtype=tf.float32)
    
    #Stack channels
    cloud_mask = tf.stack([cloud_mask, cloud_mask, cloud_mask, cloud_mask], axis=4, name='stack_cloud_mask')
    target = tf.stack([red, green, blue, nir], axis=4, name='stack_target')
    pred = tf.stack([pred_red, pred_green, pred_blue, pred_nir], axis=4, name='stack_pred')
    
    absolute_error = tf.abs(target - pred)
    absolute_error = tf.where(cloud_mask > tf_zero, tf.zeros_like(absolute_error), absolute_error)

    return tf.reduce_mean(absolute_error)

def l2_NDVI_loss(pred, target):
    #Unpack the channels 
    red, _, _, nir, _, _, _, _, _, _, _, cloud_mask = tf.unstack(target, axis=4, name='unstack_target')
    pred_red, _, _, pred_nir, _, _, _, _, _, _, _, _ = tf.unstack(pred, axis=4, name='unstack_pred')

    tf_zero = tf.constant(0, dtype=tf.float32)
    tf_one = tf.constant(1,dtype=tf.float32)
    tf_epsilon = tf.fill(tf.shape(red), 0.000000001)
    
    #compute NDVI
    target_NDVI = (nir-red)/(nir+red+tf_epsilon)
    pred_NDVI = (pred_nir-pred_red)/(pred_nir+pred_red+tf_epsilon)
    
    #Bound it 0-1 just in case
    target_NDVI = tf.where(target_NDVI < tf_zero, tf.zeros_like(target_NDVI), target_NDVI)
    pred_NDVI = tf.where(pred_NDVI < tf_zero, tf.zeros_like(pred_NDVI), pred_NDVI)
    target_NDVI = tf.where(target_NDVI > tf_one, tf.ones_like(target_NDVI), target_NDVI)
    pred_NDVI = tf.where(pred_NDVI > tf_one, tf.ones_like(pred_NDVI), pred_NDVI)
    
    
    square_error = tf.square(target_NDVI - pred_NDVI)
    square_error = tf.where(cloud_mask > tf_zero, tf.zeros_like(square_error), square_error)
    
    return tf.reduce_mean(square_error)

def variability_loss(pred, target):
    tf_zero = tf.constant(0, dtype=tf.float32)
    
    #Unpack the channels
    pred_red, pred_green, pred_blue, pred_nir, _, _, _, _, _, _, _, _ = tf.unstack(pred, axis=4, name='unstack_pred')
    red, green, blue, nir, _, _, _, _, _, _, _, cloud_mask = tf.unstack(target, axis=4, name='unstack_target')
    
    #Stack channels
    pred = tf.stack([pred_red, pred_green, pred_blue, pred_nir], axis=4, name='stack_pred')
    target = tf.stack([red, green, blue, nir], axis=4, name='stack_target')
    cloud_mask = tf.stack([cloud_mask, cloud_mask, cloud_mask, cloud_mask], axis=4, name='stack_cloud_mask')
    
    target = tf.where(cloud_mask > tf_zero, tf.zeros_like(target), target)
    pred = tf.where(cloud_mask > tf_zero, tf.zeros_like(pred), pred)
                    
    samples_norm_stdev = []
    for sample in tf.unstack(target, axis=1, name='unstack_batch'):
        mean = tf.reduce_mean(sample)
        deviations = []
        for num, frame in enumerate(tf.unstack(sample, axis=0, name='unstack_frames'), start=1):
            if num == 1:
                deviations.append(tf.constant(0,dtype=tf.float32))
            if num > 1:
                deviations.append(tf.square(tf.reduce_mean(old_frame)-tf.reduce_mean(frame)))
            old_frame = frame
            
        variance = tf.reduce_sum(tf.stack(deviations))/tf.constant(len(tf.unstack(sample, axis=0, name='unstack_frames')),dtype=tf.float32)    
        target_stdev = tf.sqrt(variance)
        
    samples_norm_stdev = []
    for sample in tf.unstack(pred, axis=1, name='unstack_batch'):
        mean = tf.reduce_mean(sample)
        deviations = []
        for num, frame in enumerate(tf.unstack(sample, axis=0, name='unstack_frames'), start=1):
            if num == 1:
                deviations.append(tf.constant(0,dtype=tf.float32))
            if num > 1:
                deviations.append(tf.square(tf.reduce_mean(old_frame)-tf.reduce_mean(frame)))
            old_frame = frame
            
        variance = tf.reduce_sum(tf.stack(deviations))/tf.constant(len(tf.unstack(sample, axis=0, name='unstack_frames')),dtype=tf.float32)    
        stdev = tf.sqrt(variance)
        
        ######Only if stdev should go towards a target other than 0
        #target_stdev = tf.constant(0.002, dtype=tf.float32)
        stdev = tf.abs(stdev - target_stdev)
        
        samples_norm_stdev.append(stdev/mean)
        
    return tf.reduce_mean(tf.stack(samples_norm_stdev))

def charbonnier_loss(x, epsilon=0.001):
    return tf.reduce_mean(tf.sqrt(tf.square(x) + tf.square(epsilon)))


def gan_loss(logits, labels, gan_loss_type):
    # use 1.0 (or 1.0 - discrim_label_smooth) for real data and 0.0 for fake data
    if gan_loss_type == 'GAN':
        # discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
        # gen_loss = tf.reduce_mean(-tf.log(predict_fake + EPS))
        if labels in (0.0, 1.0):
            labels = tf.constant(labels, dtype=logits.dtype, shape=logits.get_shape())
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
        else:
            loss = tf.reduce_mean(sigmoid_kl_with_logits(logits, labels))
    elif gan_loss_type == 'LSGAN':
        # discrim_loss = tf.reduce_mean((tf.square(predict_real - 1) + tf.square(predict_fake)))
        # gen_loss = tf.reduce_mean(tf.square(predict_fake - 1))
        loss = tf.reduce_mean(tf.square(logits - labels))
    elif gan_loss_type == 'SNGAN':
        # this is the form of the loss used in the official implementation of the SNGAN paper, but it leads to
        # worse results in our video prediction experiments
        if labels == 0.0:
            loss = tf.reduce_mean(tf.nn.softplus(logits))
        elif labels == 1.0:
            loss = tf.reduce_mean(tf.nn.softplus(-logits))
        else:
            raise NotImplementedError
    else:
        raise ValueError('Unknown GAN loss type %s' % gan_loss_type)
    return loss


def kl_loss(mu, log_sigma_sq):
    sigma_sq = tf.exp(log_sigma_sq)
    return -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_sigma_sq - tf.square(mu) - sigma_sq, axis=-1))
