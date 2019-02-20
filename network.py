import tensorflow as tf
# import tensorflow.tflearn
#80x512

def hg_inference(inputs, is_training = False, scope = 'hg', unfreeze = True, reuse = None):
    with tf.variable_scope(scope):
        #40x64
        conv_1 = tf.layers.conv2d(inputs, 256, 1, padding='same', activation = 'relu', name = 'hg_conv_1')

        conv1_1 = tf.layers.conv2d(conv_1, 256, 1, padding = 'same', name = 'conv1_1', trainable = unfreeze, reuse=reuse)
        conv1_1 = tf.layers.batch_normalization(conv1_1, training = is_training, name = 'bn1_1', trainable = unfreeze, reuse=reuse)
        conv1_1 = tf.nn.relu(conv1_1)

        conv1_2 = tf.layers.conv2d(conv1_1, 128, 3, padding = 'same', name = 'conv1_2', trainable = unfreeze, reuse = reuse)
        conv1_2 = tf.layers.batch_normalization(conv1_2, training = is_training, name = 'bn1_2', trainable = unfreeze, reuse=reuse)
        conv1_2 = tf.nn.relu(conv1_2)

        conv1_3 = tf.layers.conv2d(conv1_2, 256, 1, padding = 'same', name = 'conv1_3', trainable = unfreeze, reuse = reuse)
        conv1_3 = tf.layers.batch_normalization(conv1_3, training = is_training, name = 'bn1_3', trainable = unfreeze, reuse = reuse)
        conv1_3 = tf.nn.relu(conv1_3)
        residual1 = tf.add(conv_1, conv1_3, name='hg_block1_add')

        pool1 = tf.layers.max_pooling2d(residual1, 2, strides = (2, 2), name = 'max_pool_1')
	   #20x32
        branch1_1 = tf.layers.conv2d(residual1, 128, 1, padding='same', name = "branch1_1", trainable = unfreeze, reuse = reuse)
        branch1_1 = tf.layers.batch_normalization(branch1_1, training = is_training, name = 'branch1_1_bn_1', trainable = unfreeze, reuse=reuse)
        branch1_1 = tf.nn.relu(branch1_1)

        branch1_2 = tf.layers.conv2d(branch1_1, 128 ,3, padding = 'same', name = 'branch1_2', trainable= unfreeze, reuse=reuse)
        branch1_2 = tf.layers.batch_normalization(branch1_2, training=is_training, name= 'branch1_2_bn_1', trainable = unfreeze, reuse=reuse)
        branch1_2 = tf.nn.relu(branch1_2)

        branch1_3 = tf.layers.conv2d(branch1_2, 256 ,1, padding = 'same', name = 'branch1_3', trainable= unfreeze, reuse=reuse)
        branch1_3 = tf.layers.batch_normalization(branch1_3, training=is_training, name= 'branch1_3_bn_1', trainable = unfreeze, reuse=reuse)
        branch1_3 = tf.nn.relu(branch1_3)
        bresidual1 = tf.add(residual1, branch1_3, name='hg_branch_block1_add')

        conv2_1 = tf.layers.conv2d(pool1, 128, 1, padding = 'same', name = 'conv2_1', trainable = unfreeze, reuse=reuse)
        conv2_1 = tf.layers.batch_normalization(conv2_1, training = is_training, name='bn2_1', trainable = unfreeze, reuse=reuse)
        conv2_1 = tf.nn.relu(conv2_1)

        conv2_2 = tf.layers.conv2d(conv2_1, 128, 3, padding = 'same', name = 'conv2_2', trainable = unfreeze, reuse=reuse)
        conv2_2 = tf.layers.batch_normalization(conv2_2, training = is_training, name='bn2_2', trainable = unfreeze, reuse=reuse)
        conv2_2 = tf.nn.relu(conv2_2)

        conv2_3 = tf.layers.conv2d(conv2_2, 256, 1, padding = 'same', name = 'conv2_3', trainable = unfreeze, reuse=reuse)
        conv2_3 = tf.layers.batch_normalization(conv2_3, training = is_training, name='bn2_3', trainable = unfreeze, reuse=reuse)
        conv2_3 = tf.nn.relu(conv2_3)
        residual2 = tf.add(pool1, conv2_3, name='hg_block2_add')

        pool2 = tf.layers.max_pooling2d(residual2, 2, strides = (2, 2), name = 'max_pool_2')
	#10x16
        branch2_1 = tf.layers.conv2d(residual2, 128, 1, padding='same', name = "branch2_1", trainable = unfreeze, reuse = reuse)
        branch2_1 = tf.layers.batch_normalization(branch2_1, training = is_training, name = 'branch2_1_bn_1', trainable = unfreeze, reuse=reuse)
        branch2_1 = tf.nn.relu(branch2_1)

        branch2_2 = tf.layers.conv2d(branch2_1, 128 ,3, padding = 'same', name = 'branch2_2', trainable= unfreeze, reuse=reuse)
        branch2_2 = tf.layers.batch_normalization(branch2_2, training=is_training, name= 'branch2_2_bn_1', trainable = unfreeze, reuse=reuse)
        branch2_2 = tf.nn.relu(branch2_2)

        branch2_3 = tf.layers.conv2d(branch2_2, 256 ,1, padding = 'same', name = 'branch2_3', trainable= unfreeze, reuse=reuse)
        branch2_3 = tf.layers.batch_normalization(branch2_3, training=is_training, name= 'branch2_3_bn_1', trainable = unfreeze, reuse=reuse)
        branch2_3 = tf.nn.relu(branch2_3)
        bresidual2 = tf.add(residual2, branch2_3, name='hg_branch_block2_add')

        conv3_1 = tf.layers.conv2d(pool2, 128, 1, padding = 'same', name = 'conv3_1', trainable = unfreeze, reuse=reuse)
        conv3_1 = tf.layers.batch_normalization(conv3_1, training = is_training, name='bn3_1', trainable = unfreeze, reuse=reuse)
        conv3_1 = tf.nn.relu(conv3_1)

        conv3_2 = tf.layers.conv2d(conv3_1, 128, 3, padding = 'same', name = 'conv3_2', trainable = unfreeze, reuse=reuse)
        conv3_2 = tf.layers.batch_normalization(conv3_2, training = is_training, name='bn3_2', trainable = unfreeze, reuse=reuse)
        conv3_2 = tf.nn.relu(conv3_2)

        conv3_3 = tf.layers.conv2d(conv3_2, 256, 1, padding = 'same', name = 'conv3_3', trainable = unfreeze, reuse=reuse)
        conv3_3 = tf.layers.batch_normalization(conv3_3, training = is_training, name='bn3_3', trainable = unfreeze, reuse=reuse)
        conv3_3 = tf.nn.relu(conv3_3)
        residual3 = tf.add (pool2, conv3_3, name='hg_block3_add')

        pool3 = tf.layers.max_pooling2d(residual3, 2, strides = (2, 2), name = 'max_pool_3')
	#5x8
        branch3_1 = tf.layers.conv2d(residual3, 128, 1, padding='same', name = "branch3_1", trainable = unfreeze, reuse = reuse)
        branch3_1 = tf.layers.batch_normalization(branch3_1, training = is_training, name = 'branch3_1_bn_1', trainable = unfreeze, reuse=reuse)
        branch3_1 = tf.nn.relu(branch3_1)

        branch3_2 = tf.layers.conv2d(branch3_1, 128 ,3, padding = 'same', name = 'branch3_2', trainable= unfreeze, reuse=reuse)
        branch3_2 = tf.layers.batch_normalization(branch3_2, training=is_training, name= 'branch3_2_bn_1', trainable = unfreeze, reuse=reuse)
        branch3_2 = tf.nn.relu(branch3_2)

        branch3_3 = tf.layers.conv2d(branch3_2, 256 ,1, padding = 'same', name = 'branch3_3', trainable= unfreeze, reuse=reuse)
        branch3_3 = tf.layers.batch_normalization(branch3_3, training=is_training, name= 'branch3_3_bn_1', trainable = unfreeze, reuse=reuse)
        branch3_3 = tf.nn.relu(branch3_3)
        bresidual3 = tf.add(residual3, branch3_3, name='hg_branch_block3_add')
        ########################### BOTTLENECK ##################################################################################################
        conv4_1 = tf.layers.conv2d(pool3, 128, 1, padding = 'same', name = 'conv4_1', trainable = unfreeze, reuse=reuse)
        conv4_1 = tf.layers.batch_normalization(conv4_1, training = is_training, name='bn4_1', trainable = unfreeze, reuse=reuse)
        conv4_1 = tf.nn.relu(conv4_1)

        conv4_2 = tf.layers.conv2d(conv4_1, 128, 3, padding = 'same', name = 'conv4_2', trainable = unfreeze, reuse=reuse)
        conv4_2 = tf.layers.batch_normalization(conv4_2, training = is_training, name='bn4_2', trainable = unfreeze, reuse=reuse)
        conv4_2 = tf.nn.relu(conv4_2)

        conv4_3 = tf.layers.conv2d(conv4_2, 256, 1, padding = 'same', name = 'conv4_3', trainable = unfreeze, reuse=reuse)
        conv4_3 = tf.layers.batch_normalization(conv4_3, training = is_training, name='bn4_3', trainable = unfreeze, reuse=reuse)
        conv4_3 = tf.nn.relu(conv4_3)
        residual4 = tf.add(pool3, conv4_3, name='hg_block4_add')

        conv5_1 = tf.layers.conv2d(residual4, 128, 1, padding = 'same', name = 'conv5_1', trainable = unfreeze, reuse=reuse)
        conv5_1 = tf.layers.batch_normalization(conv5_1, training = is_training, name='bn5_1', trainable = unfreeze, reuse=reuse)
        conv5_1 = tf.nn.relu(conv5_1)

        conv5_2 = tf.layers.conv2d(conv5_1, 128, 3, padding = 'same', name = 'conv5_2', trainable = unfreeze, reuse=reuse)
        conv5_2 = tf.layers.batch_normalization(conv5_2, training = is_training, name='bn5_2', trainable = unfreeze, reuse=reuse)
        conv5_2 = tf.nn.relu(conv5_2)

        conv5_3 = tf.layers.conv2d(conv5_2, 256, 1, padding = 'same', name = 'conv5_3', trainable = unfreeze, reuse=reuse)
        conv5_3 = tf.layers.batch_normalization(conv5_3, training = is_training, name='bn5_3', trainable = unfreeze, reuse=reuse)
        conv5_3 = tf.nn.relu(conv5_3)
        residual5 = tf.add (residual4, conv5_3, name='hg_block5_add')
        ##############################################################################################################################
        up1_1 = tf.layers.conv2d_transpose(residual5, 256, 1 , strides=(2, 2), padding = 'same', name='hg_up1', trainable=unfreeze, reuse = reuse)
        up1_1 = tf.layers.batch_normalization(up1_1, training = is_training, name='up1_bn1', trainable = unfreeze, reuse=reuse)#28
        up1_1 = tf.nn.relu(up1_1)
        add1 = tf.add(up1_1, bresidual3, name='hg_up1_add')
	#10x16
        uconv1_1 = tf.layers.conv2d(add1, 128, 1, padding='same', name='hg_upconv1_1', )
        uconv1_1 = tf.layers.batch_normalization(uconv1_1, name='bn1_uconv1', trainable=unfreeze, reuse = reuse)
        uconv1_1 = tf.nn.relu(uconv1_1)

        uconv1_2 = tf.layers.conv2d(uconv1_1, 128, 3, padding='same', name='hg_upconv1_2', )
        uconv1_2 = tf.layers.batch_normalization(uconv1_2, name='bn2_uconv1' , trainable=unfreeze, reuse = reuse)
        uconv1_2 = tf.nn.relu(uconv1_2)

        uconv1_3 = tf.layers.conv2d(uconv1_2, 256, 1, padding='same', name='hg_upconv1_3', )
        uconv1_3 = tf.layers.batch_normalization(uconv1_3, name='bn3_uconv1' ,trainable=unfreeze, reuse = reuse)
        uconv1_3 = tf.nn.relu(uconv1_3)
        uresidual1 = tf.add(add1, uconv1_3, name='hg_upblock1_add')

        up2_1 = tf.layers.conv2d_transpose(uresidual1, 256, 1 , strides=(2, 2), padding = 'same', name='hg_up2', trainable=unfreeze, reuse = reuse)
        up2_1 = tf.layers.batch_normalization(up2_1, training = is_training, name='up2_bn1', trainable = unfreeze, reuse=reuse)#56
        up2_1 = tf.nn.relu(up2_1)
        add2 = tf.add(up2_1, bresidual2, name='hg_up2_add')
	#20x32
        uconv2_1 = tf.layers.conv2d(add2, 128, 1, padding='same', name='hg_upconv2_1', )
        uconv2_1 = tf.layers.batch_normalization(uconv2_1, name='bn1_uconv2' ,trainable=unfreeze, reuse = reuse)
        uconv2_1 = tf.nn.relu(uconv2_1)

        uconv2_2 = tf.layers.conv2d(uconv2_1, 128, 3, padding='same', name='hg_upconv2_2', )
        uconv2_2 = tf.layers.batch_normalization(uconv2_2, name='bn2_uconv2'  ,trainable=unfreeze, reuse = reuse)
        uconv2_2 = tf.nn.relu(uconv2_2)

        uconv2_3 = tf.layers.conv2d(uconv2_2, 256, 1, padding='same', name='hg_upconv2_3', )
        uconv2_3 = tf.layers.batch_normalization(uconv2_3, name='bn3_uconv2', trainable=unfreeze, reuse = reuse)
        uconv2_3 = tf.nn.relu(uconv2_3)
        uresidual2 = tf.add(add2, uconv2_3, name='hg_upblock1_add')

        up3_1 = tf.layers.conv2d_transpose(uresidual2, 256, 1 , strides = (2, 2), padding = 'same', name='hg_up3', trainable=unfreeze, reuse = reuse)
        up3_1 = tf.layers.batch_normalization(up3_1, training = is_training, name='up3_bn1', trainable = unfreeze, reuse=reuse)#112
        up3_1 = tf.nn.relu(up3_1)
        add3 = tf.add(up3_1, bresidual1, name='hg_up3_add')
	#40x64
        uconv3_1 = tf.layers.conv2d(add3, 128, 1, padding='same', name='hg_upconv3_1', )
        uconv3_1 = tf.layers.batch_normalization(uconv3_1, name='bn1_uconv3', trainable=unfreeze, reuse = reuse)
        uconv3_1 = tf.nn.relu(uconv3_1)

        uconv3_2 = tf.layers.conv2d(uconv3_1, 128, 3, padding='same', name='hg_upconv3_2', )
        uconv3_2 = tf.layers.batch_normalization(uconv3_2, name='bn2_uconv3' , trainable=unfreeze, reuse = reuse)
        uconv3_2 = tf.nn.relu(uconv3_2)

        uconv3_3 = tf.layers.conv2d(uconv3_2, 256, 1, padding='same', name='hg_upconv3_3', )
        uconv3_3 = tf.layers.batch_normalization(uconv3_3, name='bn3_uconv3',  trainable=unfreeze, reuse = reuse)
        uconv3_3 = tf.nn.relu(uconv3_3)
        uresidual3 = tf.add(add3, uconv3_3, name='hg_upblock1_add')

        out_hg = tf.layers.conv2d(uresidual3, 128, 1, activation='relu', padding='same', name='hg_out')
        return out_hg


def encoder_inference(inputs, is_training = False, scope = 'encoder', unfreeze = True):
    with tf.variable_scope(scope):
        Econv1_1 = tf.layers.conv2d(inputs, 32, 3, padding = 'same', name = 'conv_1_1', trainable = unfreeze)
        Econv1_1 = tf.layers.batch_normalization(Econv1_1, training = is_training, name = 'bn_1_1', trainable = unfreeze)
        Econv1_1 = tf.nn.relu(Econv1_1)
        Econv1_2 = tf.layers.conv2d(Econv1_1, 32, 3, padding = 'same', name = 'conv_1_2', trainable = unfreeze)
        Econv1_2 = tf.layers.batch_normalization(Econv1_2, training = is_training, name = 'bn_1_2', trainable = unfreeze)
        Econv1_2 = tf.nn.relu(Econv1_2)

        pool1 = tf.layers.max_pooling2d(Econv1_2, (1, 2), strides = (1, 2), name = 'max_pool_1')
        #80x256

        Econv2_1 = tf.layers.conv2d(pool1, 64, 3, padding = 'same', name = 'conv_2_1', trainable = unfreeze)
        Econv2_1 = tf.layers.batch_normalization(Econv2_1, training = is_training, name = 'bn_2_1', trainable = unfreeze)
        Econv2_1 = tf.nn.relu(Econv2_1)
        Econv2_2 = tf.layers.conv2d(Econv2_1, 64, 3, padding = 'same', name = 'conv_2_2', trainable = unfreeze)
        Econv2_2 = tf.layers.batch_normalization(Econv2_2, training = is_training, name = 'bn_2_2', trainable = unfreeze)
        Econv2_2 = tf.nn.relu(Econv2_2)

        pool2 = tf.layers.max_pooling2d(Econv2_2, 2, strides = 2, name = 'max_pool_2')
        #40x128
        Econv3_1 = tf.layers.conv2d(pool2, 128, 3, padding = 'same', name = 'conv_3_1', trainable = unfreeze)
        Econv3_1 = tf.layers.batch_normalization(Econv3_1, training = is_training, name = 'bn_3_1', trainable = unfreeze)
        Econv3_1 = tf.nn.relu(Econv3_1)
        Econv3_2 = tf.layers.conv2d(Econv3_1, 128, 3, padding = 'same', name = 'conv_3_2', trainable = unfreeze)
        Econv3_2 = tf.layers.batch_normalization(Econv3_2, training = is_training, name = 'bn_3_2', trainable = unfreeze)
        Econv3_2 = tf.nn.relu(Econv3_2)

        pool3 = tf.layers.max_pooling2d(Econv3_2, (1, 2), strides = (1, 2), name = 'max_pool_3')
        #40x64
    return pool3, Econv1_2, Econv2_2, Econv3_2

def decoder_inference(inputs, Econv1_2, Econv2_2, Econv3_2, is_training = False, scope = 'decoder', unfreeze = True):
    with tf.variable_scope(scope):
        #40x64
        up1 = tf.layers.conv2d_transpose(inputs, 128, 3, strides=(1, 2), padding = 'same', name = 'upsample_1', trainable = unfreeze)
        up1 = tf.layers.batch_normalization(up1, training = is_training, name = 'bn_1', trainable = unfreeze)
        up1 = tf.nn.relu(up1)

        #40x128
        up1 = tf.concat([up1, Econv3_2], -1, name='merge_1')
        Upconv1_1 = tf.layers.conv2d(up1, 128, 3, padding = 'same', name = 'conv_1_1', trainable = unfreeze)
        Upconv1_1 = tf.layers.batch_normalization(Upconv1_1, training = is_training, name = 'bn_1_1', trainable = unfreeze)
        Upconv1_1 = tf.nn.relu(Upconv1_1)
        Upconv1_2 = tf.layers.conv2d(Upconv1_1, 128, 3, padding = 'same', name = 'conv_1_2', trainable = unfreeze)
        Upconv1_2 = tf.layers.batch_normalization(Upconv1_2, training = is_training, name = 'bn_1_2', trainable = unfreeze)
        Upconv1_2 = tf.nn.relu(Upconv1_2)

        up2 = tf.layers.conv2d_transpose(Upconv1_2, 128, 3, strides=(2, 2), padding = 'same', name = 'upsample_2', trainable = unfreeze)
        up2 = tf.layers.batch_normalization(up2, training = is_training, name = 'bn_2', trainable = unfreeze)
        up2 = tf.nn.relu(up2)

        #80x256
        up2 = tf.concat([up2, Econv2_2], -1, name='merge_2')
        Upconv2_1 = tf.layers.conv2d(up2, 64, 3, padding = 'same', name = 'conv_2_1', trainable = unfreeze)
        Upconv2_1 = tf.layers.batch_normalization(Upconv2_1, training = is_training, name = 'bn_2_1', trainable = unfreeze)
        Upconv2_1 = tf.nn.relu(Upconv2_1)
        Upconv2_2 = tf.layers.conv2d(Upconv2_1, 64, 3, padding = 'same', name = 'conv_2_2', trainable = unfreeze)
        Upconv2_2 = tf.layers.batch_normalization(Upconv2_2, training = is_training, name = 'bn_2_2', trainable = unfreeze)
        Upconv2_2 = tf.nn.relu(Upconv2_2)

        up3 = tf.layers.conv2d_transpose(Upconv2_2, 32, 3, strides=(1, 2), padding = 'same', name = 'upsample_3', trainable = unfreeze)
        up3 = tf.layers.batch_normalization(up3, training = is_training, name = 'bn_3', trainable = unfreeze)
        up3 = tf.nn.relu(up3)

        #80x512
        up3 = tf.concat([up3, Econv1_2], -1, name='merge_3')
        Upconv3_1 = tf.layers.conv2d(up3, 32, 3, padding = 'same', name = 'conv_3_1', trainable = unfreeze)
        Upconv3_1 = tf.layers.batch_normalization(Upconv3_1, training = is_training, name = 'bn_3_1', trainable = unfreeze)
        Upconv3_1 = tf.nn.relu(Upconv3_1)
        Upconv3_2 = tf.layers.conv2d(Upconv3_1, 32, 3, padding = 'same', name = 'conv_3_2', trainable = unfreeze)
        Upconv3_2 = tf.layers.batch_normalization(Upconv3_2, training = is_training, name = 'bn_3_2', trainable = unfreeze)
        Upconv3_2 = tf.nn.relu(Upconv3_2)

        decoded = tf.layers.conv2d(Upconv3_2, 1, 1, name = 'Final_layer', trainable = unfreeze)
    return decoded

