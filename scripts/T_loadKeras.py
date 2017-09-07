import numpy as np
import pickle
import os,sys
from keras.models import Model,load_model
from keras.layers import Input, Conv3D, Deconv3DN, MaxPooling3D
from keras.layers.core import Lambda
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Cropping3D

# caffe, unlike theano, does correlation not convolution. We need
# to flip the weights 180 deg
# Keras needs h*w*i*o filters (where d is input, o is output), so we transpose

DATA_DIR='/n/coxfs01/donglai/malis_trans/'

def wT(ww):#weight transformation
    ww2 = ww.transpose((2,3,4,1,0))
    # reverse matrix [keep batch/channel]
    return np.flip(np.flip(np.flip(ww2,0),1),2)

def load_weights(model, weights):
    for layer in model.layers:
        # TODO: add a check to make sure we're not jumping over any layers with
        # trainable weights
        if layer.name in weights:
            print('Copying weights for %s' % layer.name)
            ww = weights[layer.name]
            #pp=model.get_layer(layer.name).get_weights();import pdb; pdb.set_trace()
            if 'b' in ww:
                model.get_layer(layer.name).set_weights([wT(ww['w']),ww['b']])
            else:
                model.get_layer(layer.name).set_weights([wT(ww['w'])])
        elif not layer.trainable_weights:
            # this is fine; we don't expect weights
            print('No weights for untrainable layer %s' % layer.name)
        else:
            # this isn't fine; if there are trainable weights, they should
            # probably be in the param file
            print('(!!) No weights for trainable layer %s, but it should have '
                  'weights (?!). Does the parameter file match the .prototxt?'
                  % layer.name)
def cLayer(fNum,fSize,index,opt,inputs,strides=None,num_groups=1, activation=None):
    if opt==1:#conv
        return Conv3D(fNum, fSize, padding='valid', data_format='channels_first', name='Convolution'+str(index), activation=activation)(inputs)
    else: #deconv
        # need group deconvolution
        #return Deconv3D(fNum, fSize, strides=strides, padding='valid', data_format='channels_first', use_bias=False, name='Deconvolution'+str(index))(inputs)
        return Deconv3DN(fNum, fSize, strides=strides, padding='valid', data_format='channels_first', use_bias=False, num_groups=num_groups, name='Deconvolution'+str(index))(inputs)
     
def get_unet(img_rows=204,img_cols=204,img_deps=31, img_chans=1):
    # caffe data: batch*channel*depth*rows*cols
    inputs = Input((img_chans, img_deps,img_rows, img_cols))
    conv1 = cLayer(24, (3, 3, 3), 1, 1, inputs)
    relu1 = LeakyReLU(0.005)(conv1)
    conv2 = cLayer(24, (3, 3, 3), 2, 1, relu1)
    relu2 = LeakyReLU(0.005)(conv2)
    pool1 = MaxPooling3D(pool_size=(1, 2, 2),data_format='channels_first', strides=(1, 2, 2))(relu2)

    conv3 = cLayer(72, (3, 3, 3), 3, 1, pool1)
    relu3 = LeakyReLU(0.005)(conv3)
    conv4 = cLayer(72, (3, 3, 3), 4, 1, relu3)
    relu4 = LeakyReLU(0.005)(conv4)
    pool2 = MaxPooling3D(pool_size=(1, 2, 2),data_format='channels_first', strides=(1, 2, 2))(relu4)

    conv5 = cLayer(216, (3, 3, 3), 5, 1, pool2)
    relu5 = LeakyReLU(0.005)(conv5)
    conv6 = cLayer(216, (3, 3, 3), 6, 1, relu5)
    relu6 = LeakyReLU(0.005)(conv6)
    pool3 = MaxPooling3D(pool_size=(1, 2, 2),data_format='channels_first', strides=(1, 2, 2))(relu6)
   
    conv7 = cLayer(648, (3, 3, 3), 7, 1, pool3)
    relu7 = LeakyReLU(0.005)(conv7)
    conv8 = cLayer(648, (3, 3, 3), 8, 1, relu7)
    relu8 = LeakyReLU(0.005)(conv8)

    dconv1 = cLayer(1, (1, 2, 2), 1, 2, relu8, strides=(1, 2, 2), num_groups=648)
    conv9 = cLayer(216, (1, 1, 1), 9, 1, dconv1)
    relu6_crop = Cropping3D(cropping=(2,4,4),data_format='channels_first')(relu6)
    mc1 = concatenate([conv9, relu6_crop], axis=1)
    conv10 = cLayer(216, (3, 3, 3), 10, 1, mc1)
    relu9 = LeakyReLU(0.005)(conv10)
    conv11 = cLayer(216, (3, 3, 3), 11, 1, relu9)
    relu10 = LeakyReLU(0.005)(conv11)

    dconv2 = cLayer(1, (1, 2, 2), 2, 2, relu10, strides=(1, 2, 2), num_groups=216)
    conv12 = cLayer(72, (1, 1, 1), 12, 1, dconv2)
    relu4_crop = Cropping3D(cropping=(6,16,16),data_format='channels_first')(relu4)
    mc2 = concatenate([conv12, relu4_crop], axis=1)
    conv13 = cLayer(72, (3, 3, 3), 13, 1, mc2)
    relu11 = LeakyReLU(0.005)(conv13)
    conv14 = cLayer(72, (3, 3, 3), 14, 1, relu11)
    relu12 = LeakyReLU(0.005)(conv14)
 
    dconv3 = cLayer(1, (1, 2, 2), 3, 2, relu12, strides=(1, 2, 2), num_groups=72)
    conv15 = cLayer(24, (1, 1, 1), 15, 1, dconv3)
    relu2_crop = Cropping3D(cropping=(10,40,40),data_format='channels_first')(relu2)
    mc3 = concatenate([conv15, relu2_crop], axis=1)
    conv16 = cLayer(24, (3, 3, 3), 16, 1, mc3)
    relu13 = LeakyReLU(0.005)(conv16)
    conv17 = cLayer(24, (3, 3, 3), 17, 1, relu13)
    relu14 = LeakyReLU(0.005)(conv17)   

    #conv18 = cLayer(3, (1, 1, 1), 18, 1, relu14)
    conv18 = cLayer(3, (1, 1, 1), 18, 1, relu14, activation='sigmoid')

    model = Model(inputs=[inputs], outputs=[conv18])
    return model

# construct model
mn = 'malis_keras.pkl'
# h5: need to modify code
# pkl: too many recursions
if False and os.path.exists(mn):
    #net=load_model(mn)
    net=pickle.load(open(mn,'rb'))
else:
    net=get_unet(img_rows=204,img_cols=204,img_deps=31, img_chans=1)
    # load weight
    import pickle
    with open(DATA_DIR+'net_weight.pkl','rb') as fid:
        ww=pickle.load(fid)
    # set weight
    load_weights(net, ww)
    #net.save('malis_keras.h5')
    #pickle.dump(net,open(mn,'wb'))

sn=DATA_DIR+'test/test_malis.pkl'
sn2=DATA_DIR+'test/test_malis_keras.pkl'
if os.path.exists(sn):
    with open(sn,'rb') as fid:
        data=pickle.load(fid)
else:
    data= np.random.rand(1,1,31,204,204).astype(np.float32) 
    with open(sn,'wb') as fid:
        pickle.dump(data,fid)

# benchmark speed
import time
data2=np.tile(data,(10,1,1,1,1))
#data2=data
start = time.time()
for i in range(10):
    res=net.predict(data2)
end = time.time()
print end-start
