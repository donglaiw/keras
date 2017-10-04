import numpy as np
import pickle
import os,sys
from keras.models import Model,load_model
from keras.layers import Input, Conv3D, Deconv3DN, MaxPooling3D
from keras.layers.core import Lambda
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Cropping3D
from keras.applications.affinity_unet import get_unet, load_weights

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

sn=DATA_DIR+'../keras_test/test_malis.pkl'
sn2=DATA_DIR+'../keras_test/test_malis_keras.pkl'
if os.path.exists(sn):
    with open(sn,'rb') as fid:
        data=pickle.load(fid)
else:
    try:
        data= np.random.rand(1,1,31,204,204).astype(np.float32) 
        with open(sn,'wb') as fid:
            pickle.dump(data,fid)
    except:
        # Get consistent pseudo-random data
        data = np.random.RandomState(1234).rand(1, 1, 31, 204, 204).astype(np.float32)

# benchmark speed
import time
data2=np.tile(data,(10,1,1,1,1))
#data2=data
start = time.time()
for i in range(10):
    res=net.predict(data2)
end = time.time()
print end-start
