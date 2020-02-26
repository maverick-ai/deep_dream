from keras.applications import inception_v3
from keras import backend as K
K.set_learning_phase(0)
model=inception_v3.InceptionV3(weights='imagenet',include_top=False)
layer_contribution={'mixed2':0.2,'mixed3':3.,'mixed4':2.,'mixed5':1.5}
layer_dict=dict([(layer.name,layer) for layer in model.layers])
loss=K.variable(0.)
for layer_name in layer_contribution:
    coeff=layer_contribution[layer_name]
    activation=layer_dict[layer_name].output
    scalling=K.prod(K.cast(K.shape(activation),'float32'))
    loss+=coeff*K.sum(K.square(activation[:,2:-2,2:-2,:]))/scalling
dream=model.input
grads=K.gradients(loss,dream)[0]
grads/=K.maximum(K.mean(K.abs(grads)),1e-7)
outputs=[loss,grads]
fetch_loss_and_grads=K.function([dream],outputs)
def eval_loss_and_grads(x):
    outs=fetch_loss_and_grads([x])
    loss_value=outs[0]
    grad_value=outs[1]
    return loss_value,grad_value
def gradient_ascent(x,iterations,step,max_loss=None):
    for i in range(iterations):
        loss_value,grad_value=eval_loss_and_grads(x)
        if max_loss is not None and loss_value>max_loss:
            break
        print('.... Loss value at ',i,':',loss_value)
        x+=step*grad_value
    return x
import numpy as np
import scipy
from keras.preprocessing import image
def resize_img(img,size):
    img=np.copy(img)
    factors=(1,float(size[0])/img.shape[1],float(size[1])/img.shape[2],1)
    return scipy.ndimage.zoom(img,factors,order=1)
def save_img(img,fname):
    pil_img=deprocess_image(np.copy(img))
    scipy.misc.imsave(fname,pil_img)
def preprocess_image(image_path):
    img=image.load_img(image_path)
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    img=inception_v3.preprocess_input(img)
    return img
def deprocess_image(x):
    if K.image_data_format()=='channels_first':
        x=x.reshape((3,x.shape[2],x.shape[3]))
        x=x.transpose((1,2,0))
    else:
        x=x.reshape((x.shape[1],x.shape[2],3))
    x/=2.
    x+=0.5
    x*=255
    x=np.clip(x,0,255).astype('uint8')
    return x
step=0.01
num_octave=3
octave_scale=1.4
iterations=20
max_loss=10
base_image_path=r'C:\Users\sbans\Desktop\lucy.jpg'
img=preprocess_image(base_image_path)
orignal_shape=img.shape[1:3]
successive_shapes=[orignal_shape]
for i in range(1,num_octave):
    shape=tuple([int(dim/(octave_scale**i)) for dim in orignal_shape])
    successive_shapes.append(shape)
successive_shapes=successive_shapes[::-1]
orignal_img=np.copy(img)
shrunk_orignal_img=resize_img(img,successive_shapes[0])
for shape in successive_shapes:
    print('Processing image shape',shape)
    img=resize_img(img,shape)
    img=gradient_ascent(img,iterations=iterations,step=step,max_loss=max_loss)
    upscales_shrunk_orignal_img=resize_img(shrunk_orignal_img,shape)
    same_size_orignal=resize_img(orignal_img,shape)
    loss_details=same_size_orignal-upscales_shrunk_orignal_img
    img=+loss_details
    shrunk_orignal_img=resize_img(orignal_img,shape)
    save_img(img,fname='dream_at_scale_'+str(shape)+'.png')
save_img(img,fname='final_dream.png')
    
    