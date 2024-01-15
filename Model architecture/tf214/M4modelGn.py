from tensorflow import keras
from keras.layers import *
from keras.models import Model

def build_model(input_shape=(160, 192, 128, 1), output_channels=6):
    inputs = Input(input_shape)
    conv01 = Conv3D(16,kernel_size=(3,3,1),strides=(1,1,1),padding='same',activation='relu',name='conv01')(inputs)
    conv01 = GroupNormalization(groups=8, axis=-1, name='GroupNorm01')(conv01)
    conv02 = Conv3D(16,kernel_size=(1,1,3),strides=(1,1,1),padding='same',activation='relu',name='conv02')(inputs)
    conv02 = GroupNormalization(groups=8, axis=-1, name='GroupNorm02')(conv02)

    conv11 = Conv3D(16,kernel_size=(3,3,1),strides=(1,1,1),padding='same',activation='relu',name='conv11')(conv01)
    conv11 = GroupNormalization(groups=8, axis=-1, name='GroupNorm11')(conv11)
    conv12 = Conv3D(16,kernel_size=(1,1,3),strides=(1,1,1),padding='same',activation='relu',name='conv12')(conv02)
    conv12 = GroupNormalization(groups=8, axis=-1, name='GroupNorm12')(conv12)

    conv21 = MaxPooling3D(pool_size=(2,2,1),strides=(2,2,1),padding='same')(conv11)
    conv21 = Conv3D(32,kernel_size=(3,3,1),strides=(1,1,1),padding='same',activation='relu',name='conv21')(conv21)
    conv21 = GroupNormalization(groups=8, axis=-1, name='GroupNorm21')(conv21)
    conv22 = MaxPooling3D(pool_size=(2,2,1),strides=(2,2,1),padding='same')(conv12)
    conv22 = Conv3D(32,kernel_size=(1,1,3),strides=(1,1,1),padding='same',activation='relu',name='conv22')(conv22)
    conv22 = GroupNormalization(groups=8, axis=-1, name='GroupNorm22')(conv22)

    conv31 = Conv3D(32,kernel_size=(3,3,1),strides=(1,1,1),padding='same',activation='relu',name='conv31')(conv21)
    conv31 = GroupNormalization(groups=8, axis=-1, name='GroupNorm31')(conv31)
    conv32 = Conv3D(32,kernel_size=(1,1,3),strides=(1,1,1),padding='same',activation='relu',name='conv32')(conv22)
    conv32 = GroupNormalization(groups=8, axis=-1, name='GroupNorm32')(conv32)

    convc = concatenate([MaxPooling3D(pool_size=(2,2,1),strides=(2,2,1),padding='same')(conv31),MaxPooling3D(pool_size=(2,2,1),strides=(2,2,1),padding='same')(conv32)],axis=-1)
    convc = Conv3D(64,kernel_size=(3,3,3),strides=(1,1,1),padding='same',activation='relu',name='convc')(convc)
    convc = GroupNormalization(groups=8, axis=-1, name='GroupNormc')(convc)

    convc1 = Conv3D(32,kernel_size=(3,3,3),strides=(1,1,1),padding='same',activation='relu',name='convc1')(convc)
    convc1 = GroupNormalization(groups=8, axis=-1, name='GroupNormc1')(convc1)

    conv4 = Conv3DTranspose(16,kernel_size=(3,3,3),strides=(2,2,1),padding='same',activation='relu',name='conv4')(convc1)
    conv4 = GroupNormalization(groups=8, axis=-1, name='GroupNorm4')(conv4)
    conv4 = concatenate([Conv3D(32, kernel_size=(1,1,1),strides=(1,1,1),padding='same',name='conv4c')(concatenate([conv31,conv32],axis=-1)),conv4],axis=-1)

    conv5 = Conv3D(24,kernel_size=(3,3,3),strides=(1,1,1),padding='same',activation='relu',name='conv5')(conv4)
    conv5 = GroupNormalization(groups=8, axis=-1, name='GroupNorm5')(conv5)

    conv6 = Conv3DTranspose(16,kernel_size=(3,3,3),strides=(2,2,1),padding='same',activation='relu',name='conv6')(conv5)            
    conv6 = GroupNormalization(groups=8, axis=-1, name='GroupNorm6')(conv6)
    conv6 = concatenate([Conv3D(16, kernel_size=(1,1,1),strides=(1,1,1),padding='same',name='conv6c')(concatenate([conv11,conv12],axis=-1)),conv6],axis=-1)

    conv7 = Conv3D(16,kernel_size=(3,3,3),strides=(1,1,1),padding='same',activation='relu',name='conv7')(conv6)
    conv7 = GroupNormalization(groups=8, axis=-1, name='GroupNorm7')(conv7)

    ### Output Block
    output = Conv3D(
        filters=output_channels,  # No. of tumor classes
        kernel_size=(1, 1, 1),
        strides=1,                
        name='output',
        activation='sigmoid')(conv7)
    model = Model(inputs, outputs=output)  # Create the model
    return model
