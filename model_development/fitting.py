from model_development.my_model import create_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from model_development.data_managing import create_train_vali_gens
from tensorflow.keras.models import model_from_json

ver = 1
MODEL_PATH = f'archive/model/model/{ver}'
IMAGE_PATH = f'model_development/archive/images/{ver}.1.png'
ACC_PATH, LOSS_PATH = f'model_development/archive/histories/acc{ver}.1.png', f'model_development/archive/histories/loss{ver}.1.png'

def fit_model(model : Sequential, train_vali_gens , epochs=60, extra_callbacks=[]):
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                patience=2, min_lr=0.00001, mode='auto')

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    
    callbacks = [reduce_lr, early_stopping] + extra_callbacks

    train ,val =train_vali_gens
    return model.fit(
        train,
        verbose=1,
        epochs=epochs,
        validation_data = val,
        callbacks=callbacks,
        shuffle=True,
    )

def save_hist_meter(hist, meter, path):
    # summarize history for accuracy
    plt.plot(hist.history[meter])
    plt.plot(hist.history[f'val_{meter}'])
    plt.title(f'model {meter}')
    plt.ylabel(f'{meter}')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(path)
    plt.clf()

def save_image_history_model_weights(model, history):
    model.save(MODEL_PATH)
    plot_model(model, to_file=IMAGE_PATH, show_shapes=True, show_layer_names=True)
    
    save_hist_meter(hist=history, meter='accuracy', path=ACC_PATH)
    save_hist_meter(hist=history, meter='loss', path=LOSS_PATH)

def load_model(json_path, h5_path):
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(h5_path)
    return loaded_model
def main():
    model = create_model()
    
    data = create_train_vali_gens(64)

    history = fit_model(model, data, epochs=100)

    save_image_history_model_weights(model, history=history)
    
if __name__ == '__main__':
    main()