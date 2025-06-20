import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def metrics(history):
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoka')
    plt.ylabel('Loss')
    plt.title('Historia uczenia modelu')
    plt.legend()
    plt.show()