import matplotlib.pyplot as plt
import pickle

def read(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

ta = read('results/train_accuracy.pkl')
tl = read('results/train_loss.pkl')
va = read('results/val_accuracy.pkl')
vl = read('results/val_loss.pkl')

plt.subplot(1,2,1)
plt.grid()
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
tl, = plt.plot(tl, label='Train Loss')
vl, = plt.plot(vl, label='Val Loss')
plt.legend(handles=[tl, vl]) 


plt.subplot(1,2,2)
plt.grid()
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
ta, = plt.plot(ta, label='Train Acc')
va, = plt.plot(va, label='Val Acc')
plt.legend(handles=[ta, va])
 
plt.show()
