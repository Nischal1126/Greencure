import pandas as pd
import matplotlib.pyplot as plt
import os 

save_dir = "/teamspace/studios/this_studio/plots"
data_path = "/teamspace/studios/this_studio/checkpoints/training_log.csv"
os.makedirs(save_dir, exist_ok=True)

df = pd.read_csv(data_path)

#Loss curve
plt.figure(figsize=(8, 5))
plt.plot(df['train_loss'], label = "trainig loss")
plt.plot(df['val_loss'], label = "validation loss")

plt.xticks(df['epoch'][::5]) 
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=300)


#Accuracy curve
plt.figure(figsize=(8, 5))

plt.plot(df['epoch'], df['train_acc'], label="Training Accuracy")
plt.plot(df['epoch'], df['val_acc'], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()


plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=300)

print("succesfully saved the curves")


