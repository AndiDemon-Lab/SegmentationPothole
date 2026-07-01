import pandas as pd
import matplotlib.pyplot as plt

# Baca data training log
df = pd.read_csv('../result_maskrcnn/train_log.csv')

# Plot loss
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['train_loss'], marker='o', linestyle='-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss per Epoch')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Simpan plot
plt.savefig('../result_maskrcnn/loss_plot.png', dpi=300, bbox_inches='tight')
print("Plot saved to: result_maskrcnn/loss_plot.png")

# Tampilkan plot
plt.show()
