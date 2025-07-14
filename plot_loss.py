import matplotlib.pyplot as plt

# Extracting data from your provided output
epochs = range(1, 11)  # 10 epochs
train_losses = [
  2.8349,
  2.8247,
  2.8218,
  2.8180,
  2.8140,
  2.8129,
  2.8106,
  2.8095,
  2.8090,
  2.8074,
]
val_losses = [
  2.8213,
  2.8130,
  2.8124,
  2.8094,
  2.8110,
  2.8075,
  2.8077,
  2.8055,
  2.8049,
  2.8036,
]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label="Average Training Loss", marker="o")
plt.plot(epochs, val_losses, label="Average Validation Loss", marker="x")

# Add titles and labels
plt.title("Training and Validation Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(epochs)  # Ensure all epochs are shown on the x-axis
plt.grid(True)
plt.legend()
plt.tight_layout()  # Adjust layout to prevent labels from overlapping

# Display the plot
plt.show()
