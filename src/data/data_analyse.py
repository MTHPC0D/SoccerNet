import os
import pandas as pd
import matplotlib.pyplot as plt

formatted_dir = 'datasets/number_dataset'

def count_images_in_dirs(base_dir):
    data = []
    for split in ['train', 'test']:
        for class_dir in os.listdir(os.path.join(base_dir, split)):
            class_path = os.path.join(base_dir, split, class_dir)
            if os.path.isdir(class_path):
                num_images = len(os.listdir(class_path))
                if num_images > 0:  
                    data.append({'split': split, 'class': int(class_dir), 'num_images': num_images})
    return pd.DataFrame(data)

df = count_images_in_dirs(formatted_dir)

# Sort the DataFrame by class
df = df.sort_values(by=['split', 'class'])

# Display the first few rows of the DataFrame
print(df.head())

# Visualize the distribution of images per class for training and testing
plt.figure(figsize=(14, 7))

# Class distribution for train
plt.subplot(1, 2, 1)
df_train = df[df['split'] == 'train']
plt.bar(df_train['class'].astype(str), df_train['num_images'], color='blue')
plt.xlabel('Class')
plt.ylabel('Number of images')
plt.title('Distribution of images per class (Train)')
plt.xticks(rotation=90)

# Class distribution for test
plt.subplot(1, 2, 2)
df_test = df[df['split'] == 'test']
plt.bar(df_test['class'].astype(str), df_test['num_images'], color='green')
plt.xlabel('Class')
plt.ylabel('Number of images')
plt.title('Distribution of images per class (Test)')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()
