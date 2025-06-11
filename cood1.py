import os
import random
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# === 1. 本地資料集路徑設定 ===
SOURCE_DIR = r"C:\flower_project\flower_photos"
DEST_DIR = r"C:\flower_project\flower_photos_split"
SPLITS = {'train': 0.7, 'val': 0.2, 'test': 0.1}


#強制使用CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# === 2. 分割資料集 ===
for split in SPLITS:
    for class_name in os.listdir(SOURCE_DIR):
        src_path = os.path.join(SOURCE_DIR, class_name)
        if os.path.isdir(src_path):
            Path(os.path.join(DEST_DIR, split, class_name)).mkdir(parents=True, exist_ok=True)

for class_name in os.listdir(SOURCE_DIR):
    src_class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(src_class_path):
        continue

    images = [f for f in os.listdir(src_class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)

    total = len(images)
    train_end = int(total * SPLITS['train'])
    val_end = train_end + int(total * SPLITS['val'])

    for i, img in enumerate(images):
        src = os.path.join(src_class_path, img)
        if i < train_end:
            dst = os.path.join(DEST_DIR, 'train', class_name, img)
        elif i < val_end:
            dst = os.path.join(DEST_DIR, 'val', class_name, img)
        else:
            dst = os.path.join(DEST_DIR, 'test', class_name, img)
        shutil.copy2(src, dst)

print("✅ 資料分割完成！")

# === 3. 載入與預處理資料 ===
IMAGE_SIZE = (180, 180)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

def preprocess(ds, augment=False):
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=AUTOTUNE)
    if augment:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1)
        ])
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DEST_DIR, 'train'),
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DEST_DIR, 'val'),
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DEST_DIR, 'test'),
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
train_ds = preprocess(train_ds, augment=True)
val_ds = preprocess(val_ds)
test_ds = preprocess(test_ds)

# === 4. 建立模型（使用 MobileNetV2） ===
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMAGE_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# === 5. 訓練模型 ===
early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[early_stop]
)

# 找出驗證準確率最高的 epoch
best_epoch = np.argmax(history.history['val_accuracy'])

# 擷取最佳 epoch 對應的指標
best_train_acc = history.history['accuracy'][best_epoch]
best_train_loss = history.history['loss'][best_epoch]
best_val_acc = history.history['val_accuracy'][best_epoch]
best_val_loss = history.history['val_loss'][best_epoch]

# 輸出表格格式
print("\n📊 關鍵指標表格（最佳 Epoch）:")
print(f"| Epoch | 訓練準確率 | 驗證準確率 | 訓練損失 | 驗證損失 |")
print(f"|--------|------------|------------|-----------|-----------|")
print(f"| {best_epoch+1:^6} | {best_train_acc:.4f}     | {best_val_acc:.4f}     | {best_train_loss:.4f}  | {best_val_loss:.4f}  |")

# === 顯示訓練過程平均結果（中文） ===
import numpy as np

avg_train_acc = np.mean(history.history['accuracy'])
avg_train_loss = np.mean(history.history['loss'])
avg_val_acc = np.mean(history.history['val_accuracy'])
avg_val_loss = np.mean(history.history['val_loss'])

print(f"\n✅ 訓練過程平均結果：")
print(f"訓練準確率：{avg_train_acc:.4f}，訓練損失：{avg_train_loss:.4f}")
print(f"驗證準確率：{avg_val_acc:.4f}，驗證損失：{avg_val_loss:.4f}")

# === 6. 訓練曲線視覺化 ===
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('準確率')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('損失')
plt.show()

# === 7. 測試集評估 ===
test_loss, test_acc = model.evaluate(test_ds)
print(f"\n✅ 測試集準確率：{test_acc:.4f}")

# === 8. 混淆矩陣 ===
y_true, y_pred = [], []
for images, labels in test_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp.plot(xticks_rotation=45)
plt.title("混淆矩陣")
plt.show()
#生成類別別精確率
from sklearn.metrics import classification_report

# 預測與真實標籤已在之前定義：y_true, y_pred
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(report)

# ===  隨機抽取測試樣本展示（正確/錯誤各5）

import numpy as np
import matplotlib.pyplot as plt

def visualize_correct_wrong_samples(y_true, y_pred, images, labels, class_names, num_samples=5):
    correct_idx = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t == p]
    wrong_idx = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p]

    correct_samples = np.random.choice(correct_idx, min(num_samples, len(correct_idx)), replace=False)
    wrong_samples = np.random.choice(wrong_idx, min(num_samples, len(wrong_idx)), replace=False)

    plt.figure(figsize=(15, 6))

    for i, idx in enumerate(correct_samples):
        ax = plt.subplot(2, num_samples, i + 1)
        plt.imshow(images[idx])
        plt.title(f"正確\n{class_names[y_true[idx]]}")
        plt.axis('off')

    for i, idx in enumerate(wrong_samples):
        ax = plt.subplot(2, num_samples, num_samples + i + 1)
        plt.imshow(images[idx])
        plt.title(f"錯誤\n真實: {class_names[y_true[idx]]}\n預測: {class_names[y_pred[idx]]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# 收集所有測試圖片與標籤
all_images = []
all_labels = []

for batch_images, batch_labels in test_ds:
    all_images.extend(batch_images.numpy())
    all_labels.extend(batch_labels.numpy())

all_images = np.array(all_images)
all_labels = np.array(all_labels)

visualize_correct_wrong_samples(y_true, y_pred, all_images, all_labels, class_names)

# === 9. 測試圖像預測可視化 ===
def visualize_predictions():
    for images, labels in test_ds.take(1):
        preds = model.predict(images)
        pred_labels = np.argmax(preds, axis=1)
        plt.figure(figsize=(15, 5))
        for i in range(10):
            ax = plt.subplot(2, 5, i + 1)
            plt.imshow(images[i].numpy())
            plt.title(f"預測: {class_names[pred_labels[i]]}\n實際: {class_names[labels[i]]}")
            plt.axis("off")
        plt.tight_layout()
        plt.show()

visualize_predictions()

# === 10. 儲存模型 ===
import os

# 自動取得目前的 .py 檔名（不含副檔名）
script_name = os.path.splitext(os.path.basename(__file__))[0]

# 建立儲存路徑
model_path = os.path.join("models", f"{script_name}.h5")

# 儲存模型
model.save(model_path)
print(f"✅ 模型已儲存為：{model_path}")
