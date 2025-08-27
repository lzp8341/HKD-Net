# coding=utf-8
# 教师模型保留所有（频域和空间域）特征提取模块，但用简单拼接代替动态特征融合。
import tensorflow as tf

# 获取所有 GPU
gpus = tf.config.list_physical_devices('GPU')
# 只使用第 1 张 GPU（ID=1）
if len(gpus) >= 2:
    tf.config.set_visible_devices(gpus[1], 'GPU')
    print("已强制使用 GPU 1")
else:
    print("⚠️ 系统中不足 2 张 GPU")

from tensorflow.keras import layers, models, callbacks
import pathlib
from tqdm import tqdm
import datetime
import numpy as np
import time
import json
import os
import matplotlib.pyplot as plt
import cv2  # 用于创建频带掩码
import seaborn as sns
# from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import random
import pywt
# tf.config.run_functions_eagerly(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 关闭所有TensorFlow输出

# 固定随机种子
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

class Config:
    IMAGE_SIZE = (40, 40)
    CHANNELS = 1  
    NUM_CLASSES = 8
    NUM_SAMPLES = 1000  
    BATCH_SIZE = 128
    EPOCHS = 20
    TEACHER_EPOCHS = 20  
    TEMPORAL_DIM = 64  
    LEARNING_RATE = 0.001
    TEMPERATURE = 5.0
    ALPHA = 0.5
    EVAL_PLOT_PATH = "./model_save/eval_plots/"  
    TEST_BATCH_SIZE = 64  
    LOG_DIR = "model_save/logs/fit/"
    MODEL_SAVE_PATH = "./model_save/model_save/"


class Utils:
    @staticmethod
    def setup_gpu():
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.run_functions_eagerly(True)

    @staticmethod
    def create_tensorboard(log_suffix=""):
        log_dir = Config.LOG_DIR + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + log_suffix
        return tf.summary.create_file_writer(log_dir)



def _onehot_to_label(y):
    return np.argmax(y, axis=1) if y.ndim > 1 else y

def precision_score(y_true, y_pred, average='weighted'):
    y_true = _onehot_to_label(np.array(y_true))
    y_pred = _onehot_to_label(np.array(y_pred))
    cm = tf.math.confusion_matrix(y_true, y_pred).numpy()
    precisions = cm.diagonal() / cm.sum(axis=0).clip(min=1)
    if average == 'weighted':
        weights = cm.sum(axis=1)
        return np.average(precisions, weights=weights)
    return precisions.mean()

def recall_score(y_true, y_pred, average='weighted'):
    y_true = _onehot_to_label(np.array(y_true))
    y_pred = _onehot_to_label(np.array(y_pred))
    cm = tf.math.confusion_matrix(y_true, y_pred).numpy()
    recalls = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    if average == 'weighted':
        weights = cm.sum(axis=1)
        return np.average(recalls, weights=weights)
    return recalls.mean()

def f1_score(y_true, y_pred, average='weighted'):
    p = precision_score(y_true, y_pred, average=average)
    r = recall_score(y_true, y_pred, average=average)
    return 2 * p * r / (p + r + 1e-8)

def confusion_matrix(y_true, y_pred):
    y_true = _onehot_to_label(np.array(y_true))
    y_pred = _onehot_to_label(np.array(y_pred))
    return tf.math.confusion_matrix(y_true, y_pred).numpy().tolist()


class ModelEvaluator:
    def __init__(self):
        os.makedirs(Config.EVAL_PLOT_PATH, exist_ok=True)

    @staticmethod
    def compute_metrics(model, y_true, y_pred, class_names):
        y_true = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)
        
        metrics = {
            'accuracy': tf.keras.metrics.CategoricalAccuracy()(y_true, y_pred).numpy(),
            'precision': precision_score(y_true, y_pred_labels, average='weighted'),
            'recall': recall_score(y_true, y_pred_labels, average='weighted'),
            'f1': f1_score(y_true, y_pred_labels, average='weighted'),
            'detection_rate': recall_score(y_true, y_pred_labels, average='micro')}
        
        dummy_input = np.random.rand(1, *Config.IMAGE_SIZE, Config.CHANNELS)
        start_time = time.time()
        for _ in range(100):
            _ = model.predict(dummy_input)
        metrics['inference_time'] = (time.time() - start_time) / 100
        
        return metrics

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names, filename='confusion_matrix.png'):
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(Config.EVAL_PLOT_PATH, filename))
        plt.close()
    
    def full_evaluation(self, model, test_ds, class_names, model_name="model"):
        """执行完整评估流程"""
        y_true, y_pred = [], []  
        for x, y in test_ds:
            y_true.extend(y.numpy())
            predictions = model.predict(x, verbose=0)  
            if isinstance(predictions, dict):  
                y_pred.extend(predictions['logits'])
            elif isinstance(predictions, list):  
                y_pred.extend(predictions[0])
            else:
                raise ValueError("未知的模型预测返回值类型")
        metrics = self.compute_metrics(model, np.array(y_true), np.array(y_pred), class_names)  
        self.plot_confusion_matrix(np.array(y_true), np.array(y_pred), class_names, filename=f'{model_name}_confusion_matrix.png')  
        return metrics



class DataLoader:
    def __init__(self):
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

    @staticmethod
    def load_and_preprocess_image(path, size=Config.IMAGE_SIZE):
        image = tf.io.read_file(path)
        image = tf.io.decode_png(image, channels=Config.CHANNELS)  
        image = tf.image.resize(image, size) / 255.
        return image

    def load_dataset(self, path, batch_size=Config.BATCH_SIZE):
        data_root = pathlib.Path(path)
        all_image_paths = list(data_root.glob('*/*'))
        all_image_paths = [str(path) for path in all_image_paths]
        
        label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
        label_to_index = {label: idx for idx, label in enumerate(label_names)}
        all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
        print("NUM_CLASSES Mapping:", label_to_index)  

        image_ds = tf.data.Dataset.from_tensor_slices(all_image_paths).map(
            self.load_and_preprocess_image, num_parallel_calls=self.AUTOTUNE)
        label_ds = tf.data.Dataset.from_tensor_slices(tf.keras.utils.to_categorical(all_image_labels))
        dataset = tf.data.Dataset.zip((image_ds, label_ds))
        dataset = dataset.shuffle(len(all_image_paths)).batch(batch_size).prefetch(self.AUTOTUNE)  
        for image, label in dataset.take(1):                                                       
            print("dasdad Batch of images shape:", image.shape)                                    
            print("iuklhk Batch of labels shape:", label.shape)                                      
        return dataset



class ModelBuilder:
    # -----------------------------------------
    # 纺锤形MLP来变换教师和学生的特征，使其维度一致
    # -----------------------------------------
    @staticmethod
    def create_spindle_mlp(input_dim, output_dim, name="spindle_mlp"):
        model = models.Sequential(name=name)
        model.add(layers.Dense(input_dim, activation='relu', name=f'{name}_dense1'))
        model.add(layers.BatchNormalization(name=f'{name}_bn1'))
        model.add(layers.Dense(max(input_dim // 2, 32), activation='relu', name=f'{name}_dense2')) 
        model.add(layers.BatchNormalization(name=f'{name}_bn2'))
        model.add(layers.Dense(output_dim, name=f'{name}_output'))
        return model

    # ------------------------------
    # 改进的空间域特征提取 (残差+SE注意力)
    # ------------------------------
    @staticmethod
    def residual_block(x, filters, kernel_size=(3,3), dropout_rate=0.0, prefix=''):
        shortcut = x
        x = layers.Conv2D(filters, kernel_size, padding='same', activation=None, 
                        kernel_initializer='he_normal', name=f'{prefix}_conv1')(x)
        x = layers.BatchNormalization(name=f'{prefix}_bn1')(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(filters, kernel_size, padding='same', activation=None,
                        kernel_initializer='he_normal', name=f'{prefix}_conv2')(x)
        x = layers.BatchNormalization(name=f'{prefix}_bn2')(x)
        
        se = layers.GlobalAvgPool2D()(x)                       
        se = layers.Dense(filters//16, activation='relu')(se)  
        se = layers.Dense(filters, activation='sigmoid')(se)   
        x = layers.multiply([x, se])                           
        
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1,1), padding='same')(shortcut)
        x = layers.add([shortcut, x])
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Dropout(dropout_rate)(x)
        return x
    # -------------------------------------------------------------------------------------------------------------------------------------------------------

    # ----------------------------------------
    # 定义一维 Vision Transformer（1D-ViT）模块
    # ----------------------------------------
    @staticmethod
    def create_vit1d(input_shape, num_heads=2, embed_dim=32, ff_dim=64, num_transformer_blocks=1):
        inputs = layers.Input(shape=input_shape)
        x = layers.Dense(embed_dim)(inputs)
        
        positions = tf.range(start=0, limit=input_shape[0], delta=1)
        position_embedding = layers.Embedding(input_dim=input_shape[0], output_dim=embed_dim)(positions)
        x = x + position_embedding  
        
        for _ in range(num_transformer_blocks):
            x_norm = layers.LayerNormalization(epsilon=1e-6)(x)     
            attn_output = layers.MultiHeadAttention(
                num_heads=num_heads, 
                key_dim=embed_dim//num_heads,
                attention_axes=(1,)  
            )(x_norm, x_norm)
            x = layers.Add()([x, attn_output])
            x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
            ffn_output = layers.Dense(ff_dim, activation='relu')(x_norm)
            ffn_output = layers.Dense(embed_dim)(ffn_output)
            x = layers.Add()([x, ffn_output])
        x = layers.GlobalAveragePooling1D()(x)
        return models.Model(inputs=inputs, outputs=x)
    # -------------------------------------------------------------------------------------------------------------------------------------------------------

    # ----------------------------------------
    # 小波变换模块
    # ----------------------------------------
    @staticmethod
    def wavelet_transform_block(x, wavelet_name='db1', mode='symmetric'):
        def wavelet_fn(x_batch):
            def _wavelet_decompose(image):
                coeffs = pywt.dwt2(image, wavelet=wavelet_name, mode=mode)
                LL, (LH, HL, HH) = coeffs
                return np.stack([LL, LH, HL, HH], axis=-1)
            
            x_np = x_batch.numpy()
            decomposed = np.array([_wavelet_decompose(x_np[i, :, :, 0]) for i in range(x_np.shape[0])])
            return decomposed.reshape(-1, x_np.shape[1]//2, x_np.shape[2]//2, 4)
        
        x_decomposed = layers.Lambda(
            lambda x: tf.py_function(
                func=wavelet_fn,
                inp=[x],
                Tout=tf.float32
            ),
            name='wavelet_transform')(x)
        
        x_decomposed = layers.Lambda(
            lambda x: tf.ensure_shape(x, [None, None, None, 4]),
            name='shape_ensurer'
        )(x_decomposed)
        
        x_wavelet = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x_decomposed)
        return layers.BatchNormalization()(x_wavelet)
    # -------------------------------------------------------------------------------------------------------------------------------------------------------

    # ------------------------------
    # 教师模型
    # ------------------------------
    @staticmethod
    def create_teacher_model(input_shape=(*Config.IMAGE_SIZE, Config.CHANNELS), num_classes=Config.NUM_CLASSES):
        inputs = layers.Input(shape=input_shape, name='teacher_input')     

        def spatial_feature_extractor():
            inputs = layers.Input(shape=(*Config.IMAGE_SIZE, Config.CHANNELS))
            x = ModelBuilder.residual_block(inputs, 32, prefix='spatial_block1')
            x = ModelBuilder.residual_block(x, 64, prefix='spatial_block2')
            x = ModelBuilder.residual_block(x, 128, prefix='spatial_block3')
            return models.Model(inputs=inputs, outputs=x, name='spatial_extractor')

        def temporal_feature_extractor():
            inputs = layers.Input(shape=(*Config.IMAGE_SIZE, Config.CHANNELS))
            x = layers.Reshape((40, 40))(inputs)
            vit = ModelBuilder.create_vit1d(
                input_shape=(40, 40),
                embed_dim=Config.TEMPORAL_DIM,
                num_heads=4,
                ff_dim=128,
                num_transformer_blocks=2)
            return models.Model(inputs=inputs, outputs=vit(x), name='temporal_extractor')

        def frequency_feature_extractor():
            inputs = layers.Input(shape=(*Config.IMAGE_SIZE, Config.CHANNELS))
            x_freq = ModelBuilder.wavelet_transform_block(inputs, wavelet_name='db1', mode='symmetric')
            x_freq = layers.Conv2D(256, (3,3), padding='same', activation='relu')(x_freq)
            x_freq = layers.BatchNormalization()(x_freq)
            x_freq = layers.GlobalAveragePooling2D()(x_freq)  
            x_freq = layers.Dense(256, activation='relu')(x_freq)  
            return models.Model(inputs=inputs, outputs=x_freq, name='frequency_extractor')

        def simple_concat_fusion(spatial_feat, freq_feat, temporal_feat):
            spatial = layers.GlobalAvgPool2D()(spatial_feat)  
            freq = freq_feat  
            temporal = temporal_feat  
            spatial = layers.Dense(256)(spatial)
            temporal = layers.Dense(256)(temporal) 
            return layers.Concatenate()([spatial, freq, temporal])  


        spatial_feat = spatial_feature_extractor()(inputs)
        freq_feat = frequency_feature_extractor()(inputs)
        temporal_feat = temporal_feature_extractor()(inputs)
        fused_feat = simple_concat_fusion(spatial_feat, freq_feat, temporal_feat)

        x = layers.Dense(512, activation='relu')(fused_feat)  
        x = layers.Dropout(0.3)(x)
        output_main = layers.Dense(num_classes, activation='softmax')(x)

        output_spatial = layers.Lambda(lambda x: tf.identity(x), name='spatial_output')(spatial_feat)
        output_temporal = layers.Lambda(lambda x: tf.identity(x), name='temporal_output')(temporal_feat)
        output_frequency = layers.Lambda(lambda x: tf.identity(x), name='frequency_output')(freq_feat)
        output_fused = layers.Lambda(lambda x: tf.identity(x), name='fused_output')(fused_feat)

        return models.Model(
            inputs=inputs,
            outputs=[output_main, output_spatial, output_temporal, output_frequency, output_fused],
            name='teacher_model')


    # ==============================
    # 学生模型
    # ==============================
    @staticmethod
    def create_student_model(input_shape=(*Config.IMAGE_SIZE, Config.CHANNELS),
                            num_classes=Config.NUM_CLASSES):
        inputs = layers.Input(shape=input_shape, name='student_input')
        input_copy = tf.identity(inputs, name='input_copy')

        def depthwise_block(x, filters, kernel_size=(3, 3), dropout_rate=0.0, prefix=''):
            x = layers.SeparableConv2D(filters, kernel_size, padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(dropout_rate)(x)
            return x

        x = depthwise_block(inputs, 32, (3, 3), 0.15, 'block1')
        x = depthwise_block(x, 64, (3, 3), 0.2, 'block2')
        x = depthwise_block(x, 128, (3, 3), 0.25, 'block3')      

        x_spatial = layers.Conv2D(128, (1, 1), padding='same', activation='relu',
                                name='student_spatial_adapter')(x)  

        x_reshape = layers.Reshape((-1, 128))(x)                    
        vit = ModelBuilder.create_vit1d(
            input_shape=(25, 128),
            num_heads=2,
            embed_dim=32,
            num_transformer_blocks=1)                                
        x_temporal = layers.Dense(Config.TEMPORAL_DIM,
                                name='temporal_adapter')(vit(x_reshape))  

        x_wavelet = ModelBuilder.wavelet_transform_block(inputs, wavelet_name='db1', mode='symmetric')
        x_wavelet = layers.Conv2D(64, (3, 3), activation='relu', name='wavelet_conv1')(x_wavelet)
        x_wavelet = layers.MaxPooling2D(2)(x_wavelet)                
        x_freq = layers.GlobalAveragePooling2D(name='freq_global_pool')(x_wavelet)
        x_freq = layers.Dense(256, activation='relu', name='frequency_adapter')(x_freq)      

        def student_concat_fusion(spatial, temporal, freq):
            spatial_flat = layers.GlobalAveragePooling2D()(spatial)  
            spatial = layers.Dense(256)(spatial_flat)
            temporal = layers.Dense(256)(temporal)
            freq = layers.Dense(256)(freq)
            return layers.Concatenate()([spatial, temporal, freq])   

        fused_features = student_concat_fusion(x_spatial, x_temporal, x_freq)

        x = layers.Dense(512, activation='relu')(fused_features)     
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(num_classes, activation='softmax', name='student_output')(x)

        return models.Model(
            inputs=inputs,
            outputs={
                'logits': outputs,
                'spatial': x_spatial,      
                'temporal': x_temporal,    
                'frequency': x_freq,       
                'fused': fused_features,   
                'original_input': input_copy}, name='student_model')



class Trainer:
    def __init__(self, teacher_model, student_model):
        self.teacher_model = teacher_model
        self.student_model = student_model
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_writer = tf.summary.create_file_writer(f"{Config.LOG_DIR}{current_time}/train")
        self.val_writer = tf.summary.create_file_writer(f"{Config.LOG_DIR}{current_time}/val")

    def _log_metrics(self, metrics, step, mode="train"):
        writer = self.train_writer if mode == "train" else self.val_writer
        with writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar(name, value, step=step)

    def train_teacher(self, train_ds, val_ds):
        optimizer = tf.keras.optimizers.Adam(Config.LEARNING_RATE)
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        train_acc = tf.keras.metrics.CategoricalAccuracy()
        val_acc = tf.keras.metrics.CategoricalAccuracy()

        print("\n=== 教师模型训练开始 ===")
        for epoch in range(Config.TEACHER_EPOCHS):
            train_progress = tqdm(train_ds, desc=f"教师 Epoch {epoch+1}/{Config.TEACHER_EPOCHS}", unit="batch")
            epoch_loss = []
            
            for batch_images, batch_labels in train_progress:
                with tf.GradientTape() as tape:
                    outputs = self.teacher_model(batch_images, training=True)
                    cls_output = outputs[0]
                    loss = loss_fn(batch_labels, cls_output)
                
                grads = tape.gradient(loss, self.teacher_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.teacher_model.trainable_variables))                
                train_acc.update_state(batch_labels, cls_output)
                epoch_loss.append(loss.numpy())                
                train_progress.set_postfix({"loss": f"{np.mean(epoch_loss):.4f}", "acc": f"{train_acc.result().numpy():.4f}"})

            val_progress = tqdm(val_ds, desc="验证", unit="batch")
            val_loss = []
            
            for val_images, val_labels in val_progress:
                val_outputs = self.teacher_model(val_images, training=False)
                cls_output = val_outputs[0]
                val_loss.append(loss_fn(val_labels, cls_output).numpy())
                val_acc.update_state(val_labels, cls_output)                
                val_progress.set_postfix({"val_loss": f"{np.mean(val_loss):.4f}", "val_acc": f"{val_acc.result().numpy():.4f}"})

            metrics = {
                "teacher_train_loss": np.mean(epoch_loss),
                "teacher_train_acc": train_acc.result(),
                "teacher_val_loss": np.mean(val_loss),
                "teacher_val_acc": val_acc.result()}
            self._log_metrics(metrics, epoch+1)

            print(f"Epoch {epoch+1} | "
                  f"Train Loss: {np.mean(epoch_loss):.4f} | "
                  f"Train Acc: {train_acc.result().numpy():.4f} | "
                  f"Val Loss: {np.mean(val_loss):.4f} | "
                  f"Val Acc: {val_acc.result().numpy():.4f}")

            train_acc.reset_states()
            val_acc.reset_states()
        
        return self.teacher_model

    def train_student(self, train_ds, val_ds):

        sample_batch = next(iter(train_ds.take(1)))
        teacher_outputs = self.teacher_model(sample_batch[0])
        student_outputs = self.student_model(sample_batch[0])  
        print("\n=== 训练前维度验证 ===")
        print(f"教师空域特征: {teacher_outputs[1].shape}")
        print(f"教师时域特征: {teacher_outputs[2].shape}")
        print(f"教师频域特征: {teacher_outputs[3].shape}")
        print(f"教师融合特征: {teacher_outputs[4].shape}")
        print(f"学生空域特征: {student_outputs['spatial'].shape}")
        print(f"学生时域特征: {student_outputs['temporal'].shape}")
        print(f"学生频域特征: {student_outputs['frequency'].shape}")
        print(f"学生融合特征: {student_outputs['fused'].shape}")


        optimizer = tf.keras.optimizers.Adam()
        loss_fn = DistillationLoss(self.teacher_model)
        train_acc = tf.keras.metrics.CategoricalAccuracy()
        val_acc = tf.keras.metrics.CategoricalAccuracy()

        print("\n=== 学生模型蒸馏开始 ===")
        for epoch in range(Config.EPOCHS):
            train_progress = tqdm(train_ds, desc=f"学生 Epoch {epoch+1}/{Config.EPOCHS}", unit="batch")
            epoch_losses = {k: [] for k in ['total', 'classification', 'distillation', 'spatial', 'temporal', 'frequency', 'fused']}   # 更新损失键名匹配当前架构：
            
            for batch_images, batch_labels in train_progress:
                with tf.GradientTape() as tape:
                    outputs = self.student_model(batch_images, training=True)
                    total_loss = loss_fn(batch_labels, outputs)
                    losses = loss_fn.last_losses
                
                grads = tape.gradient(total_loss, self.student_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.student_model.trainable_variables))
                
                train_acc.update_state(batch_labels, outputs['logits'])
                for k in epoch_losses.keys():
                    epoch_losses[k].append(losses[k].numpy())
                
                train_progress.set_postfix({
                    "total": f"{np.mean(epoch_losses['total']):.4f}",
                    "acc": f"{train_acc.result().numpy():.4f}",
                    "cls": f"{np.mean(epoch_losses['classification']):.4f}",
                    "distill": f"{np.mean(epoch_losses['distillation']):.4f}"})

            val_progress = tqdm(val_ds, desc="验证", unit="batch")
            val_losses = {k: [] for k in ['total', 'classification', 'distillation', 'spatial', 'temporal', 'frequency', 'fused']}
            
            for val_images, val_labels in val_progress:
                outputs = self.student_model(val_images, training=False)
                _ = loss_fn(val_labels, outputs)
                val_probs = tf.nn.softmax(outputs['logits'])
                val_acc.update_state(val_labels, val_probs)                
                losses = loss_fn.last_losses
                for k in val_losses.keys():
                    val_losses[k].append(losses[k].numpy())                
                val_progress.set_postfix({"val_loss": f"{np.mean(val_losses['total']):.4f}", "val_acc": f"{val_acc.result().numpy():.4f}"})

            train_metrics = {f"student_{k}": np.mean(v) for k, v in epoch_losses.items()}
            train_metrics["student_train_acc"] = train_acc.result()
            self._log_metrics(train_metrics, epoch+1, "train")
            
            val_metrics = {f"student_val_{k}": np.mean(v) for k, v in val_losses.items()}
            val_metrics["student_val_acc"] = val_acc.result()
            self._log_metrics(val_metrics, epoch+1, "val")

            print("\n" + "="*60)
            print(f"学生模型 Epoch {epoch+1} 训练结果".center(60))
            print("-"*60)
            print(f"Total Loss: {np.mean(epoch_losses['total']):.4f} | "
                  f"classification_loss: {np.mean(epoch_losses['classification']):.4f}\n"  
                  f"distillation_loss: {np.mean(epoch_losses['distillation']):.4f} | "     
                  f"spatial_feature_loss: {np.mean(epoch_losses['spatial']):.4f}\n"        
                  f"freq_feature_loss: {np.mean(epoch_losses['frequency']):.4f} | "        
                  f"fused_feature_loss: {np.mean(epoch_losses['fused']):.4f}")             
            print(f"Train Accuracy: {train_acc.result().numpy():.4f} | "                   
                  f"Val Accuracy: {val_acc.result().numpy():.4f}")                         
            print("="*60 + "\n")
            
            train_acc.reset_states()
            val_acc.reset_states()
        
        return self.student_model
# ------------------------------



class DistillationLoss(tf.keras.losses.Loss):
    def __init__(self, teacher_model, temperature=5.0, alpha=0.5, name='distillation_loss'):
        super().__init__(name=name)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = tf.keras.losses.KLDivergence()
        self.cce_loss = tf.keras.losses.CategoricalCrossentropy()


        dummy_input = tf.random.normal([1, *Config.IMAGE_SIZE, Config.CHANNELS])
        teacher_outputs = teacher_model(dummy_input)
        
        print("\n=== 教师模型中间层维度 ===")
        print(f"空间特征: {teacher_outputs[1].shape}")
        print(f"时域特征: {teacher_outputs[2].shape}")
        print(f"频域特征: {teacher_outputs[3].shape}")
        print(f"融合特征: {teacher_outputs[4].shape}\n")

        self.spatial_aligner = ModelBuilder.create_spindle_mlp(
            input_dim=int(tf.reduce_prod(teacher_outputs[1].shape[1:])),
            output_dim=int(tf.reduce_prod(teacher_outputs[1].shape[1:])))
        self.temporal_aligner = ModelBuilder.create_spindle_mlp(
            input_dim=int(teacher_outputs[2].shape[-1]),
            output_dim=int(teacher_outputs[2].shape[-1]))
        self.frequency_aligner = ModelBuilder.create_spindle_mlp(           
            input_dim=int(teacher_outputs[3].shape[-1]),
            output_dim=int(teacher_outputs[3].shape[-1]))
        self.fused_aligner = ModelBuilder.create_spindle_mlp(
            input_dim=int(teacher_outputs[4].shape[-1]),
            output_dim=int(teacher_outputs[4].shape[-1]))

    def call(self, y_true, y_pred):
        teacher_temporal = self.teacher_model(y_pred['original_input'])[2]
        student_temporal = y_pred['temporal']
        student_inputs = y_pred['original_input']
        student_logits = y_pred['logits']
        student_spatial = y_pred['spatial']
        student_temporal = y_pred['temporal']
        student_frequency = y_pred['frequency']                                
        student_fused = y_pred['fused']

        teacher_outputs = self.teacher_model(student_inputs)
        teacher_logits = teacher_outputs[0]
        teacher_spatial = teacher_outputs[1]
        teacher_temporal = teacher_outputs[2]
        teacher_frequency = teacher_outputs[3]                                                                  
        teacher_fused = teacher_outputs[4]

        aligned_student_spatial = self.spatial_aligner(layers.Flatten()(student_spatial))
        aligned_student_temporal = self.temporal_aligner(student_temporal)
        aligned_student_frequency = self.frequency_aligner(student_frequency)                                            
        aligned_student_fused = self.fused_aligner(student_fused)

        aligned_teacher_spatial = self.spatial_aligner(layers.Flatten()(teacher_spatial))
        aligned_teacher_temporal = self.temporal_aligner(teacher_temporal)
        aligned_teacher_frequency = self.frequency_aligner(teacher_frequency)                                     
        aligned_teacher_fused = self.fused_aligner(teacher_fused)

        classification_loss = self.cce_loss(y_true, y_pred['logits'])
        distillation_loss = self.kl_loss(
            tf.nn.softmax(teacher_logits/self.temperature, axis=1),
            tf.nn.softmax(student_logits/self.temperature, axis=1)) * (self.temperature ** 2)
        spatial_feature_loss = tf.reduce_mean(tf.square(aligned_teacher_spatial - aligned_student_spatial))
        temporal_feature_loss = tf.reduce_mean(tf.square(aligned_teacher_temporal - aligned_student_temporal))
        frequency_feature_loss = tf.reduce_mean(tf.square(aligned_teacher_frequency - aligned_student_frequency))  
        fused_feature_loss = tf.reduce_mean(tf.square(aligned_teacher_fused - aligned_student_fused))

        total_loss = (self.alpha * classification_loss + 
                     (1-self.alpha) * distillation_loss +
                     0.1 * spatial_feature_loss +
                     0.1 * temporal_feature_loss +
                     0.1 * frequency_feature_loss +                                                                
                     0.1 * fused_feature_loss)

        self.last_losses = {
            'total': total_loss,
            'classification': classification_loss,
            'distillation': distillation_loss,
            'spatial': spatial_feature_loss,
            'temporal': temporal_feature_loss,
            'frequency': frequency_feature_loss,                                                                    
            'fused': fused_feature_loss}

        return total_loss



