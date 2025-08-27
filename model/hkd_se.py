# coding=utf-8
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# tf.debugging.set_log_device_placement(True)   # 把算子放在哪打印出来
from tensorflow.keras import layers, models, callbacks
import pathlib
from tqdm import tqdm
import datetime
import numpy as np
import time
import json
import os
import matplotlib.pyplot as plt
import cv2  
import seaborn as sns
# from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import random
import pywt
# tf.config.run_functions_eagerly(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
from tensorflow.keras import Sequential
# 固定随机种子
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==============================
# 配置模块
# ==============================
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
    """把 one-hot 变整数标签"""
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
        y_true, y_pred = [], []  # 收集预测结果
        for x, y in test_ds:
            y_true.extend(y.numpy())
            predictions = model.predict(x, verbose=0)  # 添加verbose=0
            # 动态选择访问方式
            if isinstance(predictions, dict):  # 如果是字典
                y_pred.extend(predictions['logits'])
            elif isinstance(predictions, list):  # 如果是列表
                y_pred.extend(predictions[0])
            else:
                raise ValueError("未知的模型预测返回值类型")
        metrics = self.compute_metrics(model, np.array(y_true), np.array(y_pred), class_names)  # 计算指标
        # 修改文件名，添加模型名称作为前缀
        self.plot_confusion_matrix(np.array(y_true), np.array(y_pred), class_names, filename=f'{model_name}_confusion_matrix.png')  # 生成混淆矩阵
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
        
        # 添加位置编码
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
            name='wavelet_transform'
        )(x)
        
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

        def temporal_feature_extractor():
            inp = layers.Input(shape=(*Config.IMAGE_SIZE, Config.CHANNELS))
            x = layers.Reshape((Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]))(inp)
            vit = ModelBuilder.create_vit1d(
                input_shape=(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]),
                embed_dim=Config.TEMPORAL_DIM,
                num_heads=4,
                ff_dim=128,
                num_transformer_blocks=2)
            return models.Model(inputs=inp, outputs=vit(x), name='temporal_extractor')

        def frequency_feature_extractor():
            inp = layers.Input(shape=(*Config.IMAGE_SIZE, Config.CHANNELS))
            x_freq = ModelBuilder.wavelet_transform_block(inp, wavelet_name='db1', mode='symmetric')
            x_freq = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x_freq)
            x_freq = layers.BatchNormalization()(x_freq)
            x_freq = layers.GlobalAveragePooling2D()(x_freq)
            x_freq = layers.Dense(256, activation='relu')(x_freq)
            return models.Model(inputs=inp, outputs=x_freq, name='frequency_extractor')

        def dynamic_feature_fusion(temporal_feat, freq_feat):
            temporal = layers.Dense(256)(temporal_feat)
            freq = layers.Dense(256)(freq_feat)

            quality_net = models.Sequential([
                layers.Dense(128, activation='relu'),
                layers.Dense(2, activation='softmax')
            ], name='quality_net')

            enhance_net = models.Sequential([
                layers.Dense(256, activation='relu'),
                layers.LayerNormalization()
            ], name='enhance_net')

            quality = quality_net(tf.concat([temporal, freq], axis=-1))
            enhanced_temporal = quality[:, 0:1] * enhance_net(temporal)
            enhanced_freq = quality[:, 1:2] * enhance_net(freq)

            fused = enhanced_temporal + enhanced_freq
            fused += layers.Dense(256)(tf.concat([temporal, freq], axis=-1))
            return layers.LayerNormalization()(fused)

        temporal_feat = temporal_feature_extractor()(inputs)
        freq_feat = frequency_feature_extractor()(inputs)
        fused_feat = dynamic_feature_fusion(temporal_feat, freq_feat)

        x = layers.Dense(128, activation='relu')(fused_feat)
        x = layers.Dropout(0.3)(x)
        output_main = layers.Dense(num_classes, activation='softmax', name='classification')(x)

        output_temporal = layers.Lambda(lambda x: tf.identity(x), name='temporal_output')(temporal_feat)
        output_frequency = layers.Lambda(lambda x: tf.identity(x), name='frequency_output')(freq_feat)
        output_fused = layers.Lambda(lambda x: tf.identity(x), name='fused_output')(fused_feat)

        return models.Model(
            inputs=inputs,
            outputs=[output_main, output_temporal, output_frequency, output_fused],
            name='teacher_model')




    # ==============================
    # 学生模型
    # ==============================
    @staticmethod
    def create_student_model(input_shape=(*Config.IMAGE_SIZE, Config.CHANNELS), num_classes=Config.NUM_CLASSES):
        inputs = layers.Input(shape=input_shape, name='student_input')
        input_copy = tf.identity(inputs, name='input_copy')

        x_reshape = layers.Reshape((Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]))(inputs)
        vit = ModelBuilder.create_vit1d(
            input_shape=(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]),
            num_heads=2,
            embed_dim=32,
            num_transformer_blocks=1)
        x_temporal = vit(x_reshape)
        x_temporal = layers.Dense(Config.TEMPORAL_DIM, name='temporal_adapter')(x_temporal)

        x_wavelet = ModelBuilder.wavelet_transform_block(inputs, wavelet_name='db1', mode='symmetric')
        x_wavelet = layers.Conv2D(64, (3, 3), activation='relu', name='wavelet_conv1')(x_wavelet)
        x_wavelet = layers.MaxPooling2D(2)(x_wavelet)
        x_freq = layers.GlobalAveragePooling2D(name='freq_global_pool')(x_wavelet)
        x_freq = layers.Dense(256, activation='relu', name='frequency_adapter')(x_freq)

        def student_fusion(temporal, freq):
            temporal = layers.Dense(256)(temporal)
            freq = layers.Dense(256)(freq)

            gate_net = models.Sequential([
                layers.Dense(64, activation='relu'),
                layers.Dense(2, activation='softmax')
            ])
            gates = gate_net(tf.concat([temporal, freq], axis=-1))
            fused = gates[:, 0:1] * temporal + gates[:, 1:2] * freq
            return fused

        fused_features = student_fusion(x_temporal, x_freq)

        x = layers.Dense(128, activation='relu')(fused_features)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(num_classes, activation='softmax', name='student_output')(x)

        return models.Model(
            inputs=inputs,
            outputs={
                'logits': outputs,
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
        self.best_val_acc = 0.0
        self.patience = 5
        self.wait = 0

    def _log_metrics(self, metrics, step, mode="train"):
        writer = self.train_writer if mode == "train" else self.val_writer
        with writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar(name, value, step=step)

    def train_teacher(self, train_ds, val_ds, callbacks=None):
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=Config.LEARNING_RATE,
                decay_steps=Config.TEACHER_EPOCHS * len(train_ds)))
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        train_acc = tf.keras.metrics.CategoricalAccuracy()
        val_acc = tf.keras.metrics.CategoricalAccuracy()

        print("\n=== 教师模型训练开始 ===")
        for epoch in range(Config.TEACHER_EPOCHS):
            # 训练阶段
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
                train_progress.set_postfix({
                    "loss": f"{np.mean(epoch_loss):.4f}", 
                    "acc": f"{train_acc.result().numpy():.4f}"})

            val_loss, val_acc = self._evaluate_epoch(val_ds, loss_fn, val_acc)
            
            metrics = {
                "teacher_train_loss": np.mean(epoch_loss),
                "teacher_train_acc": train_acc.result(),
                "teacher_val_loss": val_loss,
                "teacher_val_acc": val_acc.result()}
            self._log_metrics(metrics, epoch+1)

            current_val_acc = val_acc.result().numpy()
            if current_val_acc > self.best_val_acc:
                self.best_val_acc = current_val_acc
                self.wait = 0
                os.makedirs(os.path.dirname(f"{Config.MODEL_SAVE_PATH}teacher_best.h5"), exist_ok=True)
                self.teacher_model.save_weights(f"{Config.MODEL_SAVE_PATH}teacher_best.h5")
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f"\n早停触发，最佳验证准确率: {self.best_val_acc:.4f}")
                    break

            print(f"Epoch {epoch+1} | "
                  f"Train Loss: {np.mean(epoch_loss):.4f} | "
                  f"Train Acc: {train_acc.result().numpy():.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc.result().numpy():.4f}")

            train_acc.reset_states()
            val_acc.reset_states()
        
        # 加载最佳模型
        self.teacher_model.load_weights(f"{Config.MODEL_SAVE_PATH}teacher_best.h5")
        return self.teacher_model

    def train_student(self, train_ds, val_ds):
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=Config.LEARNING_RATE,
                decay_steps=Config.EPOCHS * len(train_ds)))
        
        loss_fn = AdaptiveDistillationLoss(self.teacher_model)
        train_acc = tf.keras.metrics.CategoricalAccuracy()
        val_acc = tf.keras.metrics.CategoricalAccuracy()
        self.best_val_acc = 0.0
        self.wait = 0

        self._validate_dimensions(train_ds)

        print("\n=== 学生模型蒸馏开始 ===")
        for epoch in range(Config.EPOCHS):
            train_progress = tqdm(train_ds, desc=f"学生 Epoch {epoch+1}/{Config.EPOCHS}", unit="batch")
            epoch_losses = {k: [] for k in [
                'total', 'classification', 'distillation', 
                'temporal', 'frequency', 'fused',
                'weight_temporal', 'weight_frequency', 'weight_fused']}
            
            for batch_images, batch_labels in train_progress:
                with tf.GradientTape() as tape:
                    outputs = self.student_model(batch_images, training=True)
                    total_loss = loss_fn(batch_labels, outputs)
                    losses = loss_fn.last_losses
                
                trainable_vars = (self.student_model.trainable_variables + loss_fn.trainable_variables)
                grads = tape.gradient(total_loss, trainable_vars)
                optimizer.apply_gradients(zip(grads, trainable_vars))
        
                train_acc.update_state(batch_labels, outputs['logits'])
                for k in epoch_losses.keys():
                    if k in losses:
                        epoch_losses[k].append(losses[k].numpy())
                
                train_progress.set_postfix({
                    "total": f"{np.mean(epoch_losses['total']):.4f}",
                    "acc": f"{train_acc.result().numpy():.4f}",
                    "cls": f"{np.mean(epoch_losses['classification']):.4f}",
                    "distill": f"{np.mean(epoch_losses['distillation']):.4f}"})

            val_loss, val_acc = self._evaluate_student(val_ds, loss_fn, val_acc)
            
            train_metrics = {
                f"student_{k}": np.mean(v) 
                for k, v in epoch_losses.items() 
                if not k.startswith('weight_')}
            train_metrics["student_train_acc"] = train_acc.result()
            self._log_metrics(train_metrics, epoch+1, "train")
            
            weight_metrics = {
                f"weight/{k.split('_')[1]}": np.mean(v)
                for k, v in epoch_losses.items()
                if k.startswith('weight_')}
            self._log_metrics(weight_metrics, epoch+1, "train")
            
            val_metrics = {
                "student_val_total": val_loss,
                "student_val_acc": val_acc.result()}
            self._log_metrics(val_metrics, epoch+1, "val")

            self._print_epoch_summary(epoch, epoch_losses, train_acc, val_acc)
            
            current_val_acc = val_acc.result().numpy()
            if current_val_acc > self.best_val_acc:
                self.best_val_acc = current_val_acc
                self.wait = 0
                self.student_model.save_weights(f"{Config.MODEL_SAVE_PATH}student_best.h5")
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f"\n早停触发，最佳验证准确率: {self.best_val_acc:.4f}")
                    break

            train_acc.reset_states()
            val_acc.reset_states()
        
        self.student_model.load_weights(f"{Config.MODEL_SAVE_PATH}student_best.h5")
        return self.student_model

    def _validate_dimensions(self, train_ds):
        sample_batch = next(iter(train_ds.take(1)))
        teacher_outputs = self.teacher_model(sample_batch[0])
        student_outputs = self.student_model(sample_batch[0])  
        print("\n=== 训练前维度验证 ===")
        print(f"教师时域特征: {teacher_outputs[1].shape}")
        print(f"教师频域特征: {teacher_outputs[2].shape}")
        print(f"教师融合特征: {teacher_outputs[3].shape}")
        print(f"学生时域特征: {student_outputs['temporal'].shape}")
        print(f"学生频域特征: {student_outputs['frequency'].shape}")
        print(f"学生融合特征: {student_outputs['fused'].shape}")

    def _evaluate_epoch(self, dataset, loss_fn, metric):
        """评估epoch"""
        losses = []
        progress = tqdm(dataset, desc="验证", unit="batch")
        for x, y in progress:
            outputs = self.teacher_model(x, training=False)
            losses.append(loss_fn(y, outputs[0]).numpy())
            metric.update_state(y, outputs[0])
            progress.set_postfix({
                "val_loss": f"{np.mean(losses):.4f}", 
                "val_acc": f"{metric.result().numpy():.4f}"})
        return np.mean(losses), metric

    def _evaluate_student(self, dataset, loss_fn, metric):
        losses = []
        progress = tqdm(dataset, desc="验证", unit="batch")
        for x, y in progress:
            outputs = self.student_model(x, training=False)
            _ = loss_fn(y, outputs)
            losses.append(loss_fn.last_losses['total'].numpy())
            metric.update_state(y, outputs['logits'])
            progress.set_postfix({
                "val_loss": f"{np.mean(losses):.4f}", 
                "val_acc": f"{metric.result().numpy():.4f}"})
        return np.mean(losses), metric

    def _print_epoch_summary(self, epoch, losses, train_acc, val_acc):
        print("\n" + "="*60)
        print(f"Student Model Epoch {epoch+1} training results".center(60))
        print("-"*60)
        print(f"Total Loss: {np.mean(losses['total']):.4f} | "
              f"classification_loss: {np.mean(losses['classification']):.4f}\n"
              f"distillation_loss: {np.mean(losses['distillation']):.4f} | "
              f"freq_feature_loss: {np.mean(losses['frequency']):.4f} | "
              f"fused_feature_loss: {np.mean(losses['fused']):.4f}")
        print(f"Weight allocation - temporal: {np.mean(losses['weight_temporal']):.3f} | "
              f"frequency: {np.mean(losses['weight_frequency']):.3f} | "
              f"fused: {np.mean(losses['weight_fused']):.3f}")
        print(f"Train Accuracy: {train_acc.result().numpy():.4f} | "
              f"Val Accuracy: {val_acc.result().numpy():.4f}")
        print("="*60 + "\n")




class AdaptiveDistillationLoss(tf.keras.losses.Loss):
    def __init__(self, teacher_model, temperature=5.0, alpha=0.5, name='adaptive_distill_loss'):
        super().__init__(name=name)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = tf.Variable(alpha, trainable=False, dtype=tf.float32, name='alpha')
        
        self.domain_weights = tf.Variable(
            initial_value=[1.0, 1.0, 1.0, 1.0], 
            trainable=True,  
            dtype=tf.float32, 
            name='domain_weights')
        
        self.kl_loss = tf.keras.losses.KLDivergence()
        self.cce_loss = tf.keras.losses.CategoricalCrossentropy()
        
        self.temporal_aligner = None
        self.frequency_aligner = None
        self.fused_aligner = None
        self._built = False
    
    def build(self, input_shape):
        if self._built:
            return
            
        dummy_input = tf.random.normal([1, *Config.IMAGE_SIZE, Config.CHANNELS])
        teacher_outputs = self.teacher_model(dummy_input)
        
        temporal_dim = teacher_outputs[1].shape[-1]
        self.temporal_aligner = ModelBuilder.create_spindle_mlp(
            input_dim=temporal_dim,
            output_dim=temporal_dim,
            name='temporal_aligner')
        self.temporal_aligner.build((None, temporal_dim))
        
        freq_dim = teacher_outputs[2].shape[-1]
        self.frequency_aligner = ModelBuilder.create_spindle_mlp(
            input_dim=freq_dim,
            output_dim=freq_dim,
            name='frequency_aligner')
        self.frequency_aligner.build((None, freq_dim))
        
        fused_dim = teacher_outputs[3].shape[-1]
        self.fused_aligner = ModelBuilder.create_spindle_mlp(
            input_dim=fused_dim,
            output_dim=fused_dim,
            name='fused_aligner')
        self.fused_aligner.build((None, fused_dim))
        
        self._built = True
    
    def call(self, y_true, y_pred):
        if not self._built:
            self.build(y_pred['original_input'].shape)
        
        teacher_outputs = self.teacher_model(y_pred['original_input'])
        teacher_logits = teacher_outputs[0]
        
        aligned_student = {
            'temporal': self.temporal_aligner(y_pred['temporal']),
            'frequency': self.frequency_aligner(y_pred['frequency']),
            'fused': self.fused_aligner(y_pred['fused'])}
        
        aligned_teacher = {
            'temporal': self.temporal_aligner(teacher_outputs[1]),
            'frequency': self.frequency_aligner(teacher_outputs[2]),
            'fused': self.fused_aligner(teacher_outputs[3])}
        
        classification_loss = self.cce_loss(y_true, y_pred['logits'])
        
        distillation_loss = self.kl_loss(
            tf.nn.softmax(teacher_logits/self.temperature, axis=1),
            tf.nn.softmax(y_pred['logits']/self.temperature, axis=1)
        ) * (self.temperature ** 2)
        
        feature_losses = {
            'temporal': tf.reduce_mean(tf.square(aligned_teacher['temporal'] - aligned_student['temporal'])),
            'frequency': tf.reduce_mean(tf.square(aligned_teacher['frequency'] - aligned_student['frequency'])),
            'fused': tf.reduce_mean(tf.square(aligned_teacher['fused'] - aligned_student['fused']))}
        
        weights = tf.nn.softmax(self.domain_weights)
        weighted_feature_loss = (
            weights[0] * feature_losses['temporal'] +
            weights[1] * feature_losses['frequency'] +
            weights[2] * feature_losses['fused'])
        
        total_loss = (self.alpha * classification_loss + 
                     (1 - self.alpha) * distillation_loss + 
                     weighted_feature_loss)
        
        self.last_losses = {
            'total': total_loss,
            'classification': classification_loss,
            'distillation': distillation_loss,
            'temporal': feature_losses['temporal'],
            'frequency': feature_losses['frequency'],
            'fused': feature_losses['fused'],
            'weight_temporal': weights[0],
            'weight_frequency': weights[1],
            'weight_fused': weights[2]}
        return total_loss
    
    @property
    def trainable_variables(self):
        vars = [self.domain_weights]
        if self._built:
            vars.extend(self.temporal_aligner.trainable_variables)
            vars.extend(self.frequency_aligner.trainable_variables)
            vars.extend(self.fused_aligner.trainable_variables)
        return vars
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'teacher_model': tf.keras.models.clone_model(self.teacher_model),
            'temperature': self.temperature.numpy(),
            'alpha': self.alpha.numpy(),
            'domain_weights': self.domain_weights.numpy()})
        return config




