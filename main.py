import os, logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # 关闭 INFO/WARNING
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
import numpy as np
from hkd_net import Config, DataLoader, ModelBuilder, Trainer, ModelEvaluator, Utils
import pathlib
import json
import sys
from datetime import datetime

# ==========================================================================================
# 主程序
# ==========================================================================================
if __name__ == "__main__":
    Utils.setup_gpu()
    loader = DataLoader()  # 创建实例
    tf.config.optimizer.set_jit(True)

    # 加载数据
    train_ds = loader.load_dataset("./Train")
    val_ds   = loader.load_dataset("./Test")

    # 构建模型
    teacher = ModelBuilder.create_teacher_model()
    student = ModelBuilder.create_student_model()

    # 训练流程
    trainer = Trainer(teacher, student)
    trained_teacher = trainer.train_teacher(train_ds, val_ds)
    distilled_student = trainer.train_student(train_ds, val_ds)

    # 保存模型
    trained_teacher.save(Config.MODEL_SAVE_PATH + "./teacher_model.h5")
    distilled_student.save(Config.MODEL_SAVE_PATH + "./distilled_student_model.h5")


    # ================================ 训练完成后评估 =====================================
    evaluator = ModelEvaluator()
    data_root = pathlib.Path("./Test")
    class_names = sorted([item.name for item in data_root.glob('*/') if item.is_dir()])
    
    print("\n=== Teacher Model Evaluation ===")
    teacher_metrics = evaluator.full_evaluation(trained_teacher, val_ds, class_names, model_name="Teacher Model")
    print("Teacher Model Metries:")  
    print(f"accuracy: {teacher_metrics['accuracy']:.4f}")
    print(f"precision: {teacher_metrics['precision']:.4f}")
    print(f"recall: {teacher_metrics['recall']:.4f}") 
    print(f"f1: {teacher_metrics['f1']:.4f}")
    print(f"inference_time: {teacher_metrics['inference_time']*1000:.2f}ms")
    print("\n=== Student Model Evaluation ===")
    student_metrics = evaluator.full_evaluation(distilled_student, val_ds, class_names, model_name="Student Model")
    print("Student Model Metries:") 
    print(f"accuracy: {student_metrics['accuracy']:.4f}")
    print(f"precision: {student_metrics['precision']:.4f}")
    print(f"recall: {student_metrics['recall']:.4f}")
    print(f"f1: {student_metrics['f1']:.4f}") 
    print(f"inference_time: {student_metrics['inference_time']*1000:.2f}ms")
    
    def convert_to_float(metrics):
        return {k: float(v) if isinstance(v, np.float32) else v for k, v in metrics.items()}
    with open(f"{Config.MODEL_SAVE_PATH}evaluation.json", 'w') as f:
        json.dump({'teacher': convert_to_float(teacher_metrics), 'student': convert_to_float(student_metrics)}, f, indent=4)


