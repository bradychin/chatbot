import os

########################################################################
# Modify model path.
MODEL_PATH = 'models/best_model'
########################################################################
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#--------- main.py ---------#
model_path = os.path.join(PROJECT_ROOT, MODEL_PATH)


#--------- model_trainer.py ---------#
dataset_file_path = os.path.join(PROJECT_ROOT, 'data/personachat_full.csv')
checkpoint_file_path = 'models/checkpoints/checkpoint_epoch_{epoch}_batch_{batch_index}'
best_model_file_path = 'models/val loss/model_best_val_loss_{best_validation_loss:.4f}'
final_model_file_path = 'models/model'

epochs = 4
batch_size = 4
patience = 1
gradient_accumulation = 2
lr = 2e-5
weight_decay=0.01

#--------- dialog_management.py ---------#
max_length = 128

#--------- logger.py ---------#
log_path = os.path.join(PROJECT_ROOT, 'src/log/app.log')