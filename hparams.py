# Audio:
num_mels = 80
num_freq = 1025
sample_rate = 20000
frame_length_ms = 50
frame_shift_ms = 12.5
preemphasis = 0.97
min_level_db = -100
ref_level_db = 20
griffin_lim_iters = 60
power = 1.5
signal_normalization = True
use_lws = False

# Text
text_cleaners = ['english_cleaners']

# Train
batch_size = 20
epochs = 10000
dataset_path = "dataset"
learning_rate = 0.0008
weight_decay = 1e-6
checkpoint_path = "./model_new"
grad_clip_thresh = 0.8
decay_step = [5000, 10000, 50000]
save_step = 100
log_step = 5
clear_Time = 20
