# nohup bash train_tv_spatial.sh > train_tv_spatial.out &
# nohup bash train_tv_spatial.sh > train_tv_spatial_clip.out &

# nohup bash train_tv_spatial.sh > train_tv_spatial_0_001.out &
# nohup bash train_tv_spatial.sh > train_tv_spatial_0_001_iter_20000.out &
DEVICE="0"

DATA="/data/pavia_university.mat"

SNR="30"

INR="siren"
K="9"
LAM="1.25"
ALPHA="0.0025"
NUM_ITERS="30000"
LR="3e-5"
#LOG_DIR="./runs,tv_spatial/snr=$SNR/k=$K,lam=$LAM,inr=$INR,alpha=$ALPHA,lr=$LR"
#CUDA_VISIBLE_DEVICES=$DEVICE python -u train_tv_spatial.py --data $DATA \
#  --snr $SNR \
#  --lam $LAM \
#  --k $K \
#  --inr $INR \
#  --cuda \
#  --alpha $ALPHA \
#  --log_dir $LOG_DIR

LOG_DIR="./runs,tv_spatial/snr=$SNR/k=$K,lam=$LAM,inr=$INR,alpha=$ALPHA,lr=$LR"
CUDA_VISIBLE_DEVICES=$DEVICE python -u train_tv_spatial.py --data $DATA \
  --snr $SNR \
  --lam $LAM \
  --k $K \
  --inr $INR \
  --cuda \
  --num_iters $NUM_ITERS \
  --alpha $ALPHA \
  --lr $LR
  --log_dir $LOG_DIR

# w/ clip
#LOG_DIR="./runs,tv_spatial/snr=$SNR/k=$K,lam=$LAM,inr=$INR,alpha=$ALPHA"clip"
#CUDA_VISIBLE_DEVICES=$DEVICE python -u train_tv_spatial.py --data $DATA \
#  --snr $SNR \
#  --lam $LAM \
#  --k $K \
#  --inr $INR \
#  --cuda \
#  --clip \
#  --alpha $ALPHA \
#  --log_dir $LOG_DIR
