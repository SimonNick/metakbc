# meta-kbc
Meta-learning in Knowledge Base Completion

# Run

```
python metakbc/cli.py \
--dataset "Toy_A=>B_16" \
--method offline \
--rule_method attention \
--adv_method embedding \
--lr 0.1 \
--meta_lr 1.0 \
--epochs_outer 20 \
--epochs_inner 10 \
--batches_train 0 \
--epochs_adv 100 \
--rank 100 \
--batch_size 100 \
--reg_weight 1e-5 \
--seed 2
```
