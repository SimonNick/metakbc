# meta-kbc
Meta-learning in Knowledge Base Completion

# Run

```
python cli.py --dataset "Toy_A,B=>C,D,E=>F_1024" --valid 1 --rule_method attention --epochs_outer 10 --epochs_inner 10 --batch_size 100 --rank 100 --lam 0.5 --meta_lr 0.5 --lr 1e-1 --method offline --epochs_adv 100 --adv_method embedding --seed 44
```
