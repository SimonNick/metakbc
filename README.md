# META-KBC

[![wercker status](https://app.wercker.com/status/d1bfa901741714e764a76b9533e71608/m/master "wercker status")](https://app.wercker.com/project/byKey/d1bfa901741714e764a76b9533e71608)

Meta-learning in Knowledge Base Completion.

Baselines:

```bash
$ ls *data=umls*.log | xargs grep "dev res" | 
    awk '{ print $6 " " $1 }' | tr ":" " " | 
    grep -v H@1 | sort -n | awk '{ print $1 " " $2 }' | tail -n 1
0.971172 rc_v1.V=3_b=100_data=umls_e=100_f2=0_i=standard_k=1000_lr=0.1_m=complex_n3=0.005_o=adagrad.log

$ cat rc_v1.V=3_b=100_data=umls_e=100_f2=0_i=standard_k=1000_lr=0.1_m=complex_n3=0.005_o=adagrad.log | grep 971172 -A 1
INFO:meta-cli.py:Epoch 69/100	dev results	MRR 0.971172	H@1 0.954755	H@3 0.984663	H@5 0.991564	H@10 0.994632
INFO:meta-cli.py:Epoch 69/100	test results	MRR 0.965119	H@1 0.941755	H@3 0.990923	H@5 0.994705	H@10 0.996974
```

```bash
$ ls *data=nations*.log | xargs grep "dev res" | 
    awk '{ print $6 " " $1 }' | tr ":" " " | 
    grep -v H@1 | sort -n | awk '{ print $1 " " $2 }' | tail -n 1
0.840101 rc_v1.V=3_b=100_data=nations_e=100_f2=0_i=standard_k=1000_lr=0.1_m=complex_n3=0.001_o=adagrad.log

$ cat rc_v1.V=3_b=100_data=nations_e=100_f2=0_i=standard_k=1000_lr=0.1_m=complex_n3=0.001_o=adagrad.log | grep 840101 -A 1
INFO:meta-cli.py:Epoch 15/100	dev results	MRR 0.840101	H@1 0.743719	H@3 0.924623	H@5 0.974874	H@10 0.997487
INFO:meta-cli.py:Epoch 15/100	test results	MRR 0.810315	H@1 0.699005	H@3 0.917910	H@5 0.957711	H@10 1.000000
```

```bash
$ ls *data=kinship*.log | xargs grep "dev res" | 
    awk '{ print $6 " " $1 }' | tr ":" " " | 
    grep -v H@1 | sort -n | awk '{ print $1 " " $2 }' | tail -n 1
0.891966 rc_v1.V=3_b=100_data=kinship_e=100_f2=0_i=standard_k=1000_lr=0.1_m=complex_n3=0.005_o=adagrad.log

$ cat rc_v1.V=3_b=100_data=kinship_e=100_f2=0_i=standard_k=1000_lr=0.1_m=complex_n3=0.005_o=adagrad.log | grep 891966 -A 1
INFO:meta-cli.py:Epoch 24/100	dev results	MRR 0.891966	H@1 0.833801	H@3 0.942416	H@5 0.963951	H@10 0.978933
INFO:meta-cli.py:Epoch 24/100	test results	MRR 0.889283	H@1 0.821229	H@3 0.949721	H@5 0.972998	H@10 0.985102
```