# train
A => A 100 | 50 | 50
B => C 100 | 50 | 50
D => E 100 | 50 | 50
F, G => H 100 | 50 | 50
I, J => K 100 | 50 | 50

# train
e0 A e1
e0 B e1
e0 D e1
e0 F e1
e1 G e2
e0 I e1
e1 J e2

# valid, test
e1 A e0
e0 C e1
e0 E e1
e0 H e2
e0 K e2