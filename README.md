# NSC
A repo for all codes that solve the non standard cosmology

You can install this code by going to the parent directory an entering 

```bash
$ pip install -e NSC
```

Then you can check if it's worked by 

```bash 
$ python -m NSC.solve_Cosmo
```

If that works then your should be good to go

```python
>>> from NSC import solve_Cosmo
>>> invdil, Teq, Tdec = solve_Cosmo.GW_input_lam(1e9, 6, 1e16)
>>> print(invdil, Teq, Tdec)
```
which gives the output 

```
8432.144564356551 29.76302382120224 0.002411100022863685
```