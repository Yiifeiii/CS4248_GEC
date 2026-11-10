| run | phase | param | lr | wd | dropout | ls | seed | dev_tune F0.5 |
|---|---|---|---|---|---|---|---|---|
| t5-tune.base.lr3e-5.seed42 | 1 | base | 3e-5 | 0.01 | 0.1 | 0.1 | 42 | 55.61 |
| t5-tune.base.lr3e-5.seed123 | 1 | base | 3e-5 | 0.01 | 0.1 | 0.1 | 123 | 55.44 |
| t5-tune.base.lr3e-5.seed7 | 1 | base | 3e-5 | 0.01 | 0.1 | 0.1 | 7 | 55.29 |
| t5-tune.base.lr1e-5.seed42 | 1 | lr | 1e-5 | 0.01 | 0.1 | 0.1 | 42 | 54.28 |
| t5-tune.base.lr5e-5.seed42 | 1 | lr | 5e-5 | 0.01 | 0.1 | 0.1 | 42 | 55.72 |
| t5-tune.wd0.lr3e-5.seed42 | 2 | wd | 3e-5 | 0 | 0.1 | 0.1 | 42 | 55.13 |
| t5-tune.wd0.1.lr3e-5.seed42 | 2 | wd | 3e-5 | 0.1 | 0.1 | 0.1 | 42 | 55.25 |
| t5-tune.dropout0.lr3e-5.seed42 | 2 | dropout | 3e-5 | 0.01 | 0 | 0.1 | 42 | 54.56 |
| t5-tune.dropout0.3.lr3e-5.seed42 | 2 | dropout | 3e-5 | 0.01 | 0.3 | 0.1 | 42 | 50.73 |
| t5-tune.ls0.lr3e-5.seed42 | 2 | ls | 3e-5 | 0.01 | 0.1 | 0 | 42 | 54.87 |
| t5-tune.ls0.2.lr3e-5.seed42 | 2 | ls | 3e-5 | 0.01 | 0.1 | 0.2 | 42 | 55.03 |
