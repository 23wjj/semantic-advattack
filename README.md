# Semantic-advattack

- Conducting adversarial attack at semantic communication system.

- The semantic extraction part of the proposed method in "Deep Learning-Enabled Semantic Communication Systems with Task-Unaware Transmitter and Dynamic Data".

## Notes
* ```MLP_MNIST_model.py```

  * code for training the pragmatic function

  * run it by ```python MLP_MNIST_model.py```

* ```MNIST.py```

  * code for training the semantic coding network

  * run it by ```python MNIST.py```
  * default compression rate is 1.0

* ```MNIST_fgsm.py```

  * code for conducting FGSM attack

  * before running it, make directory in the following structure to saving the data produced when running the code

    ```python
    # fgsm_attack
    #  |______________
    #  |              |
    #  0.2_epsilon    [other epsilon]
    #  |____________________
    #  |           |        |
    #  adversarial data  reverse
    ```

  

  * run it by ```python MNIST_fgsm.py --epsilon 0.2```

* ```MNIST_pgd.py```

  * code for conducting PGD attack

  * before running it, make directory in the following structure to saving the data produced when running the code

    ```python
    # pgd_attack
    #  |_________________________________
    #  |                                 |
    #  0.01_alpha_0.1_epsilon    [other alpha and epsilon]
    #  |____________________
    #  |           |        |
    #  adversarial data  reverse
    ```

  * run it by ```python MNIST_pgd.py --alpha 0.01 --epsilon 0.1```



## Visualization
* ```plot_fgsm.py```
  * code for visualization aggregation result and adversarial samples in FGSM attack
  * just run it by ```python plot_fgsm.py --epsilon 0.2```
  * make sure you have producing the data after attacking
* ```plot_pgd.py```
  * code for visualization aggregation result and adversarial samples in PGD attack
  * just run it by ```python plot_pgd.py --alpha 0.01 epsilon 0.1```
  * make sure you have producing the data after attacking
