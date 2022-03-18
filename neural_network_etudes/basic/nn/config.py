class Config:
    """
    Class representing a simple neural network.

    Attributes:
        epochs (int): number of passes over the dataset
        mini_batch_size (int): number of examples to include in a minibatch
        dropout (bool): use dropout
        p_dropout (float): probability of removing neuron in dropout regularization
        dropout_l (list): mask of boolean values where the value at each position corresponds to a layer
                          and specifies whether to perform the dropout regularization for that layer or not
        l2_reg (bool): use l2-regularization or not
        lbd (float): lambda parameter for l2 regularization
        adam (bool): use Adam optimization
        beta_1 (float): beta_1 parameter
        beta_2 (float): beta_2 parameter
        adaptive_lr (bool): use adaptive learning rate or not
    """

    def __init__(self,
                 sizes=(100, 100),
                 epochs=1,
                 mini_batch_size=2056,
                 eta=0.002,
                 dropout=True,
                 p_dropout=0.4,
                 dropout_l=(False, True, False, False),
                 l2_reg=True,
                 lbd=0.01,
                 adam=True,
                 beta_1=0.9,
                 beta_2=0.999,
                 adaptive_lr=False
                 ):
        self.sizes = list(sizes)
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.eta = eta
        self.dropout = dropout
        self.p_dropout = p_dropout
        self.dropout_l = list(dropout_l)
        self.l2_reg = l2_reg
        self.lbd = lbd
        self.adam = adam
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.adaptive_lr = adaptive_lr

    @classmethod
    def from_args(cls, parsed_args):
        return cls(
            parsed_args.sizes,
            parsed_args.epochs,
            parsed_args.mini_batch_size,
            parsed_args.eta,
            parsed_args.dropout,
            parsed_args.p_dropout,
            parsed_args.dropout_l,
            parsed_args.l2_reg,
            parsed_args.lbd,
            parsed_args.adam,
            parsed_args.beta_1,
            parsed_args.beta_2,
            parsed_args.adaptive_lr
        )
