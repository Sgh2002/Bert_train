from transformers import PretrainedConfig

class L4qtcOnlyConfig(PretrainedConfig):
    model_type = "L4qtcOnly"
    def __init__(
            self, 
            vocab_size=53206,
            hidden_size=768,
            encoder_output_size=512,
            ml_intermediate_size_L1 = 768,
            mlp_intermediate_size_L2=256,
            mlp_dropout=0.1,
            num_classes=5,
            **kwargs):
        """
        Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Mixtral model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`MixtralModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
       
        """
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.ml_intermediate_size_L1 = ml_intermediate_size_L1
        self.mlp_intermediate_size_L2 = mlp_intermediate_size_L2
        self.mlp_dropout = mlp_dropout
        self.num_classes= num_classes
        self.encoder_output_size = encoder_output_size
      
        super().__init__(
            **kwargs,
        )