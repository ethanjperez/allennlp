{
  "dataset_reader": {
    "type": "squad",
    "token_indexers": {
      "tokens": {
        "type": "openai_transformer_byte_pair",
        "model_path": "datasets/openai/openai-transformer-lm-2018.07.23.tar.gz"
      }
    }
  },
  "train_data_path": "allennlp/tests/fixtures/data/squad.json",
  "validation_data_path": "allennlp/tests/fixtures/data/squad.json",
  "model": {
    "type": "bidaf",
    "text_field_embedder": {
      "allow_unmatched_keys": true,
      "embedder_to_indexer_map": {
        "tokens": ["tokens", "tokens-offsets"]
      },
      "token_embedders": {
        "tokens": {
          "type": "openai_transformer_embedder",
          "transformer": {
              "model_path": "datasets/openai/openai-transformer-lm-2018.07.23.tar.gz"
          },
          "top_layer_only": true
        }
      }
    },
    "num_highway_layers": 1,
    "phrase_layer": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 768,
      "hidden_size": 25,
      "num_layers": 1,
      "dropout": 0.0
    },
    "similarity_function": {
      "type": "linear",
      "combination": "x,y,x*y",
      "tensor_1_dim": 50,
      "tensor_2_dim": 50
    },
    "modeling_layer": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 200,
      "hidden_size": 25,
      "num_layers": 1,
      "dropout": 0.0
    },
    "span_end_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 350,
      "hidden_size": 25,
      "num_layers": 1,
      "dropout": 0.0
    },
    "dropout": 0.0
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["passage", "num_tokens"], ["question", "num_tokens"]],
    "batch_size": 32
  },

  "trainer": {
    "num_epochs": 20,
    "patience": 3,
    "validation_metric": "+em",
    "cuda_device": -1,
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.67,
      "mode": "max",
      "patience": 1
    },
    "optimizer": {
      "lr": 0.0000625,
      "type": "adam",
      "betas": [0.9, 0.999]
    }
  }
}
