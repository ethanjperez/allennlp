{
  "dataset_reader": {
    "type": "race",
    "lazy": true,
    "token_indexers": {
      "tokens": {
          "type": "bert-pretrained",
          "pretrained_model": "datasets/bert/uncased_L-12_H-768_A-12/vocab.txt",
          "do_lowercase": false,
          "use_starting_offsets": true
      }
    }
  },
  "train_data_path": "allennlp/tests/fixtures/data/race.json",
  "validation_data_path": "allennlp/tests/fixtures/data/race.json",
  "model": {
    "type": "bidaf",
    "text_field_embedder": {
        "allow_unmatched_keys": true,
        "embedder_to_indexer_map": {
            "tokens": ["tokens", "tokens-offsets"]
        },
        "token_embedders": {
            "tokens": {
                "type": "bert-pretrained",
                "pretrained_model": "bert-base-uncased",
                "requires_grad": true,
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
    "batch_size": 16
  },

  "trainer": {
    "num_epochs": 4,
    "patience": 4,
    "validation_metric": "+em",
    "cuda_device": -1,
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.67,
      "mode": "max",
      "patience": 1
    },
    "optimizer": {
      "lr": 0.00003,
      "type": "adam",
      "betas": [0.9, 0.999]
    }
  }
}
