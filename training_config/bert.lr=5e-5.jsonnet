{
  "dataset_reader": {
    "type": "squad",
    "token_indexers": {
      "tokens": {
          "type": "bert-pretrained",
          "pretrained_model": "datasets/bert/uncased_L-12_H-768_A-12/vocab.txt",
          "do_lowercase": true,
          "use_starting_offsets": true
      }
    }
  },
  "train_data_path": "datasets/squad/squad-train-v1.1.json",
  "validation_data_path": "datasets/squad/squad-dev-v1.1.json",
  "model": {
    "type": "bert-qa",
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
    "span_end_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 2304,
      "hidden_size": 100,
      "num_layers": 1,
      "dropout": 0.1
    },
    "regularizer": [
      [
        "scalar_parameters",
        {
          "type": "l2",
          "alpha": 0.01
        }
      ]
    ],
    "dropout": 0.1
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["passage", "num_tokens"], ["question", "num_tokens"]],
    "batch_size": 16
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
      "lr": 0.00005,
      "type": "adam",
      "betas": [0.9, 0.999]
    }
  }
}
