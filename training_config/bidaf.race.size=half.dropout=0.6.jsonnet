// Configuration for the a machine comprehension model based on:
//   Seo, Min Joon et al. “Bidirectional Attention Flow for Machine Comprehension.”
//   ArXiv/1611.01603 (2016)
{
  "dataset_reader": {
    "type": "race",
    "lazy": true,
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "token_characters": {
        "type": "characters",
        "character_tokenizer": {
          "byte_encoding": "utf-8",
          "start_tokens": [259],
          "end_tokens": [260]
        }
      }
    }
  },
  "train_data_path": "datasets/race/race-train-v1.0.json",
  "validation_data_path": "datasets/race/race-dev-v1.0.json",
  "model": {
    "type": "bidaf",
    "text_field_embedder": {
        "token_embedders": {
            "tokens": {
                "type": "embedding",
                "pretrained_file": "datasets/glove/glove.6B.100d.txt.gz",
                "embedding_dim": 100,
                "trainable": false
            },
            "token_characters": {
                "type": "character_encoding",
                "embedding": {
                "num_embeddings": 262,
                "embedding_dim": 16
                },
                "encoder": {
                "type": "cnn",
                "embedding_dim": 16,
                "num_filters": 100,
                "ngram_filter_sizes": [5]
                },
                "dropout": 0.6
            }
        }
    },
    "num_highway_layers": 2,
    "phrase_layer": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 200,
      "hidden_size": 50,
      "num_layers": 1,
      "dropout": 0.6
    },
    "similarity_function": {
      "type": "linear",
      "combination": "x,y,x*y",
      "tensor_1_dim": 100,
      "tensor_2_dim": 100
    },
    "modeling_layer": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 400,
      "hidden_size": 50,
      "num_layers": 2,
      "dropout": 0.6
    },
    "span_end_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 700,
      "hidden_size": 50,
      "num_layers": 1,
      "dropout": 0.6
    },
    "dropout": 0.6
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["passage", "num_tokens"], ["question", "num_tokens"]],
    "batch_size": 40
  },

  "trainer": {
    "num_epochs": 200,
    "grad_norm": 5.0,
    "patience": 10,
    "validation_metric": "+em",
    "cuda_device": 0,
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "mode": "max",
      "patience": 2
    },
    "optimizer": {
      "type": "adam",
      "betas": [0.9, 0.9]
    }
  }
}
