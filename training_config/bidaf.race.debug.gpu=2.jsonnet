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
  "train_data_path": "allennlp/tests/fixtures/data/race.json",
  "validation_data_path": "allennlp/tests/fixtures/data/race.json",
  "model": {
    "type": "bidaf",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 2,
            "trainable": false
        },
        "token_characters": {
            "type": "character_encoding",
            "embedding": {
            "num_embeddings": 262,
            "embedding_dim": 8
            },
            "encoder": {
            "type": "cnn",
            "embedding_dim": 8,
            "num_filters": 8,
            "ngram_filter_sizes": [5]
            }
        }
      }
    },
    "num_highway_layers": 1,
    "phrase_layer": {
      "type": "lstm",
      "input_size": 10,
      "hidden_size": 10,
      "num_layers": 1
    },
    "similarity_function": {
      "type": "linear",
      "combination": "x,y,x*y",
      "tensor_1_dim": 10,
      "tensor_2_dim": 10
    },
    "modeling_layer": {
      "type": "lstm",
      "input_size": 40,
      "hidden_size": 10,
      "num_layers": 1
    },
    "span_end_encoder": {
      "type": "lstm",
      "input_size": 70,
      "hidden_size": 10,
      "num_layers": 1
    }
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["passage", "num_tokens"], ["question", "num_tokens"]],
    "padding_noise": 0.0,
    "batch_size": 40
  },
  "trainer": {
    "num_epochs": 1,
    "grad_norm": 10.0,
    "patience" : 12,
    "cuda_device" : [0, 1],
    "optimizer": {
      "type": "adadelta",
      "lr": 0.5,
      "rho": 0.95
    }
  },
  "gpu_allocations": {
    "trainer": 0,
    "judge": 0,
    "debate": 1
  }
}
