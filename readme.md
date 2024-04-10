This work is part of my Master Thesis.

It focuses on testing two different learnable prompting techniques (Deep and Shallow prompt tuning) on two different Speech Foundation models, Audio Spectrogram Transformers (AST) and Wav2Vec.

The models managed to achieve close to state of the art results on a Spoken Language Understanding task (Intent Classification the FluentSpeech dataset) although they were only trained on less than 1% of the model weights, without lowering the model generalization capabilities, typical of standard fine-tuning approaches.